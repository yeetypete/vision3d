/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h> // NOLINT(misc-include-cleaner)
#include <torch/headeronly/util/Exception.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include "iou_box3d/iou_box3d.h"
#include "iou_box3d/iou_utils.cuh"
#include "utils/pytorch3d_cutils.h"

// Parallelize over N*M computations which can each be done
// independently
__global__ void IoUBox3DKernel(
    const float* __restrict__ boxes1,
    const float* __restrict__ boxes2,
    float* __restrict__ vols,
    float* __restrict__ ious,
    int64_t N,
    int64_t M) {
  const int64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  std::array<FaceVerts, NUM_TRIS> box1_tris{};
  std::array<FaceVerts, NUM_TRIS> box2_tris{};
  std::array<FaceVerts, NUM_PLANES> box1_planes{};
  std::array<FaceVerts, NUM_PLANES> box2_planes{};

  for (int64_t i = tid; i < N * M; i += stride) {
    const int64_t n = i / M; // box1 index
    const int64_t m = i % M; // box2 index

    const BoxView box1{boxes1 + n * 8 * 3};
    const BoxView box2{boxes2 + m * 8 * 3};

    // Convert to array of structs of face vertices i.e. effectively (F, 3, 3)
    // FaceVerts is a data type defined in iou_utils.cuh
    GetBoxTris(box1, box1_tris);
    GetBoxTris(box2, box2_tris);

    // Calculate the position of the center of the box which is used in
    // several calculations.
    const float3 box1_center = BoxCenter(box1);
    const float3 box2_center = BoxCenter(box2);

    // Convert to an array of face vertices
    GetBoxPlanes(box1, box1_planes);
    GetBoxPlanes(box2, box2_planes);

    // Get Box Volumes
    const float box1_vol = BoxVolume(box1_tris, box1_center, NUM_TRIS);
    const float box2_vol = BoxVolume(box2_tris, box2_center, NUM_TRIS);

    // Tris in Box1 intersection with Planes in Box2
    // Initialize box1 intersecting faces. MAX_TRIS is the
    // max faces possible in the intersecting shape.
    // TODO: determine if the value of MAX_TRIS is sufficient or
    // if we should store the max tris for each NxM computation
    // and throw an error if any exceeds the max.
    std::array<FaceVerts, MAX_TRIS> box1_intersect{};
    for (int j = 0; j < NUM_TRIS; ++j) {
      // Initialize the faces from the box
      box1_intersect[j] = box1_tris[j];
    }
    // Get the count of the actual number of faces in the intersecting shape
    int box1_count = BoxIntersections(box2_planes, box2_center, box1_intersect);

    // Tris in Box2 intersection with Planes in Box1
    std::array<FaceVerts, MAX_TRIS> box2_intersect{};
    for (int j = 0; j < NUM_TRIS; ++j) {
      box2_intersect[j] = box2_tris[j];
    }
    const int box2_count =
        BoxIntersections(box1_planes, box1_center, box2_intersect);

    // If there are overlapping regions in Box2, remove any coplanar faces
    if (box2_count > 0) {
      // Identify if any triangles in Box2 are coplanar with Box1
      std::array<Keep, MAX_TRIS> tri2_keep{};
      for (int j = 0; j < MAX_TRIS; ++j) {
        // Initialize the valid faces to be true
        tri2_keep[j].keep = j < box2_count;
      }
      for (int b1 = 0; b1 < box1_count; ++b1) {
        for (int b2 = 0; b2 < box2_count; ++b2) {
          const bool is_coplanar =
              IsCoplanarTriTri(box1_intersect[b1], box2_intersect[b2]);
          const float area = FaceArea(box1_intersect[b1]);
          if (is_coplanar && area > aEpsilon) {
            tri2_keep[b2].keep = false;
          }
        }
      }

      // Keep only the non coplanar triangles in Box2 - add them to the
      // Box1 triangles.
      for (int b2 = 0; b2 < box2_count; ++b2) {
        if (tri2_keep[b2].keep) {
          box1_intersect[box1_count] = box2_intersect[b2];
          // box1_count will determine the total faces in the
          // intersecting shape
          box1_count++;
        }
      }
    }

    // Initialize the vol and iou to 0.0 in case there are no triangles
    // in the intersecting shape.
    float vol = 0.0;
    float iou = 0.0;

    // If there are triangles in the intersecting shape
    if (box1_count > 0) {
      // The intersecting shape is a polyhedron made up of the
      // triangular faces that are all now in box1_intersect.
      // Calculate the polyhedron center
      const float3 poly_center = PolyhedronCenter(box1_intersect, box1_count);
      // Compute intersecting polyhedron volume
      vol = BoxVolume(box1_intersect, poly_center, box1_count);
      // Compute IoU
      iou = vol / (box1_vol + box2_vol - vol);
    }

    // Write the volume and IoU to global memory
    vols[n * M + m] = vol;
    ious[n * M + m] = iou;
  }
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> IoUBox3DCuda(
    torch::stable::Tensor boxes1, // (N, 8, 3)
    torch::stable::Tensor boxes2) { // (M, 8, 3)
  // Check inputs are on the same device
  CHECK_CUDA(boxes1);
  CHECK_CUDA(boxes2);
  STD_TORCH_CHECK(
      boxes1.get_device_index() == boxes2.get_device_index(),
      "boxes1 and boxes2 must be on the same CUDA device");
  STD_TORCH_CHECK(
      boxes1.scalar_type() == boxes2.scalar_type(),
      "boxes1 and boxes2 must have the same dtype");
  STD_TORCH_CHECK(
      boxes1.dim() == 3 && boxes1.size(1) == 8 && boxes1.size(2) == 3,
      "boxes1 must have shape (N, 8, 3)");
  STD_TORCH_CHECK(
      boxes2.dim() == 3 && boxes2.size(1) == 8 && boxes2.size(2) == 3,
      "boxes2 must have shape (M, 8, 3)");

  boxes1 = torch::stable::contiguous(boxes1);
  boxes2 = torch::stable::contiguous(boxes2);

  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);

  auto vols = torch::stable::new_zeros(boxes1, {N, M});
  auto ious = torch::stable::new_zeros(boxes1, {N, M});

  if (N == 0 || M == 0) {
    return std::make_tuple(std::move(vols), std::move(ious));
  }

  // Set the device for the kernel launch based on the device of boxes1
  const int32_t device_index = boxes1.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);

  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  auto* stream = static_cast<cudaStream_t>(raw_stream);

  const size_t blocks = 512;
  const size_t threads = 256;

  IoUBox3DKernel<<<blocks, threads, 0, stream>>>(
      boxes1.const_data_ptr<float>(),
      boxes2.const_data_ptr<float>(),
      vols.mutable_data_ptr<float>(),
      ious.mutable_data_ptr<float>(),
      N,
      M);

  STD_TORCH_CHECK(
      cudaGetLastError() == cudaSuccess, "IoUBox3DKernel launch failed");

  return std::make_tuple(std::move(vols), std::move(ious));
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CUDA, m) {
  m.impl("iou_box3d", TORCH_BOX(IoUBox3DCuda));
}
