/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>
#include "iou_box3d/iou_box3d.h"
#include "iou_box3d/iou_utils.h"
#include "utils/pytorch3d_cutils.h"
#include "utils/vec3.h"

std::tuple<torch::stable::Tensor, torch::stable::Tensor> IoUBox3DCpu(
    torch::stable::Tensor boxes1,
    torch::stable::Tensor boxes2) {
  CHECK_CPU(boxes1);
  CHECK_CPU(boxes2);
  boxes1 = torch::stable::contiguous(boxes1);
  boxes2 = torch::stable::contiguous(boxes2);

  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);

  auto vols = torch::stable::new_zeros(boxes1, {N, M});
  auto ious = torch::stable::new_zeros(boxes1, {N, M});

  const float* boxes1_data = boxes1.const_data_ptr<float>();
  const float* boxes2_data = boxes2.const_data_ptr<float>();
  float* vols_data = vols.mutable_data_ptr<float>();
  float* ious_data = ious.mutable_data_ptr<float>();

  // Iterate through the N boxes in boxes1
  for (int64_t n = 0; n < N; ++n) {
    const BoxView box1{boxes1_data + n * 8 * 3};
    // Convert to vector of face vertices i.e. effectively (F, 3, 3)
    // face_verts is a data type defined in iou_utils.h
    const face_verts box1_tris = GetBoxTris(box1);

    // Calculate the position of the center of the box which is used in
    // several calculations.
    const vec3<float> box1_center = BoxCenter(box1);

    // Convert to vector of face vertices i.e. effectively (P, 4, 3)
    const face_verts box1_planes = GetBoxPlanes(box1);

    // Get Box Volumes
    const float box1_vol = BoxVolume(box1_tris, box1_center);

    // Iterate through the M boxes in boxes2
    for (int64_t m = 0; m < M; ++m) {
      // Repeat above steps for box2
      // TODO: check if caching these value helps performance.
      const BoxView box2{boxes2_data + m * 8 * 3};
      const face_verts box2_tris = GetBoxTris(box2);
      const vec3<float> box2_center = BoxCenter(box2);
      const face_verts box2_planes = GetBoxPlanes(box2);
      const float box2_vol = BoxVolume(box2_tris, box2_center);

      // Every triangle in one box will be compared to each plane in the other
      // box. There are 3 possible outcomes:
      // 1. If the triangle is fully inside, then it will
      //    remain as is.
      // 2. If the triagnle it is fully outside, it will be removed.
      // 3. If the triangle intersects with the (infinite) plane, it
      //    will be broken into subtriangles such that each subtriangle is full
      //    inside the plane and part of the intersecting tetrahedron.

      // Tris in Box1 -> Planes in Box2
      face_verts box1_intersect =
          BoxIntersections(box1_tris, box2_planes, box2_center);
      // Tris in Box2 -> Planes in Box1
      face_verts box2_intersect =
          BoxIntersections(box2_tris, box1_planes, box1_center);

      // If there are overlapping regions in Box2, remove any coplanar faces
      if (!box2_intersect.empty()) {
        // Identify if any triangles in Box2 are coplanar with Box1
        std::vector<int> tri2_keep(box2_intersect.size());
        std::ranges::fill(tri2_keep, 1);
        for (const auto& b1 : box1_intersect) {
          for (size_t b2 = 0; b2 < box2_intersect.size(); ++b2) {
            const bool is_coplanar = IsCoplanarTriTri(b1, box2_intersect[b2]);
            const float area = FaceArea(b1);
            if (is_coplanar && area > aEpsilon) {
              tri2_keep[b2] = 0;
            }
          }
        }

        // Keep only the non coplanar triangles in Box2 - add them to the
        // Box1 triangles.
        for (size_t b2 = 0; b2 < box2_intersect.size(); ++b2) {
          if (tri2_keep[b2] == 1) {
            box1_intersect.push_back(box2_intersect[b2]);
          }
        }
      }

      // Initialize the vol and iou to 0.0 in case there are no triangles
      // in the intersecting shape.
      float vol = 0.0;
      float iou = 0.0;

      // If there are triangles in the intersecting shape
      if (!box1_intersect.empty()) {
        // The intersecting shape is a polyhedron made up of the
        // triangular faces that are all now in box1_intersect.
        // Calculate the polyhedron center
        const vec3<float> polyhedron_center = PolyhedronCenter(box1_intersect);
        // Compute intersecting polyhedron volume
        vol = BoxVolume(box1_intersect, polyhedron_center);
        // Compute IoU
        iou = vol / (box1_vol + box2_vol - vol);
      }
      // Save out volume and IoU
      vols_data[n * M + m] = vol;
      ious_data[n * M + m] = iou;
    }
  }
  return std::make_tuple(std::move(vols), std::move(ious));
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CPU, m) {
  m.impl("iou_box3d", TORCH_BOX(IoUBox3DCpu));
}
