/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <tuple>
#include "iou_box3d/iou_box3d.h"
#include "iou_box3d/iou_utils.h"
#include "utils/pytorch3d_cutils.h"

namespace {

// Accumulate per-source-plane area and area-weighted centroid for one clipped
// triangle. `plane_offset` is added to the source-plane label so that triangles
// from box1 (label in [0, 6)) and box2 (label in [0, 6) + 6 = [6, 12)) write
// into disjoint slots of the per-pair output buffer. `weight` is 1.0 for a
// triangle without a coplanar partner in the other box, or 0.5 when the
// triangle is one half of a coplanar pair — splitting the attribution
// gives the symmetric subgradient at exact-coplanar configurations.
inline void AccumulateTri(
    const std::vector<vec3<float>>& tri,
    int label,
    int plane_offset,
    float weight,
    float* face_area_row,
    float* face_area_centroid_row) {
  const float area = FaceArea(tri);
  if (area <= 0.0f) {
    return;
  }
  const vec3<float> centroid = (tri[0] + tri[1] + tri[2]) / 3.0f;
  const int p = label + plane_offset;
  const float wA = weight * area;
  face_area_row[p] += wA;
  face_area_centroid_row[p * 3 + 0] += wA * centroid.x;
  face_area_centroid_row[p * 3 + 1] += wA * centroid.y;
  face_area_centroid_row[p * 3 + 2] += wA * centroid.z;
}

} // namespace

std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>
IoUBox3DCpu(torch::stable::Tensor boxes1, torch::stable::Tensor boxes2) {
  CHECK_CPU(boxes1);
  CHECK_CPU(boxes2);
  boxes1 = torch::stable::contiguous(boxes1);
  boxes2 = torch::stable::contiguous(boxes2);

  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);

  auto vols = torch::stable::new_zeros(boxes1, {N, M});
  auto ious = torch::stable::new_zeros(boxes1, {N, M});
  // Per-input-plane state for the analytic backward. Planes 0..5 come from
  // boxes1, planes 6..11 come from boxes2.
  auto face_area = torch::stable::new_zeros(boxes1, {N, M, 12});
  auto face_area_centroid = torch::stable::new_zeros(boxes1, {N, M, 12, 3});

  const float* boxes1_data = boxes1.const_data_ptr<float>();
  const float* boxes2_data = boxes2.const_data_ptr<float>();
  float* vols_data = vols.mutable_data_ptr<float>();
  float* ious_data = ious.mutable_data_ptr<float>();
  float* face_area_data = face_area.mutable_data_ptr<float>();
  float* face_area_centroid_data = face_area_centroid.mutable_data_ptr<float>();

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

      // Initialize per-triangle source-plane labels from _TRI_TO_PLANE.
      std::vector<int> tri1_labels(NUM_TRIS);
      std::vector<int> tri2_labels(NUM_TRIS);
      for (int t = 0; t < NUM_TRIS; ++t) {
        tri1_labels[t] = _TRI_TO_PLANE[t];
        tri2_labels[t] = _TRI_TO_PLANE[t];
      }

      // Tris in Box1 -> Planes in Box2
      face_verts box1_intersect = BoxIntersectionsLabeled(
          box1_tris, box2_planes, box2_center, tri1_labels);
      // Tris in Box2 -> Planes in Box1
      face_verts box2_intersect = BoxIntersectionsLabeled(
          box2_tris, box1_planes, box1_center, tri2_labels);

      // Detect coplanar pairs and mark partner weights. A triangle without a
      // partner stays at weight 1; a triangle paired with one (or more) in the
      // other box gets weight 0.5. Volume math still drops coplanar box2 tris
      // (since the same physical face would otherwise double-count); but for
      // face_area attribution, the half-and-half split gives the symmetric
      // subgradient at exact-coplanar configurations.
      std::vector<float> weight1(box1_intersect.size(), 1.0f);
      std::vector<float> weight2(box2_intersect.size(), 1.0f);
      if (box2_intersect.size() > 0) {
        for (size_t b1 = 0; b1 < box1_intersect.size(); ++b1) {
          const float area = FaceArea(box1_intersect[b1]);
          if (area <= aEpsilon) {
            continue;
          }
          for (size_t b2 = 0; b2 < box2_intersect.size(); ++b2) {
            if (IsCoplanarTriTri(box1_intersect[b1], box2_intersect[b2])) {
              weight1[b1] = 0.5f;
              weight2[b2] = 0.5f;
            }
          }
        }
      }

      // Build the merged-for-volume list: all box1 tris + box2 tris without a
      // coplanar partner. Matches the previous dedup semantics for vol.
      face_verts merged = box1_intersect;
      for (size_t b2 = 0; b2 < box2_intersect.size(); ++b2) {
        if (weight2[b2] == 1.0f) {
          merged.push_back(box2_intersect[b2]);
        }
      }

      // Initialize the vol and iou to 0.0 in case there are no triangles
      // in the intersecting shape.
      float vol = 0.0;
      float iou = 0.0;

      // If there are triangles in the intersecting shape
      if (merged.size() > 0) {
        // The intersecting shape is a polyhedron made up of the
        // triangular faces in `merged`.
        const vec3<float> polyhedron_center = PolyhedronCenter(merged);
        // Compute intersecting polyhedron volume
        vol = BoxVolume(merged, polyhedron_center);
        // Compute IoU
        iou = vol / (box1_vol + box2_vol - vol);

        // Accumulate per-input-plane face area and area-weighted centroid.
        // Iterate both clipped lists (not the merged one): box1's labels are
        // in [0, NUM_PLANES); box2's labels get +NUM_PLANES applied here.
        // Coplanar-pair triangles contribute weight 0.5 from each side, so
        // the total face area on a shared plane is still A but split A/2
        // between the two source planes.
        float* face_area_row = face_area_data + (n * M + m) * 12;
        float* face_area_centroid_row =
            face_area_centroid_data + (n * M + m) * 12 * 3;
        for (size_t k = 0; k < box1_intersect.size(); ++k) {
          AccumulateTri(
              box1_intersect[k],
              tri1_labels[k],
              /*plane_offset=*/0,
              weight1[k],
              face_area_row,
              face_area_centroid_row);
        }
        for (size_t k = 0; k < box2_intersect.size(); ++k) {
          AccumulateTri(
              box2_intersect[k],
              tri2_labels[k],
              /*plane_offset=*/NUM_PLANES,
              weight2[k],
              face_area_row,
              face_area_centroid_row);
        }
      }
      // Save out volume and IoU
      vols_data[n * M + m] = vol;
      ious_data[n * M + m] = iou;
    }
  }
  return std::make_tuple(
      std::move(vols),
      std::move(ious),
      std::move(face_area),
      std::move(face_area_centroid));
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CPU, m) {
  m.impl("iou_box3d", TORCH_BOX(IoUBox3DCpu));
}
