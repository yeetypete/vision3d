/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/macros/Macros.h>
#include <tuple>

// Lightweight view of a single (8, 3) box from contiguous float data.
struct BoxView {
  const float* data;

  struct Row {
    const float* p;
    C10_HOST_DEVICE float operator[](int j) const {
      return p[j];
    }
  };

  C10_HOST_DEVICE Row operator[](int i) const {
    return Row{data + i * 3};
  }

  C10_HOST_DEVICE int size(int /*dim*/) const {
    return 8;
  }
};

// Calculate the intersection volume and IoU metric for two batches of boxes.
// Also emits per-input-plane face area + area-weighted centroid of the
// intersection polyhedron — the state needed by the differentiable backward.
//
// Args:
//     boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
//     boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
// Returns:
//     vol: (N, M) tensor of the volume of the intersecting convex shapes
//     iou: (N, M) tensor of the intersection over union which is
//          defined as: `iou = vol / (vol1 + vol2 - vol)`
//     face_area: (N, M, 12) tensor of the area of the intersection polyhedron
//          face that lies on each of the 12 input box planes (planes 0..5 from
//          boxes1, planes 6..11 from boxes2). Zero when the plane does not
//          support the intersection.
//     face_area_centroid: (N, M, 12, 3) tensor of the area-weighted centroid
//          (i.e. area * centroid) of each face. Stored area-weighted so the
//          zero-area case is well-defined without a divide.

// CPU implementation
std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>
IoUBox3DCpu(torch::stable::Tensor boxes1, torch::stable::Tensor boxes2);

// CUDA implementation
std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>
IoUBox3DCuda(torch::stable::Tensor boxes1, torch::stable::Tensor boxes2);
