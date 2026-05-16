/*
 * Analytic backward for iou_box3d (CPU).
 *
 * Stub: returns zero-filled grads on the 8-corner inputs. The math is
 * filled in by a follow-up commit.
 */

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <tuple>
#include "iou_box3d/iou_box3d.h"
#include "utils/pytorch3d_cutils.h"

std::tuple<torch::stable::Tensor, torch::stable::Tensor> IoUBox3DBackwardCpu(
    torch::stable::Tensor boxes1,
    torch::stable::Tensor boxes2,
    torch::stable::Tensor /*vol*/,
    torch::stable::Tensor /*face_area*/,
    torch::stable::Tensor /*face_area_centroid*/,
    torch::stable::Tensor /*grad_vol*/,
    torch::stable::Tensor /*grad_iou*/) {
  CHECK_CPU(boxes1);
  CHECK_CPU(boxes2);
  STD_TORCH_CHECK(
      boxes1.dim() == 3 && boxes1.size(1) == 8 && boxes1.size(2) == 3,
      "boxes1 must have shape (N, 8, 3)");
  STD_TORCH_CHECK(
      boxes2.dim() == 3 && boxes2.size(1) == 8 && boxes2.size(2) == 3,
      "boxes2 must have shape (M, 8, 3)");

  auto grad_boxes1 = torch::stable::new_zeros(boxes1, {boxes1.size(0), 8, 3});
  auto grad_boxes2 = torch::stable::new_zeros(boxes2, {boxes2.size(0), 8, 3});
  return std::make_tuple(std::move(grad_boxes1), std::move(grad_boxes2));
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CPU, m) {
  m.impl("iou_box3d_backward", TORCH_BOX(IoUBox3DBackwardCpu));
}
