// vision3d::iou_box3d schema definition. Backend implementations and their
// STABLE_TORCH_LIBRARY_IMPL registrations live in iou_box3d_cpu.cpp and
// iou_box3d.cu.

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(vision3d, m) {
  m.def(
      "iou_box3d(Tensor boxes1, Tensor boxes2) -> (Tensor, Tensor, Tensor, Tensor)");
}
