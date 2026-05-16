// vision3d::iou_box3d schema definition. Backend implementations and their
// STABLE_TORCH_LIBRARY_IMPL registrations live in iou_box3d_cpu.cpp and
// iou_box3d.cu.

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(vision3d, m) {
  m.def(
      "iou_box3d(Tensor boxes1, Tensor boxes2) -> (Tensor, Tensor, Tensor, Tensor)");
  // Backward op that consumes forward-saved state + upstream grads and emits
  // grads on the 8-corner input tensors. Autograd wiring is done from Python
  // via torch.library.register_autograd (the recommended path for stable-ABI
  // custom ops; see PyTorch's "Custom C++ and CUDA Operators" tutorial).
  m.def(
      "iou_box3d_backward(Tensor boxes1, Tensor boxes2, Tensor vol, Tensor face_area, Tensor face_area_centroid, Tensor grad_vol, Tensor grad_iou) -> (Tensor, Tensor)");
}
