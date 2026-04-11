// vision3d custom-op registration.
//
// Exposes the PyTorch3D-derived iou_box3d kernel as a first-class PyTorch
// operator under the ``vision3d`` namespace, using the torchvision-style
// TORCH_LIBRARY registration pattern.
//
// The underlying algorithm (Sutherland–Hodgman polyhedron clipping +
// divergence-theorem volume) was copied verbatim from PyTorch3D under the
// BSD-3 license. See LICENSE-pytorch3d at the repo root and the file
// headers in ``iou_box3d/`` for details.

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>
#include <tuple>

#include "iou_box3d/iou_box3d.h"

namespace vision3d {
namespace ops {

// Dispatcher entry point. Looks up the registered kernel via PyTorch's
// operator dispatcher and calls it. Separating this from the kernel
// itself lets TORCH_LIBRARY_IMPL wire device-specific implementations
// (CPU and CUDA) in independently.
std::tuple<at::Tensor, at::Tensor> iou_box3d(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("vision3d::iou_box3d", "")
                       .typed<decltype(iou_box3d)>();
  return op.call(boxes1, boxes2);
}

TORCH_LIBRARY_FRAGMENT(vision3d, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vision3d::iou_box3d(Tensor boxes1, Tensor boxes2) -> (Tensor, Tensor)"));
}

TORCH_LIBRARY_IMPL(vision3d, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("vision3d::iou_box3d"), TORCH_FN(IoUBox3DCpu));
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(vision3d, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("vision3d::iou_box3d"), TORCH_FN(IoUBox3DCuda));
}
#endif

} // namespace ops
} // namespace vision3d
