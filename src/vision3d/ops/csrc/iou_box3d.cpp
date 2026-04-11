// vision3d::iou_box3d schema + dispatcher stub. Kernel implementations
// (and their TORCH_LIBRARY_IMPL registrations) live in iou_box3d_cpu.cpp
// and iou_box3d.cu.

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>
#include <tuple>

namespace vision3d {
namespace ops {

std::tuple<at::Tensor, at::Tensor> iou_box3d(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("vision3d::iou_box3d", "")
                       .typed<decltype(iou_box3d)>();
  return op.call(boxes1, boxes2);
}

TORCH_LIBRARY_FRAGMENT(vision3d, m) {
  m.set_python_module("vision3d.ops._meta_registrations");
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vision3d::iou_box3d(Tensor boxes1, Tensor boxes2) -> (Tensor, Tensor)"));
}

} // namespace ops
} // namespace vision3d
