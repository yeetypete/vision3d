// vision3d::voxelize schema definition. Backend implementations and their
// STABLE_TORCH_LIBRARY_IMPL registrations live in voxelize/voxelize_cpu.cpp and
// voxelize/voxelize.cu.

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(vision3d, m) {
  m.def(
      "voxelize(Tensor points, Tensor point_cloud_range, Tensor voxel_size, int max_points_per_voxel, int? max_voxels) -> (Tensor, Tensor, Tensor)");
}
