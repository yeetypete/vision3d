// Stub CUDA implementation. The real kernels will land in the next commit;
// for now we register a STABLE_TORCH_LIBRARY_IMPL that throws so the
// dispatcher knows about the CUDA backend and the build succeeds.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <tuple>
#include "voxelize/voxelize.h"

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
VoxelizeCuda(
    torch::stable::Tensor /*points*/,
    torch::stable::Tensor /*point_cloud_range*/,
    torch::stable::Tensor /*voxel_size*/,
    int64_t /*max_points_per_voxel*/,
    int64_t /*max_voxels*/) {
  STD_TORCH_CHECK(false, "vision3d::voxelize CUDA kernel not yet implemented");
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CUDA, m) {
  m.impl("voxelize", TORCH_BOX(VoxelizeCuda));
}
