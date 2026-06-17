// Bucket points into a 3D voxel grid (pillar or true voxel). See the
// ``voxelize`` docstring in ``_voxelize.py`` for the full documentation.
//
// Args:
//     points: [N, C] float32 point cloud, columns 0..2 are (x, y, z).
//     point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max).
//     voxel_size: (dx, dy, dz) voxel size.
//     max_points_per_voxel: Cap on points stored per voxel.
//     max_voxels: Optional cap on the number of output voxels.
// Returns:
//     (voxels, coords, num_points): voxels [P, max_points_per_voxel, C]
//     float32, coords [P, 3] int64 (iz, iy, ix), num_points [P] int64. P
//     is the number of non-empty voxels.

#pragma once

#include <torch/csrc/stable/tensor.h>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
VoxelizeCpu(
    torch::stable::Tensor points,
    std::vector<double> point_cloud_range,
    std::vector<double> voxel_size,
    int64_t max_points_per_voxel,
    std::optional<int64_t> max_voxels);

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
VoxelizeCuda(
    torch::stable::Tensor points,
    std::vector<double> point_cloud_range,
    std::vector<double> voxel_size,
    int64_t max_points_per_voxel,
    std::optional<int64_t> max_voxels);
