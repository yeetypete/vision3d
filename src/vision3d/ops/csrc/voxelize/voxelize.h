// vision3d::voxelize: bucket points into a 3D voxel grid (pillar or true
// voxel).
//
// For PointPillars-style pillars set voxel_size = (dx, dy, z_max - z_min) so
// the grid has a single z-slice and every kept point lands at iz = 0. For 3D
// voxel detectors (VoxelNet, SECOND, CenterPoint-Voxel) use a normal
// three-axis voxel_size.
//
// The op is not differentiable.
//
// Grid dimensions are computed as round((max - min) / voxel_size) with ties
// going away from zero. Ranges that are not an exact multiple of voxel_size
// round to the nearest integer cell count. Points landing in the partial
// trailing cell are clamped to the final grid index. Points with non-finite
// coordinates (NaN, +/-Inf) are treated as out-of-range and dropped.
//
// Args:
//     points: [N, C] float32 point cloud. The first three columns must be
//         (x, y, z). Remaining columns may be arbitrary per-point features.
//         Points outside point_cloud_range along any axis are discarded.
//     point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max).
//     voxel_size: (dx, dy, dz) voxel size.
//     max_points_per_voxel: Cap on points stored per voxel. Surplus points
//         are dropped in input order.
//     max_voxels: Optional cap on the number of output voxels. std::nullopt
//         means no cap. When set, points that would have created a new voxel
//         beyond max_voxels are silently dropped. Points landing in
//         already-allocated voxels still get written.
//
// Returns:
//     (voxels, coords, num_points) where voxels is
//     [P, max_points_per_voxel, C] per-voxel point buffers padded with zeros
//     at unfilled slots, coords is [P, 3] int64 (iz, iy, ix) voxel indices,
//     and num_points is [P] int64 point counts per voxel, each in
//     [1, max_points_per_voxel]. P is the number of non-empty voxels
//     (capped at max_voxels when set). When no points fall inside
//     point_cloud_range, all three tensors have a leading dimension of zero.

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
