// vision3d::voxelize: bucket points into a 3D voxel grid on the X/Y/Z axes.
//
// Inputs:
//     points: [N, C] float32 point cloud (first 3 columns are x, y, z).
//     point_cloud_range: [6] float32 (x_min, y_min, z_min, x_max, y_max,
//         z_max).
//     voxel_size: [3] float32 (dx, dy, dz). For PointPillars-style
//         pillars set dz = (z_max - z_min) so the grid has a single
//         z-slice and every kept point lands at iz = 0.
//     max_points_per_voxel: cap on per-voxel point count (excess dropped in
//         input order).
//     max_voxels: cap on the number of output voxels, or -1 for no cap.
//         When the cap bites, points that would have created a new voxel
//         beyond ``max_voxels`` are silently dropped. Points landing in
//         already-allocated voxels still get written.
//
// Outputs:
//     voxels: [P, max_points_per_voxel, C] float32 per-voxel point buffers
//         zero-padded at unfilled slots.
//     coords: [P, 3] int64 (iz, iy, ix) voxel indices.
//     num_points: [P] int64 point counts per voxel, each in
//         [1, max_points_per_voxel].
//
// P is the number of non-empty voxels and depends on the input points. When
// no points lie inside ``point_cloud_range``, the leading dimension of every
// output is 0.

#pragma once

#include <torch/csrc/stable/tensor.h>
#include <tuple>

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
VoxelizeCpu(
    torch::stable::Tensor points,
    torch::stable::Tensor point_cloud_range,
    torch::stable::Tensor voxel_size,
    int64_t max_points_per_voxel,
    int64_t max_voxels);

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
VoxelizeCuda(
    torch::stable::Tensor points,
    torch::stable::Tensor point_cloud_range,
    torch::stable::Tensor voxel_size,
    int64_t max_points_per_voxel,
    int64_t max_voxels);
