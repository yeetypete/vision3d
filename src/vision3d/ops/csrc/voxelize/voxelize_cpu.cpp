#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>
#include "utils/pytorch3d_cutils.h"
#include "voxelize/voxelize.h"

namespace {

// Sort-based CPU voxelize. Matches the CUDA implementation's output
// ordering (voxel rows sorted by ascending flat cell id).
//   1. Compute flat cell id per point. Drop out-of-range points.
//   2. Sort the kept (cell_id, point_idx) pairs by cell_id (stable).
//   3. Walk the sorted pairs to dedup into voxels, capped by max_voxels
//      and max_points_per_voxel.
//   4. Allocate outputs and write.
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
voxelize_cpu_impl(
    const torch::stable::Tensor& points,
    const torch::stable::Tensor& point_cloud_range,
    const torch::stable::Tensor& voxel_size,
    int64_t max_points_per_voxel,
    int64_t max_voxels) {
  CHECK_CPU(points);
  CHECK_CPU(point_cloud_range);
  CHECK_CPU(voxel_size);
  STD_TORCH_CHECK(
      points.scalar_type() == torch::headeronly::ScalarType::Float,
      "points must be float32");
  STD_TORCH_CHECK(
      point_cloud_range.scalar_type() == torch::headeronly::ScalarType::Float,
      "point_cloud_range must be float32");
  STD_TORCH_CHECK(
      voxel_size.scalar_type() == torch::headeronly::ScalarType::Float,
      "voxel_size must be float32");
  STD_TORCH_CHECK(
      points.dim() == 2, "points must be 2D [N, C], got ", points.dim(), "D");
  STD_TORCH_CHECK(
      point_cloud_range.numel() == 6, "point_cloud_range must have 6 elements");
  STD_TORCH_CHECK(voxel_size.numel() == 3, "voxel_size must have 3 elements");
  STD_TORCH_CHECK(
      max_points_per_voxel > 0, "max_points_per_voxel must be positive");
  STD_TORCH_CHECK(
      max_voxels == -1 || max_voxels > 0,
      "max_voxels must be -1 (no cap) or positive");

  auto points_c = torch::stable::contiguous(points);
  auto range_c = torch::stable::contiguous(point_cloud_range);
  auto size_c = torch::stable::contiguous(voxel_size);

  const int64_t N = points_c.size(0);
  const int64_t C = points_c.size(1);
  // The dedup walk stores per-point indices as int32_t to keep memory low and
  // match the CUDA backend. Cap N accordingly.
  STD_TORCH_CHECK(
      N <= static_cast<int64_t>(INT32_MAX),
      "voxelize: more than 2^31-1 input points not supported (got ",
      N,
      ")");

  const float* pcr = range_c.const_data_ptr<float>();
  const float* vs = size_c.const_data_ptr<float>();
  const float x_min = pcr[0], y_min = pcr[1], z_min = pcr[2];
  const float x_max = pcr[3], y_max = pcr[4], z_max = pcr[5];
  const float dx = vs[0], dy = vs[1], dz = vs[2];
  STD_TORCH_CHECK(
      dx > 0.0f && dy > 0.0f && dz > 0.0f,
      "voxel_size must be positive on every axis, got (",
      dx,
      ", ",
      dy,
      ", ",
      dz,
      ")");
  STD_TORCH_CHECK(
      x_max > x_min && y_max > y_min && z_max > z_min,
      "point_cloud_range must have max > min on every axis, got (",
      x_min,
      ", ",
      y_min,
      ", ",
      z_min,
      ") .. (",
      x_max,
      ", ",
      y_max,
      ", ",
      z_max,
      ")");
  const int64_t nx = static_cast<int64_t>(std::lround((x_max - x_min) / dx));
  const int64_t ny = static_cast<int64_t>(std::lround((y_max - y_min) / dy));
  const int64_t nz = static_cast<int64_t>(std::lround((z_max - z_min) / dz));
  const int64_t cells_per_z = ny * nx;

  const float* pts = points_c.const_data_ptr<float>();

  // (cell_id, point_idx) pairs for in-range points.
  std::vector<std::pair<int64_t, int32_t>> sorted;
  sorted.reserve(N);
  for (int64_t i = 0; i < N; ++i) {
    const float x = pts[i * C + 0];
    const float y = pts[i * C + 1];
    const float z = pts[i * C + 2];
    if (x < x_min || x >= x_max || y < y_min || y >= y_max || z < z_min ||
        z >= z_max) {
      continue;
    }
    int64_t ix = static_cast<int64_t>((x - x_min) / dx);
    int64_t iy = static_cast<int64_t>((y - y_min) / dy);
    int64_t iz = static_cast<int64_t>((z - z_min) / dz);
    if (ix >= nx)
      ix = nx - 1;
    if (iy >= ny)
      iy = ny - 1;
    if (iz >= nz)
      iz = nz - 1;
    sorted.emplace_back(
        iz * cells_per_z + iy * nx + ix, static_cast<int32_t>(i));
  }
  std::stable_sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) {
    return a.first < b.first;
  });

  // Dedup walk: build per-voxel point index lists, capped by max_voxels.
  std::vector<int64_t> unique_cells;
  std::vector<std::vector<int32_t>> voxel_points;
  int64_t prev_cell = -1;
  for (const auto& [cell, pt] : sorted) {
    if (cell != prev_cell) {
      if (max_voxels >= 0 &&
          static_cast<int64_t>(unique_cells.size()) >= max_voxels) {
        break;
      }
      unique_cells.push_back(cell);
      voxel_points.emplace_back();
      prev_cell = cell;
    }
    auto& bucket = voxel_points.back();
    if (static_cast<int64_t>(bucket.size()) < max_points_per_voxel) {
      bucket.push_back(pt);
    }
  }

  const int64_t P = static_cast<int64_t>(unique_cells.size());
  auto voxels =
      torch::stable::new_zeros(points_c, {P, max_points_per_voxel, C});
  auto coords = torch::stable::new_zeros(
      points_c, {P, 3}, torch::headeronly::ScalarType::Long);
  auto num_points = torch::stable::new_zeros(
      points_c, {P}, torch::headeronly::ScalarType::Long);

  if (P == 0) {
    return std::make_tuple(
        std::move(voxels), std::move(coords), std::move(num_points));
  }

  float* voxels_data = voxels.mutable_data_ptr<float>();
  int64_t* coords_data = coords.mutable_data_ptr<int64_t>();
  int64_t* num_points_data = num_points.mutable_data_ptr<int64_t>();

  for (int64_t v = 0; v < P; ++v) {
    const int64_t cell = unique_cells[v];
    coords_data[v * 3 + 0] = cell / cells_per_z;
    coords_data[v * 3 + 1] = (cell / nx) % ny;
    coords_data[v * 3 + 2] = cell % nx;
    const auto& bucket = voxel_points[v];
    num_points_data[v] = static_cast<int64_t>(bucket.size());
    for (int64_t slot = 0; slot < static_cast<int64_t>(bucket.size()); ++slot) {
      const float* src = pts + static_cast<int64_t>(bucket[slot]) * C;
      float* dst = voxels_data + (v * max_points_per_voxel + slot) * C;
      for (int64_t c = 0; c < C; ++c) {
        dst[c] = src[c];
      }
    }
  }

  return std::make_tuple(
      std::move(voxels), std::move(coords), std::move(num_points));
}

} // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
VoxelizeCpu(
    torch::stable::Tensor points,
    torch::stable::Tensor point_cloud_range,
    torch::stable::Tensor voxel_size,
    int64_t max_points_per_voxel,
    int64_t max_voxels) {
  return voxelize_cpu_impl(
      points, point_cloud_range, voxel_size, max_points_per_voxel, max_voxels);
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CPU, m) {
  m.impl("voxelize", TORCH_BOX(VoxelizeCpu));
}
