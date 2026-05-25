#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "utils/pytorch3d_cutils.h"
#include "voxelize/voxelize.h"

namespace {

// Two-pass CPU voxelize:
//   Pass 1 walks every point, computes its (iz, iy, ix) bucket if in range,
//          and assigns it a voxel index via a flat-cell -> voxel-idx hashmap.
//   Pass 2 allocates the output tensors at the now-known P size and writes
//          each point to its voxel slot (capped at max_points_per_voxel,
//          surplus dropped in input order).
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

  const float* pcr = range_c.const_data_ptr<float>();
  const float* vs = size_c.const_data_ptr<float>();
  const float x_min = pcr[0], y_min = pcr[1], z_min = pcr[2];
  const float x_max = pcr[3], y_max = pcr[4], z_max = pcr[5];
  const float dx = vs[0], dy = vs[1], dz = vs[2];
  const int64_t nx = static_cast<int64_t>(std::lround((x_max - x_min) / dx));
  const int64_t ny = static_cast<int64_t>(std::lround((y_max - y_min) / dy));
  const int64_t nz = static_cast<int64_t>(std::lround((z_max - z_min) / dz));
  const int64_t cells_per_z = ny * nx;

  const float* pts = points_c.const_data_ptr<float>();

  // Pass 1: classify each input point.
  std::vector<int32_t> point_to_voxel(N, -1);
  std::unordered_map<int64_t, int32_t> cell_to_voxel;
  std::vector<int64_t> ordered_cells;
  cell_to_voxel.reserve(N / 2);
  ordered_cells.reserve(N / 2);

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
    const int64_t cell = iz * cells_per_z + iy * nx + ix;
    auto it = cell_to_voxel.find(cell);
    int32_t voxel_idx;
    if (it == cell_to_voxel.end()) {
      // Cap on the number of voxels: drop any point that would create a new
      // voxel beyond ``max_voxels``. Points landing in already-claimed cells
      // still flow through pass 2.
      if (max_voxels != -1 &&
          static_cast<int64_t>(ordered_cells.size()) >= max_voxels) {
        continue;
      }
      voxel_idx = static_cast<int32_t>(ordered_cells.size());
      cell_to_voxel.emplace(cell, voxel_idx);
      ordered_cells.push_back(cell);
    } else {
      voxel_idx = it->second;
    }
    point_to_voxel[i] = voxel_idx;
  }

  const int64_t P = static_cast<int64_t>(ordered_cells.size());

  auto voxels =
      torch::stable::new_zeros(points_c, {P, max_points_per_voxel, C});
  auto coords = torch::stable::new_empty(
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
    const int64_t cell = ordered_cells[v];
    coords_data[v * 3 + 0] = cell / cells_per_z;
    coords_data[v * 3 + 1] = (cell / nx) % ny;
    coords_data[v * 3 + 2] = cell % nx;
  }

  // Pass 2: scatter points into voxel slots.
  for (int64_t i = 0; i < N; ++i) {
    const int32_t v = point_to_voxel[i];
    if (v < 0) {
      continue;
    }
    const int64_t slot = num_points_data[v];
    if (slot >= max_points_per_voxel) {
      continue;
    }
    float* dst = voxels_data + (v * max_points_per_voxel + slot) * C;
    const float* src = pts + i * C;
    for (int64_t c = 0; c < C; ++c) {
      dst[c] = src[c];
    }
    num_points_data[v] = slot + 1;
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
