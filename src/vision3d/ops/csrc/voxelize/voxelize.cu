// CUDA implementation of vision3d::voxelize via CUB sort + RLE + scan.

#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h> // NOLINT(misc-include-cleaner)
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include "utils/pytorch3d_cutils.h"
#include "voxelize/voxelize.h"

namespace {

// nx*ny*nz fits in 32 bits for any realistic grid
constexpr uint32_t OUT_OF_RANGE = UINT32_MAX;
constexpr int THREADS = 256;

__global__ void compute_cell_ids_kernel(
    const float* __restrict__ points,
    int64_t N,
    int64_t C,
    float x_min,
    float y_min,
    float z_min,
    float x_max,
    float y_max,
    float z_max,
    float dx,
    float dy,
    float dz,
    int64_t nx,
    int64_t ny,
    int64_t nz,
    uint32_t* __restrict__ keys,
    int32_t* __restrict__ indices) {
  const int64_t cells_per_z = ny * nx;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
    indices[i] = static_cast<int32_t>(i);
    const float x = points[i * C + 0];
    const float y = points[i * C + 1];
    const float z = points[i * C + 2];

    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || x < x_min ||
        x >= x_max || y < y_min || y >= y_max || z < z_min || z >= z_max) {
      keys[i] = OUT_OF_RANGE;
      continue;
    }
    auto ix = static_cast<int64_t>((x - x_min) / dx);
    auto iy = static_cast<int64_t>((y - y_min) / dy);
    auto iz = static_cast<int64_t>((z - z_min) / dz);
    if (ix >= nx) {
      ix = nx - 1;
    }
    if (iy >= ny) {
      iy = ny - 1;
    }
    if (iz >= nz) {
      iz = nz - 1;
    }
    keys[i] = static_cast<uint32_t>(iz * cells_per_z + iy * nx + ix);
  }
}

__global__ void scatter_voxels_kernel(
    const float* __restrict__ points,
    int64_t C,
    int64_t max_points_per_voxel,
    int64_t actual_voxels,
    const uint32_t* __restrict__ unique_cells,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ sorted_indices,
    int64_t nx,
    int64_t ny,
    float* __restrict__ voxels,
    int64_t* __restrict__ coords,
    int64_t* __restrict__ num_points) {
  const int64_t cells_per_z = ny * nx;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = blockIdx.x * blockDim.x + threadIdx.x; v < actual_voxels;
       v += stride) {
    const uint32_t cell = unique_cells[v];
    coords[v * 3 + 0] = static_cast<int64_t>(cell / cells_per_z);
    coords[v * 3 + 1] = static_cast<int64_t>((cell / nx) % ny);
    coords[v * 3 + 2] = static_cast<int64_t>(cell % nx);
    const int64_t total = counts[v];
    const int64_t kept =
        total < max_points_per_voxel ? total : max_points_per_voxel;
    num_points[v] = kept;
    const int32_t off = offsets[v];
    for (int64_t k = 0; k < kept; ++k) {
      const int32_t pt = sorted_indices[off + k];
      const float* src = points + static_cast<int64_t>(pt) * C;
      float* dst = voxels + (v * max_points_per_voxel + k) * C;
      for (int64_t c = 0; c < C; ++c) {
        dst[c] = src[c];
      }
    }
  }
}

int blocks_for(int64_t n) {
  const int64_t b = std::clamp<int64_t>((n + THREADS - 1) / THREADS, 1, 4096);
  return static_cast<int>(b);
}

} // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
VoxelizeCuda(
    torch::stable::Tensor points,
    std::vector<double> point_cloud_range,
    std::vector<double> voxel_size,
    int64_t max_points_per_voxel,
    std::optional<int64_t> max_voxels) {
  CHECK_CUDA(points);
  STD_TORCH_CHECK(
      points.scalar_type() == torch::headeronly::ScalarType::Float,
      "points must be float32");
  STD_TORCH_CHECK(
      points.dim() == 2, "points must be 2D [N, C], got ", points.dim(), "D");
  STD_TORCH_CHECK(
      point_cloud_range.size() == 6,
      "point_cloud_range must have 6 elements, got ",
      point_cloud_range.size());
  STD_TORCH_CHECK(
      voxel_size.size() == 3,
      "voxel_size must have 3 elements, got ",
      voxel_size.size());
  STD_TORCH_CHECK(
      max_points_per_voxel > 0, "max_points_per_voxel must be positive");
  STD_TORCH_CHECK(
      !max_voxels.has_value() || *max_voxels > 0,
      "max_voxels must be positive or unset");

  points = torch::stable::contiguous(points);

  const int64_t N = points.size(0);
  const int64_t C = points.size(1);
  // CUB sort/RLE/scan use int32_t element counts; cap N to match.
  STD_TORCH_CHECK(
      N <= static_cast<int64_t>(INT32_MAX),
      "voxelize: more than 2^31-1 input points not supported (got ",
      N,
      ")");

  const auto x_min = static_cast<float>(point_cloud_range[0]);
  const auto y_min = static_cast<float>(point_cloud_range[1]);
  const auto z_min = static_cast<float>(point_cloud_range[2]);
  const auto x_max = static_cast<float>(point_cloud_range[3]);
  const auto y_max = static_cast<float>(point_cloud_range[4]);
  const auto z_max = static_cast<float>(point_cloud_range[5]);
  const auto dx = static_cast<float>(voxel_size[0]);
  const auto dy = static_cast<float>(voxel_size[1]);
  const auto dz = static_cast<float>(voxel_size[2]);
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
  // Match CPU's rounding (round-half-away-from-zero) so both backends
  // agree on grid dims even when (max - min) is not an exact multiple of
  // voxel_size.
  const auto nx = static_cast<int64_t>(std::lround((x_max - x_min) / dx));
  const auto ny = static_cast<int64_t>(std::lround((y_max - y_min) / dy));
  const auto nz = static_cast<int64_t>(std::lround((z_max - z_min) / dz));
  STD_TORCH_CHECK(
      nx * ny * nz < static_cast<int64_t>(OUT_OF_RANGE),
      "voxelize: grid too large, nx*ny*nz must be < UINT32_MAX, got ",
      nx,
      " * ",
      ny,
      " * ",
      nz,
      " (",
      nx * ny * nz,
      ")");

  if (N == 0) {
    auto voxels_empty =
        torch::stable::new_zeros(points, {0, max_points_per_voxel, C});
    auto coords_empty = torch::stable::new_zeros(
        points, {0, 3}, torch::headeronly::ScalarType::Long);
    auto num_points_empty = torch::stable::new_zeros(
        points, {0}, torch::headeronly::ScalarType::Long);
    return std::make_tuple(
        std::move(voxels_empty),
        std::move(coords_empty),
        std::move(num_points_empty));
  }

  const int32_t device_index = points.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  auto* stream = static_cast<cudaStream_t>(raw_stream);

  // Scratch tensors. Going through torch's allocator (via stable's
  // ``new_empty``) means CUB temp storage benefits from caching reuse.
  // ``new_empty`` is safe here because ``compute_cell_ids_kernel`` writes
  // every element of ``keys_in`` and ``idx_in`` (one slot per input point).
  auto keys_in = torch::stable::new_empty(
      points, {N}, torch::headeronly::ScalarType::UInt32);
  auto keys_out = torch::stable::new_empty(
      points, {N}, torch::headeronly::ScalarType::UInt32);
  auto idx_in =
      torch::stable::new_empty(points, {N}, torch::headeronly::ScalarType::Int);
  auto idx_out =
      torch::stable::new_empty(points, {N}, torch::headeronly::ScalarType::Int);

  compute_cell_ids_kernel<<<blocks_for(N), THREADS, 0, stream>>>(
      points.const_data_ptr<float>(),
      N,
      C,
      x_min,
      y_min,
      z_min,
      x_max,
      y_max,
      z_max,
      dx,
      dy,
      dz,
      nx,
      ny,
      nz,
      keys_in.mutable_data_ptr<uint32_t>(),
      idx_in.mutable_data_ptr<int32_t>());
  STD_TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "compute_cell_ids_kernel launch failed");

  // Sort (cell_id, point_idx) pairs by cell_id ascending. Out-of-range
  // sentinels land at the tail.
  size_t sort_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      sort_bytes,
      keys_in.const_data_ptr<uint32_t>(),
      keys_out.mutable_data_ptr<uint32_t>(),
      idx_in.const_data_ptr<int32_t>(),
      idx_out.mutable_data_ptr<int32_t>(),
      static_cast<int>(N),
      0,
      static_cast<int>(sizeof(uint32_t) * 8),
      stream);
  auto sort_temp = torch::stable::new_empty(
      points,
      {static_cast<int64_t>(sort_bytes)},
      torch::headeronly::ScalarType::Byte);
  cub::DeviceRadixSort::SortPairs(
      sort_temp.mutable_data_ptr<uint8_t>(),
      sort_bytes,
      keys_in.const_data_ptr<uint32_t>(),
      keys_out.mutable_data_ptr<uint32_t>(),
      idx_in.const_data_ptr<int32_t>(),
      idx_out.mutable_data_ptr<int32_t>(),
      static_cast<int>(N),
      0,
      static_cast<int>(sizeof(uint32_t) * 8),
      stream);

  // Run-length encode the sorted keys -> (unique_cells, counts, num_runs).
  // The output includes a final OUT_OF_RANGE run for out-of-range points if
  // any were present. Trimmed on host below.
  auto unique_cells = torch::stable::new_empty(
      points, {N}, torch::headeronly::ScalarType::UInt32);
  auto counts =
      torch::stable::new_empty(points, {N}, torch::headeronly::ScalarType::Int);
  auto num_runs =
      torch::stable::new_zeros(points, {1}, torch::headeronly::ScalarType::Int);
  size_t rle_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(
      nullptr,
      rle_bytes,
      keys_out.const_data_ptr<uint32_t>(),
      unique_cells.mutable_data_ptr<uint32_t>(),
      counts.mutable_data_ptr<int32_t>(),
      num_runs.mutable_data_ptr<int32_t>(),
      static_cast<int>(N),
      stream);
  auto rle_temp = torch::stable::new_empty(
      points,
      {static_cast<int64_t>(rle_bytes)},
      torch::headeronly::ScalarType::Byte);
  cub::DeviceRunLengthEncode::Encode(
      rle_temp.mutable_data_ptr<uint8_t>(),
      rle_bytes,
      keys_out.const_data_ptr<uint32_t>(),
      unique_cells.mutable_data_ptr<uint32_t>(),
      counts.mutable_data_ptr<int32_t>(),
      num_runs.mutable_data_ptr<int32_t>(),
      static_cast<int>(N),
      stream);

  // One round-trip to the host: num_runs (the RLE result) plus the largest
  // sorted key. Since OUT_OF_RANGE = UINT32_MAX sorts to the very end,
  // ``keys_out[N - 1]`` tells us whether the final RLE run is the OOR
  // sentinel without a second sync.
  struct HostState {
    int32_t num_runs;
    uint32_t last_sorted_key;
  };
  HostState host{};
  STD_TORCH_CHECK(
      cudaMemcpyAsync(
          &host.num_runs,
          num_runs.const_data_ptr<int32_t>(),
          sizeof(int32_t),
          cudaMemcpyDeviceToHost,
          stream) == cudaSuccess,
      "cudaMemcpyAsync(num_runs) failed");
  STD_TORCH_CHECK(
      cudaMemcpyAsync(
          &host.last_sorted_key,
          keys_out.const_data_ptr<uint32_t>() + (N - 1),
          sizeof(uint32_t),
          cudaMemcpyDeviceToHost,
          stream) == cudaSuccess,
      "cudaMemcpyAsync(last_sorted_key) failed");
  STD_TORCH_CHECK(
      cudaStreamSynchronize(stream) == cudaSuccess,
      "cudaStreamSynchronize failed");

  int64_t voxel_count = host.num_runs;
  if (host.last_sorted_key == OUT_OF_RANGE) {
    --voxel_count; // drop the out-of-range bucket
  }

  if (max_voxels.has_value() && voxel_count > *max_voxels) {
    voxel_count = *max_voxels;
  }

  auto voxels_out =
      torch::stable::new_zeros(points, {voxel_count, max_points_per_voxel, C});
  auto coords_out = torch::stable::new_zeros(
      points, {voxel_count, 3}, torch::headeronly::ScalarType::Long);
  auto num_points_out = torch::stable::new_zeros(
      points, {voxel_count}, torch::headeronly::ScalarType::Long);

  if (voxel_count == 0) {
    return std::make_tuple(
        std::move(voxels_out),
        std::move(coords_out),
        std::move(num_points_out));
  }

  // Exclusive scan on counts to get per-voxel offsets into ``idx_out``.
  auto offsets = torch::stable::new_empty(
      points, {voxel_count}, torch::headeronly::ScalarType::Int);
  size_t scan_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr,
      scan_bytes,
      counts.const_data_ptr<int32_t>(),
      offsets.mutable_data_ptr<int32_t>(),
      static_cast<int>(voxel_count),
      stream);
  auto scan_temp = torch::stable::new_empty(
      points,
      {static_cast<int64_t>(scan_bytes)},
      torch::headeronly::ScalarType::Byte);
  cub::DeviceScan::ExclusiveSum(
      scan_temp.mutable_data_ptr<uint8_t>(),
      scan_bytes,
      counts.const_data_ptr<int32_t>(),
      offsets.mutable_data_ptr<int32_t>(),
      static_cast<int>(voxel_count),
      stream);

  scatter_voxels_kernel<<<blocks_for(voxel_count), THREADS, 0, stream>>>(
      points.const_data_ptr<float>(),
      C,
      max_points_per_voxel,
      voxel_count,
      unique_cells.const_data_ptr<uint32_t>(),
      counts.const_data_ptr<int32_t>(),
      offsets.const_data_ptr<int32_t>(),
      idx_out.const_data_ptr<int32_t>(),
      nx,
      ny,
      voxels_out.mutable_data_ptr<float>(),
      coords_out.mutable_data_ptr<int64_t>(),
      num_points_out.mutable_data_ptr<int64_t>());
  STD_TORCH_CHECK(
      cudaGetLastError() == cudaSuccess, "scatter_voxels_kernel launch failed");

  return std::make_tuple(
      std::move(voxels_out), std::move(coords_out), std::move(num_points_out));
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CUDA, m) {
  m.impl("voxelize", TORCH_BOX(VoxelizeCuda));
}
