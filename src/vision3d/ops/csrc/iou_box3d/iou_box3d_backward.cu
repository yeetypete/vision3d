/*
 * Analytic backward for iou_box3d (CUDA).
 *
 * Mirrors the CPU backward (iou_box3d_backward_cpu.cpp): for each (n, m)
 * pair, computes per-plane gradient contributions via the closed-form
 *     dV/dd_p = A_p,    dV/dn_p = -A_p * c_p
 * and chains them back to the 8-corner inputs. Per-box-volume V1, V2
 * gradients are computed via signed-tetrahedron decomposition.
 *
 * Parallelism: one thread per (n, m) pair. Atomic adds accumulate into
 * grad_boxes1[n] and grad_boxes2[m] since multiple (n, m) threads share
 * the same n or m. V1 and V2 are recomputed per-thread; this is a small
 * amount of redundant work (M evaluations per n) but keeps the kernel
 * single-pass.
 */

#include <math.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <tuple>
#include "iou_box3d/iou_box3d.h"
#include "iou_box3d/iou_utils.cuh"
#include "utils/pytorch3d_cutils.h"

namespace {

constexpr float kIoUDenomEps = 1e-12f;

__device__ inline void atomicAddFloat3(float* dst, const float3& v) {
  atomicAdd(dst + 0, v.x);
  atomicAdd(dst + 1, v.y);
  atomicAdd(dst + 2, v.z);
}

__device__ inline float3 BoxCenterFromCorners(const float3 corners[8]) {
  float3 s = make_float3(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < 8; ++i) {
    s = s + corners[i];
  }
  return s / 8.0f;
}

// Forward-matching box volume: sum_t |det(r_t)| / 6.
__device__ inline float BoxVolumeForward(const float3 corners[8]) {
  const float3 ctr = BoxCenterFromCorners(corners);
  float V = 0.0f;
  for (int t = 0; t < NUM_TRIS; ++t) {
    const float3 r0 = corners[_TRIS[t].v0] - ctr;
    const float3 r1 = corners[_TRIS[t].v1] - ctr;
    const float3 r2 = corners[_TRIS[t].v2] - ctr;
    V += fabsf(dot(r0, cross(r1, r2)));
  }
  return V / 6.0f;
}

// Chain a grad_V (scalar grad on the box volume) into grad_box_data via the
// signed-tet decomposition. Writes through atomicAdd so multiple (n, m)
// threads can contribute concurrently.
__device__ inline void AccumulateBoxVolumeGrad(
    const float3 corners[8],
    float grad_V,
    float* grad_box_data) {
  if (grad_V == 0.0f) {
    return;
  }
  const float3 ctr = BoxCenterFromCorners(corners);
  for (int t = 0; t < NUM_TRIS; ++t) {
    const int i0 = _TRIS[t].v0;
    const int i1 = _TRIS[t].v1;
    const int i2 = _TRIS[t].v2;
    const float3 r0 = corners[i0] - ctr;
    const float3 r1 = corners[i1] - ctr;
    const float3 r2 = corners[i2] - ctr;
    const float det = dot(r0, cross(r1, r2));
    const float sign_t = det >= 0.0f ? 1.0f : -1.0f;
    const float scale = grad_V * sign_t / 6.0f;
    const float3 g_r0 = scale * cross(r1, r2);
    const float3 g_r1 = scale * cross(r2, r0);
    const float3 g_r2 = scale * cross(r0, r1);
    const float3 g_ctr_part = (g_r0 + g_r1 + g_r2) / 8.0f;
    atomicAddFloat3(grad_box_data + i0 * 3, g_r0 - g_ctr_part);
    atomicAddFloat3(grad_box_data + i1 * 3, g_r1 - g_ctr_part);
    atomicAddFloat3(grad_box_data + i2 * 3, g_r2 - g_ctr_part);
    const float3 neg_part = (-1.0f) * g_ctr_part;
    for (int k = 0; k < 8; ++k) {
      if (k == i0 || k == i1 || k == i2) {
        continue;
      }
      atomicAddFloat3(grad_box_data + k * 3, neg_part);
    }
  }
}

// Per-plane gradient: chain grad_vol_total through n_out, d_out back to the
// four plane corners via the diagonal parameterization
//     m = (v2 - v0) x (v3 - v1).
// See iou_box3d_backward_cpu.cpp for the full derivation.
__device__ inline void AccumulatePlaneGrad(
    const float3 corners[8],
    const float3& box_center,
    int plane_idx,
    float face_area_p,
    const float3& Ac_p,
    float grad_vol_total,
    float* grad_box_data) {
  if (face_area_p <= 0.0f || grad_vol_total == 0.0f) {
    return;
  }
  const FaceVertsIdx& pidx = _PLANES[plane_idx];
  const float3& v0 = corners[pidx.v0];
  const float3& v1 = corners[pidx.v1];
  const float3& v2 = corners[pidx.v2];
  const float3& v3 = corners[pidx.v3];

  const float3 a = v2 - v0;
  const float3 b = v3 - v1;
  const float3 m = cross(a, b);
  const float m_norm = fmaxf(norm(m), kEpsilon);
  const float3 n_hat = m / m_norm;
  const float3 plane_ctr = (v0 + v1 + v2 + v3) / 4.0f;
  const float s = dot(n_hat, plane_ctr - box_center) >= 0.0f ? 1.0f : -1.0f;
  const float3 n_out = s * n_hat;

  const float3 dV_dn = (-1.0f) * Ac_p;
  const float dV_dd = face_area_p;
  const float3 gN = grad_vol_total * dV_dn;
  const float gD = grad_vol_total * dV_dd;

  const float3 combined = gN + gD * plane_ctr;
  const float n_dot = dot(n_hat, combined);
  const float3 grad_m = (s / m_norm) * (combined - n_dot * n_hat);

  const float3 d_term = (gD / 4.0f) * n_out;
  atomicAddFloat3(grad_box_data + pidx.v0 * 3, cross(grad_m, b) + d_term);
  atomicAddFloat3(grad_box_data + pidx.v1 * 3, cross(a, grad_m) + d_term);
  atomicAddFloat3(grad_box_data + pidx.v2 * 3, cross(b, grad_m) + d_term);
  atomicAddFloat3(grad_box_data + pidx.v3 * 3, cross(grad_m, a) + d_term);
}

__global__ void IoUBox3DBackwardKernel(
    const float* __restrict__ boxes1,
    const float* __restrict__ boxes2,
    const float* __restrict__ vol_in,
    const float* __restrict__ face_area_in,
    const float* __restrict__ face_area_centroid_in,
    const float* __restrict__ grad_vol_in,
    const float* __restrict__ grad_iou_in,
    float* __restrict__ grad_boxes1,
    float* __restrict__ grad_boxes2,
    int64_t N,
    int64_t M) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  float3 corners1[8];
  float3 corners2[8];

  for (size_t i = tid; i < static_cast<size_t>(N * M); i += stride) {
    const size_t n = i / M;
    const size_t m = i % M;

    // Load corners for the current pair.
    const float* b1 = boxes1 + n * 8 * 3;
    const float* b2 = boxes2 + m * 8 * 3;
    for (int k = 0; k < 8; ++k) {
      corners1[k] = make_float3(b1[k * 3 + 0], b1[k * 3 + 1], b1[k * 3 + 2]);
      corners2[k] = make_float3(b2[k * 3 + 0], b2[k * 3 + 1], b2[k * 3 + 2]);
    }

    const float3 center1 = BoxCenterFromCorners(corners1);
    const float3 center2 = BoxCenterFromCorners(corners2);
    const float V1 = BoxVolumeForward(corners1);
    const float V2 = BoxVolumeForward(corners2);

    const float v = vol_in[i];
    const float denom = fmaxf(V1 + V2 - v, kIoUDenomEps);
    const float inv_denom2 = 1.0f / (denom * denom);
    const float diou_dvol = (V1 + V2) * inv_denom2;
    const float diou_dV = -v * inv_denom2;

    const float gv = grad_vol_in[i];
    const float gi = grad_iou_in[i];
    const float grad_vol_total = gv + gi * diou_dvol;
    const float grad_V1_partial = gi * diou_dV;
    const float grad_V2_partial = gi * diou_dV;

    // Intersection-volume gradient via per-plane contributions.
    if (grad_vol_total != 0.0f) {
      const float* fa_row = face_area_in + i * 12;
      const float* fac_row = face_area_centroid_in + i * 12 * 3;
      for (int p = 0; p < NUM_PLANES; ++p) {
        const float3 Ac = make_float3(
            fac_row[p * 3 + 0], fac_row[p * 3 + 1], fac_row[p * 3 + 2]);
        AccumulatePlaneGrad(
            corners1,
            center1,
            p,
            fa_row[p],
            Ac,
            grad_vol_total,
            grad_boxes1 + n * 8 * 3);
      }
      for (int p = 0; p < NUM_PLANES; ++p) {
        const int gp = NUM_PLANES + p;
        const float3 Ac = make_float3(
            fac_row[gp * 3 + 0], fac_row[gp * 3 + 1], fac_row[gp * 3 + 2]);
        AccumulatePlaneGrad(
            corners2,
            center2,
            p,
            fa_row[gp],
            Ac,
            grad_vol_total,
            grad_boxes2 + m * 8 * 3);
      }
    }

    // Per-box-volume gradients chained through tetrahedron decomp.
    AccumulateBoxVolumeGrad(corners1, grad_V1_partial, grad_boxes1 + n * 8 * 3);
    AccumulateBoxVolumeGrad(corners2, grad_V2_partial, grad_boxes2 + m * 8 * 3);
  }
}

} // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor> IoUBox3DBackwardCuda(
    torch::stable::Tensor boxes1,
    torch::stable::Tensor boxes2,
    torch::stable::Tensor vol,
    torch::stable::Tensor face_area,
    torch::stable::Tensor face_area_centroid,
    torch::stable::Tensor grad_vol,
    torch::stable::Tensor grad_iou) {
  CHECK_CUDA(boxes1);
  CHECK_CUDA(boxes2);
  STD_TORCH_CHECK(
      boxes1.get_device_index() == boxes2.get_device_index(),
      "boxes1 and boxes2 must be on the same CUDA device");
  STD_TORCH_CHECK(
      boxes1.dim() == 3 && boxes1.size(1) == 8 && boxes1.size(2) == 3,
      "boxes1 must have shape (N, 8, 3)");
  STD_TORCH_CHECK(
      boxes2.dim() == 3 && boxes2.size(1) == 8 && boxes2.size(2) == 3,
      "boxes2 must have shape (M, 8, 3)");

  boxes1 = torch::stable::contiguous(boxes1);
  boxes2 = torch::stable::contiguous(boxes2);
  vol = torch::stable::contiguous(vol);
  face_area = torch::stable::contiguous(face_area);
  face_area_centroid = torch::stable::contiguous(face_area_centroid);
  grad_vol = torch::stable::contiguous(grad_vol);
  grad_iou = torch::stable::contiguous(grad_iou);

  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);

  auto grad_boxes1 = torch::stable::new_zeros(boxes1, {N, 8, 3});
  auto grad_boxes2 = torch::stable::new_zeros(boxes2, {M, 8, 3});

  if (N == 0 || M == 0) {
    return std::make_tuple(std::move(grad_boxes1), std::move(grad_boxes2));
  }

  const int32_t device_index = boxes1.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);

  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  auto stream = static_cast<cudaStream_t>(raw_stream);

  const size_t blocks = 512;
  const size_t threads = 256;

  IoUBox3DBackwardKernel<<<blocks, threads, 0, stream>>>(
      boxes1.const_data_ptr<float>(),
      boxes2.const_data_ptr<float>(),
      vol.const_data_ptr<float>(),
      face_area.const_data_ptr<float>(),
      face_area_centroid.const_data_ptr<float>(),
      grad_vol.const_data_ptr<float>(),
      grad_iou.const_data_ptr<float>(),
      grad_boxes1.mutable_data_ptr<float>(),
      grad_boxes2.mutable_data_ptr<float>(),
      N,
      M);

  STD_TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "IoUBox3DBackwardKernel launch failed");

  return std::make_tuple(std::move(grad_boxes1), std::move(grad_boxes2));
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CUDA, m) {
  m.impl("iou_box3d_backward", TORCH_BOX(IoUBox3DBackwardCuda));
}
