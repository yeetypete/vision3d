/*
 * Analytic backward for iou_box3d (CPU).
 *
 * Implements the closed-form gradient of the intersection-volume + IoU
 * computation w.r.t. the 8-corner inputs, using state recorded by the
 * forward (per-input-plane face area + area-weighted centroid).
 *
 * For each of the 12 input planes p (6 from boxes1, 6 from boxes2), the
 * intersection volume V satisfies (with outward unit normal n_p, offset
 * d_p, face area A_p, face centroid c_p):
 *
 *     dV / d d_p = A_p,           dV / d n_p = -A_p * c_p.
 *
 * We parameterize each plane by its diagonal cross product
 *     m = (v2 - v0) x (v3 - v1)
 * so all four plane corners receive nonzero contributions. The plane's
 * sign is chosen so n_p points OUTWARD from the source box; this matches
 * the forward's effective orientation (the forward orients normals INWARD
 * via `PlaneNormalDirection` and then tests via `IsInside`, which is
 * equivalent up to a global sign flip that the math handles).
 *
 * Box volumes V1, V2 (used in iou = vol / (V1 + V2 - vol)) are recomputed
 * here from corners via the same signed-tetrahedron decomposition the
 * forward uses. Sign per tetrahedron is taken from sign(det_t) so the
 * formula matches the forward's `abs(det)/6` semantics.
 */

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <array>
#include <cmath>
#include <tuple>
#include "iou_box3d/iou_box3d.h"
#include "iou_box3d/iou_utils.h"
#include "utils/pytorch3d_cutils.h"

namespace {

using Corners = std::array<vec3<float>, 8>;

inline Corners ZeroCorners() {
  const vec3<float> z(0.0f, 0.0f, 0.0f);
  return Corners{z, z, z, z, z, z, z, z};
}

// Read 8 corners of one box from contiguous (8, 3) float data.
inline Corners ReadCorners(const float* box_data) {
  return Corners{
      vec3<float>(box_data[0], box_data[1], box_data[2]),
      vec3<float>(box_data[3], box_data[4], box_data[5]),
      vec3<float>(box_data[6], box_data[7], box_data[8]),
      vec3<float>(box_data[9], box_data[10], box_data[11]),
      vec3<float>(box_data[12], box_data[13], box_data[14]),
      vec3<float>(box_data[15], box_data[16], box_data[17]),
      vec3<float>(box_data[18], box_data[19], box_data[20]),
      vec3<float>(box_data[21], box_data[22], box_data[23])};
}

inline vec3<float> BoxCenterOf(const Corners& c) {
  vec3<float> s(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < 8; ++i) {
    s = s + c[i];
  }
  return s / 8.0f;
}

// Forward-matching box volume: sum_t |det(r_t)| / 6 where r_t are the three
// face-triangle vertices relative to the box centroid.
inline float BoxVolumeForward(const Corners& c) {
  const vec3<float> ctr = BoxCenterOf(c);
  float V = 0.0f;
  for (int t = 0; t < NUM_TRIS; ++t) {
    const vec3<float> r0 = c[_TRIS[t][0]] - ctr;
    const vec3<float> r1 = c[_TRIS[t][1]] - ctr;
    const vec3<float> r2 = c[_TRIS[t][2]] - ctr;
    const float det = dot(r0, cross(r1, r2));
    V += std::abs(det);
  }
  return V / 6.0f;
}

// Backward of BoxVolumeForward: distribute `grad_V` into per-corner grads.
// d(|det_t|)/d(corner) = sign(det_t) * d(det_t)/d(corner). Each corner k
// contributes to a tetrahedron via direct membership in _TRIS[t] AND via
// the box centroid (which is a 1/8 weighted sum of all 8 corners).
inline void AccumulateBoxVolumeGrad(
    const Corners& corners,
    float grad_V,
    Corners& grad_corners) {
  if (grad_V == 0.0f) {
    return;
  }
  const vec3<float> ctr = BoxCenterOf(corners);
  for (int t = 0; t < NUM_TRIS; ++t) {
    const int i0 = _TRIS[t][0];
    const int i1 = _TRIS[t][1];
    const int i2 = _TRIS[t][2];
    const vec3<float> r0 = corners[i0] - ctr;
    const vec3<float> r1 = corners[i1] - ctr;
    const vec3<float> r2 = corners[i2] - ctr;
    const float det = dot(r0, cross(r1, r2));
    const float sign_t = det >= 0.0f ? 1.0f : -1.0f;
    const float scale = grad_V * sign_t / 6.0f;
    // d(det)/d(r_i): r_{i+1} x r_{i+2} (cyclic).
    const vec3<float> g_r0 = scale * cross(r1, r2);
    const vec3<float> g_r1 = scale * cross(r2, r0);
    const vec3<float> g_r2 = scale * cross(r0, r1);
    // r_i = corner[i_i] - ctr, ctr = (1/8) sum corner. So d(r_i)/d(corner_k) =
    // delta_{i_i, k} I - (1/8) I. Distribute accordingly.
    const vec3<float> g_ctr_part = (g_r0 + g_r1 + g_r2) / 8.0f;
    grad_corners[i0] = grad_corners[i0] + g_r0 - g_ctr_part;
    grad_corners[i1] = grad_corners[i1] + g_r1 - g_ctr_part;
    grad_corners[i2] = grad_corners[i2] + g_r2 - g_ctr_part;
    for (int k = 0; k < 8; ++k) {
      if (k == i0 || k == i1 || k == i2) {
        continue;
      }
      grad_corners[k] = grad_corners[k] - g_ctr_part;
    }
  }
}

// Accumulate the per-plane gradient contribution into the 4 corner grads on
// that plane. `grad_vol_total` is the upstream gradient on the intersection
// volume V. `face_area_p` and `Ac_p` (= area * centroid) come from the
// forward's saved state for this input plane.
inline void AccumulatePlaneGrad(
    const Corners& corners,
    const vec3<float>& box_center,
    int plane_idx,
    float face_area_p,
    const vec3<float>& Ac_p,
    float grad_vol_total,
    Corners& grad_corners) {
  if (face_area_p <= 0.0f || grad_vol_total == 0.0f) {
    return;
  }
  const int* pidx = _PLANES[plane_idx];
  const vec3<float>& v0 = corners[pidx[0]];
  const vec3<float>& v1 = corners[pidx[1]];
  const vec3<float>& v2 = corners[pidx[2]];
  const vec3<float>& v3 = corners[pidx[3]];

  // m = (v2 - v0) x (v3 - v1). Diagonals; every corner contributes.
  const vec3<float> a = v2 - v0;
  const vec3<float> b = v3 - v1;
  const vec3<float> m = cross(a, b);
  const float m_norm = std::fmax(norm(m), static_cast<float>(kEpsilon));
  const vec3<float> n_hat = m / m_norm;
  const vec3<float> plane_ctr = (v0 + v1 + v2 + v3) / 4.0f;

  // Outward sign: n should point AWAY from box center.
  const float s = dot(n_hat, plane_ctr - box_center) >= 0.0f ? 1.0f : -1.0f;
  const vec3<float> n_out = s * n_hat;

  // dV/dn_out = -Ac_p, dV/dd_out = A_p.
  const vec3<float> dV_dn = (-1.0f) * Ac_p;
  const float dV_dd = face_area_p;

  // Upstream-scaled.
  const vec3<float> gN = grad_vol_total * dV_dn;
  const float gD = grad_vol_total * dV_dd;

  // dV/dm via reverse-mode through (n_out, d_out):
  //     dV/dm = (s/|m|) (I - n_hat n_hat^T) [dV/dn_out + plane_ctr * dV/dd_out]
  const vec3<float> combined = gN + gD * plane_ctr;
  const float n_dot = dot(n_hat, combined);
  const vec3<float> grad_m = (s / m_norm) * (combined - n_dot * n_hat);

  // Distribute through m = a x b with a = v2 - v0, b = v3 - v1.
  //   dm/dv0 = skew(b)            ->  dV/dv0 (through m) = grad_m x b
  //   dm/dv1 = -skew(a)           ->  dV/dv1 (through m) = a x grad_m
  //   dm/dv2 = -skew(b)           ->  dV/dv2 (through m) = b x grad_m
  //   dm/dv3 = skew(a)            ->  dV/dv3 (through m) = grad_m x a
  // plus the plane_ctr direct dep on each corner: dplane_ctr/dv_i = (1/4) I
  //   -> dV/dv_i (from d_out direct via plane_ctr) = (gD / 4) * n_out.
  const vec3<float> d_term = (gD / 4.0f) * n_out;
  grad_corners[pidx[0]] = grad_corners[pidx[0]] + cross(grad_m, b) + d_term;
  grad_corners[pidx[1]] = grad_corners[pidx[1]] + cross(a, grad_m) + d_term;
  grad_corners[pidx[2]] = grad_corners[pidx[2]] + cross(b, grad_m) + d_term;
  grad_corners[pidx[3]] = grad_corners[pidx[3]] + cross(grad_m, a) + d_term;
}

inline void WriteCorners(const Corners& c, float* out) {
  for (int i = 0; i < 8; ++i) {
    out[i * 3 + 0] = c[i].x;
    out[i * 3 + 1] = c[i].y;
    out[i * 3 + 2] = c[i].z;
  }
}

constexpr float kIoUDenomEps = 1e-12f;

} // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor> IoUBox3DBackwardCpu(
    torch::stable::Tensor boxes1,
    torch::stable::Tensor boxes2,
    torch::stable::Tensor vol,
    torch::stable::Tensor face_area,
    torch::stable::Tensor face_area_centroid,
    torch::stable::Tensor grad_vol,
    torch::stable::Tensor grad_iou) {
  CHECK_CPU(boxes1);
  CHECK_CPU(boxes2);
  CHECK_CPU(vol);
  CHECK_CPU(face_area);
  CHECK_CPU(face_area_centroid);
  CHECK_CPU(grad_vol);
  CHECK_CPU(grad_iou);
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

  const float* boxes1_data = boxes1.const_data_ptr<float>();
  const float* boxes2_data = boxes2.const_data_ptr<float>();
  const float* vol_data = vol.const_data_ptr<float>();
  const float* face_area_data = face_area.const_data_ptr<float>();
  const float* face_area_centroid_data =
      face_area_centroid.const_data_ptr<float>();
  const float* grad_vol_data = grad_vol.const_data_ptr<float>();
  const float* grad_iou_data = grad_iou.const_data_ptr<float>();
  float* grad_boxes1_data = grad_boxes1.mutable_data_ptr<float>();
  float* grad_boxes2_data = grad_boxes2.mutable_data_ptr<float>();

  // Precompute per-box state (corners, center, V_box).
  std::vector<Corners> corners1;
  corners1.reserve(N);
  std::vector<vec3<float>> centers1;
  centers1.reserve(N);
  std::vector<float> V1(N);
  for (int64_t n = 0; n < N; ++n) {
    corners1.push_back(ReadCorners(boxes1_data + n * 24));
    centers1.push_back(BoxCenterOf(corners1[n]));
    V1[n] = BoxVolumeForward(corners1[n]);
  }
  std::vector<Corners> corners2;
  corners2.reserve(M);
  std::vector<vec3<float>> centers2;
  centers2.reserve(M);
  std::vector<float> V2(M);
  for (int64_t m = 0; m < M; ++m) {
    corners2.push_back(ReadCorners(boxes2_data + m * 24));
    centers2.push_back(BoxCenterOf(corners2[m]));
    V2[m] = BoxVolumeForward(corners2[m]);
  }

  // Accumulators in vec3 space; written out at the end.
  std::vector<Corners> grad_corners1(N, ZeroCorners());
  std::vector<Corners> grad_corners2(M, ZeroCorners());

  // Per-box volume gradients accumulate from each (n, m) pair via the iou
  // chain rule. We sum first, then chain through the tetrahedron decomp.
  std::vector<float> grad_V1(N, 0.0f);
  std::vector<float> grad_V2(M, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t m = 0; m < M; ++m) {
      const int64_t nm = n * M + m;
      const float v = vol_data[nm];
      const float denom = std::fmax(V1[n] + V2[m] - v, kIoUDenomEps);
      const float inv_denom2 = 1.0f / (denom * denom);
      // diou/dvol = (V1 + V2) / denom^2; diou/dV1 = diou/dV2 = -v / denom^2.
      const float diou_dvol = (V1[n] + V2[m]) * inv_denom2;
      const float diou_dV = -v * inv_denom2;

      const float gv = grad_vol_data[nm];
      const float gi = grad_iou_data[nm];
      const float grad_vol_total = gv + gi * diou_dvol;
      grad_V1[n] += gi * diou_dV;
      grad_V2[m] += gi * diou_dV;

      if (grad_vol_total == 0.0f) {
        continue;
      }

      // Planes 0..5 belong to boxes1; planes 6..11 belong to boxes2.
      const float* fa_row = face_area_data + nm * 12;
      const float* fac_row = face_area_centroid_data + nm * 12 * 3;
      for (int p = 0; p < NUM_PLANES; ++p) {
        const vec3<float> Ac(
            fac_row[p * 3 + 0], fac_row[p * 3 + 1], fac_row[p * 3 + 2]);
        AccumulatePlaneGrad(
            corners1[n],
            centers1[n],
            p,
            fa_row[p],
            Ac,
            grad_vol_total,
            grad_corners1[n]);
      }
      for (int p = 0; p < NUM_PLANES; ++p) {
        const int gp = NUM_PLANES + p;
        const vec3<float> Ac(
            fac_row[gp * 3 + 0], fac_row[gp * 3 + 1], fac_row[gp * 3 + 2]);
        AccumulatePlaneGrad(
            corners2[m],
            centers2[m],
            p,
            fa_row[gp],
            Ac,
            grad_vol_total,
            grad_corners2[m]);
      }
    }
  }

  // Chain the V_box gradients through the corner-level tet decomp.
  for (int64_t n = 0; n < N; ++n) {
    AccumulateBoxVolumeGrad(corners1[n], grad_V1[n], grad_corners1[n]);
  }
  for (int64_t m = 0; m < M; ++m) {
    AccumulateBoxVolumeGrad(corners2[m], grad_V2[m], grad_corners2[m]);
  }

  // Write out.
  for (int64_t n = 0; n < N; ++n) {
    WriteCorners(grad_corners1[n], grad_boxes1_data + n * 24);
  }
  for (int64_t m = 0; m < M; ++m) {
    WriteCorners(grad_corners2[m], grad_boxes2_data + m * 24);
  }

  return std::make_tuple(std::move(grad_boxes1), std::move(grad_boxes2));
}

STABLE_TORCH_LIBRARY_IMPL(vision3d, CPU, m) {
  m.impl("iou_box3d_backward", TORCH_BOX(IoUBox3DBackwardCpu));
}
