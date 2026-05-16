"""Tests for box3d_iou adapted from PyTorch3D's ``test_iou_box3d.py``.

Provenance:
    https://github.com/facebookresearch/pytorch3d/blob/main/tests/test_iou_box3d.py
    (BSD 3-Clause, Meta Platforms, Inc.)

The vision3d port:

* Calls the raw ``torch.ops.vision3d.iou_box3d`` schema directly so we can
  exercise both ``vol`` and ``iou``. The user-facing ``box3d_iou`` only
  returns ``iou``.
* Uses pytest + the conftest ``device`` fixture for CPU / CUDA parametrization.
* Drops the upstream naive-Python reference, the Objectron / real_boxes
  data-file fixtures, and tests of upstream-only validation that vision3d's
  port does not perform (coplanar-verts / zero-area planes raise in
  PyTorch3D but not here).
* Preserves the test corners verbatim — the expected volumes / IoUs are
  Meshlab-validated or come from upstream GH issue regressions, and we
  want bit-for-bit comparable coverage.

Existing functional tests live in ``test_ops_box3d_iou.py``; this file
focuses on the ground-truth cases that aren't already covered there, plus
regressions for upstream issues #992, #1082, and #1287.
"""

import random

import pytest
import torch
from torch import Tensor

from vision3d import _extension  # noqa: F401  # loads ``_C`` into torch.ops
from vision3d.ops import _meta_registrations  # noqa: F401  # fake-tensor kernels

UNIT_BOX = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]


def overlap(corners1: Tensor, corners2: Tensor) -> tuple[Tensor, Tensor]:
    """Thin wrapper that returns just ``(vol, iou)`` from the schema.

    Args:
        corners1: ``[N, 8, 3]`` first box corners.
        corners2: ``[M, 8, 3]`` second box corners.

    Returns:
        ``(vol, iou)`` pair of ``[N, M]`` tensors.
    """
    corners1 = corners1.to(torch.float32).contiguous()
    corners2 = corners2.to(torch.float32).contiguous()
    vol, iou, _, _ = torch.ops.vision3d.iou_box3d(corners1, corners2)
    return vol, iou


def random_rotation(device: torch.device) -> Tensor:
    """Uniformly random 3x3 rotation matrix.

    Local replacement for ``pytorch3d.transforms.random_rotation``. Uses the
    QR decomposition of a random normal matrix; signs of the columns are
    fixed via the diagonal of R to produce a uniform distribution on SO(3).

    Args:
        device: Device on which the rotation tensor is allocated.

    Returns:
        ``[3, 3]`` proper rotation matrix (``det = +1``).
    """
    g = torch.randn(3, 3, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(g)
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def create_box(xyz: Tensor, whl: Tensor) -> Tensor:
    """Axis-aligned 8-corner box from center + extents.

    Matches the corner ordering used by the PyTorch3D test suite.

    Args:
        xyz: ``[3]`` center position.
        whl: ``[3]`` extents along the x, y, z axes.

    Returns:
        ``[8, 3]`` corner positions.
    """
    x, y, z = xyz
    w, h, le = whl
    return torch.tensor(
        [
            [x - w / 2.0, y - h / 2.0, z - le / 2.0],
            [x + w / 2.0, y - h / 2.0, z - le / 2.0],
            [x + w / 2.0, y + h / 2.0, z - le / 2.0],
            [x - w / 2.0, y + h / 2.0, z - le / 2.0],
            [x - w / 2.0, y - h / 2.0, z + le / 2.0],
            [x + w / 2.0, y - h / 2.0, z + le / 2.0],
            [x + w / 2.0, y + h / 2.0, z + le / 2.0],
            [x - w / 2.0, y + h / 2.0, z + le / 2.0],
        ],
        device=xyz.device,
        dtype=torch.float32,
    )


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Deterministic seeds for randomized cases."""
    random.seed(1)
    torch.manual_seed(1)


class TestSameBox:
    """Cases derived from upstream tests #1 and #14: ``vol(B, B) = vol(B)``."""

    def test_axis_aligned_unit_box(self, device: torch.device) -> None:
        box1 = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        vol, iou = overlap(box1[None], box1[None])
        torch.testing.assert_close(vol, torch.tensor([[1.0]], device=device))
        torch.testing.assert_close(iou, torch.tensor([[1.0]], device=device))

    def test_rotated_box_against_itself(self, device: torch.device) -> None:
        # Upstream #14 — yaw-rotated unit cube against itself.
        corners = (
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )
            * 0.5
        )
        yaw = torch.tensor(0.185)
        rot = torch.tensor(
            [
                [torch.cos(yaw), 0.0, torch.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-torch.sin(yaw), 0.0, torch.cos(yaw)],
            ],
            dtype=torch.float32,
            device=device,
        )
        rotated = (rot.mm(corners.t())).t()
        _, iou = overlap(rotated[None], rotated[None])
        torch.testing.assert_close(
            iou, torch.tensor([[1.0]], device=device), atol=1e-2, rtol=0
        )


class TestAxisAlignedTranslation:
    """Upstream tests #2–#4: translated boxes have predictable overlap volume."""

    def test_translate_x(self, device: torch.device) -> None:
        box1 = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        dd = random.random()
        box2 = box1 + torch.tensor([[dd, 0.0, 0.0]], device=device)
        expected = torch.tensor([[1 - dd]], device=device)
        vol, _ = overlap(box1[None], box2[None])
        torch.testing.assert_close(vol, expected)
        # Symmetry.
        vol, _ = overlap(box2[None], box1[None])
        torch.testing.assert_close(vol, expected)

    def test_translate_y(self, device: torch.device) -> None:
        box1 = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        dd = random.random()
        box2 = box1 + torch.tensor([[0.0, dd, 0.0]], device=device)
        expected = torch.tensor([[1 - dd]], device=device)
        vol, _ = overlap(box1[None], box2[None])
        torch.testing.assert_close(vol, expected)
        vol, _ = overlap(box2[None], box1[None])
        torch.testing.assert_close(vol, expected)

    def test_translate_all_axes(self, device: torch.device) -> None:
        box1 = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        ddx, ddy, ddz = random.random(), random.random(), random.random()
        box2 = box1 + torch.tensor([[ddx, ddy, ddz]], device=device)
        expected = torch.tensor([[(1 - ddx) * (1 - ddy) * (1 - ddz)]], device=device)
        vol, _ = overlap(box1[None], box2[None])
        torch.testing.assert_close(vol, expected)
        # Symmetry.
        vol, _ = overlap(box2[None], box1[None])
        torch.testing.assert_close(vol, expected)
        # IoU against itself, after the shift, is still 1.
        _, iou = overlap(box2[None], box2[None])
        torch.testing.assert_close(iou, torch.tensor([[1.0]], device=device))


class TestRotationInvariance:
    """Upstream tests #5–#6: rotating + translating both boxes preserves vol."""

    def test_random_rotation(self, device: torch.device) -> None:
        box1 = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        ddx, ddy, ddz = random.random(), random.random(), random.random()
        box2 = box1 + torch.tensor([[ddx, ddy, ddz]], device=device)
        rot = random_rotation(device)
        box1r = box1 @ rot.transpose(0, 1)
        box2r = box2 @ rot.transpose(0, 1)
        expected = torch.tensor([[(1 - ddx) * (1 - ddy) * (1 - ddz)]], device=device)
        vol, _ = overlap(box1r[None], box2r[None])
        torch.testing.assert_close(vol, expected, atol=1e-3, rtol=1e-4)
        vol, _ = overlap(box2r[None], box1r[None])
        torch.testing.assert_close(vol, expected, atol=1e-3, rtol=1e-4)

    def test_random_rotation_translation(self, device: torch.device) -> None:
        box1 = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        ddx, ddy, ddz = random.random(), random.random(), random.random()
        box2 = box1 + torch.tensor([[ddx, ddy, ddz]], device=device)
        rot = random_rotation(device)
        tt = torch.rand((1, 3), dtype=torch.float32, device=device)
        box1r = box1 @ rot.transpose(0, 1) + tt
        box2r = box2 @ rot.transpose(0, 1) + tt
        expected = torch.tensor([[(1 - ddx) * (1 - ddy) * (1 - ddz)]], device=device)
        vol, _ = overlap(box1r[None], box2r[None])
        torch.testing.assert_close(vol, expected, atol=1e-3, rtol=1e-4)


class TestMeshlabValidated:
    """Volumes computed in Meshlab from upstream test #7 and #18–#19.

    Procedure: load corners as a mesh, compute convex hull, run
    "Quality Measure and Computation → Compute Geometric Measures",
    read "Mesh Volume" from stdout.
    """

    def test_hand_coded_pair_from_meshlab(self, device: torch.device) -> None:
        box1r = torch.tensor(
            [
                [3.1673, -2.2574, 0.4817],
                [4.6470, 0.2223, 2.4197],
                [5.2200, 1.1844, 0.7510],
                [3.7403, -1.2953, -1.1869],
                [-4.9316, 2.5724, 0.4856],
                [-3.4519, 5.0521, 2.4235],
                [-2.8789, 6.0142, 0.7549],
                [-4.3586, 3.5345, -1.1831],
            ],
            device=device,
            dtype=torch.float32,
        )
        box2r = torch.tensor(
            [
                [0.5623, 4.0647, 3.4334],
                [3.3584, 4.3191, 1.1791],
                [3.0724, -5.9235, -0.3315],
                [0.2763, -6.1779, 1.9229],
                [-2.0773, 4.6121, 0.2213],
                [0.7188, 4.8665, -2.0331],
                [0.4328, -5.3761, -3.5436],
                [-2.3633, -5.6305, -1.2893],
            ],
            device=device,
            dtype=torch.float32,
        )
        vol_inters = 33.558529
        vol_box1 = 65.899010
        vol_box2 = 156.386719
        iou_mesh = vol_inters / (vol_box1 + vol_box2 - vol_inters)
        vol, iou = overlap(box1r[None], box2r[None])
        torch.testing.assert_close(
            vol, torch.tensor([[vol_inters]], device=device), atol=1e-1, rtol=0
        )
        torch.testing.assert_close(
            iou, torch.tensor([[iou_mesh]], device=device), atol=1e-1, rtol=0
        )

    def test_gh1287_pair_a(self, device: torch.device) -> None:
        box18a = torch.tensor(
            [
                [-105.6248, -32.7026, -1.2279],
                [-106.4690, -30.8895, -1.2279],
                [-106.4690, -30.8895, -3.0279],
                [-105.6248, -32.7026, -3.0279],
                [-110.1575, -34.8132, -1.2279],
                [-111.0017, -33.0001, -1.2279],
                [-111.0017, -33.0001, -3.0279],
                [-110.1575, -34.8132, -3.0279],
            ],
            device=device,
            dtype=torch.float32,
        )
        box18b = torch.tensor(
            [
                [-105.5094, -32.9504, -1.0641],
                [-106.4272, -30.9793, -1.0641],
                [-106.4272, -30.9793, -3.1916],
                [-105.5094, -32.9504, -3.1916],
                [-110.0421, -35.0609, -1.0641],
                [-110.9599, -33.0899, -1.0641],
                [-110.9599, -33.0899, -3.1916],
                [-110.0421, -35.0609, -3.1916],
            ],
            device=device,
            dtype=torch.float32,
        )
        vol_inters = 17.108501
        vol_box1 = 18.000067
        vol_box2 = 23.128527
        iou_mesh = vol_inters / (vol_box1 + vol_box2 - vol_inters)
        vol, iou = overlap(box18a[None], box18b[None])
        torch.testing.assert_close(
            vol, torch.tensor([[vol_inters]], device=device), atol=1e-2, rtol=0
        )
        torch.testing.assert_close(
            iou, torch.tensor([[iou_mesh]], device=device), atol=1e-2, rtol=0
        )

    def test_gh1287_pair_b(self, device: torch.device) -> None:
        box19a = torch.tensor(
            [
                [-59.4785, -15.6003, 0.4398],
                [-60.2263, -13.6928, 0.4398],
                [-60.2263, -13.6928, -1.3909],
                [-59.4785, -15.6003, -1.3909],
                [-64.1743, -17.4412, 0.4398],
                [-64.9221, -15.5337, 0.4398],
                [-64.9221, -15.5337, -1.3909],
                [-64.1743, -17.4412, -1.3909],
            ],
            device=device,
            dtype=torch.float32,
        )
        box19b = torch.tensor(
            [
                [-59.4874, -15.5775, -0.1512],
                [-60.2174, -13.7155, -0.1512],
                [-60.2174, -13.7155, -1.9820],
                [-59.4874, -15.5775, -1.9820],
                [-64.1832, -17.4185, -0.1512],
                [-64.9132, -15.5564, -0.1512],
                [-64.9132, -15.5564, -1.9820],
                [-64.1832, -17.4185, -1.9820],
            ],
            device=device,
            dtype=torch.float32,
        )
        vol_inters = 12.505723
        vol_box1 = 18.918238
        vol_box2 = 18.468531
        iou_mesh = vol_inters / (vol_box1 + vol_box2 - vol_inters)
        vol, iou = overlap(box19a[None], box19b[None])
        torch.testing.assert_close(
            vol, torch.tensor([[vol_inters]], device=device), atol=1e-2, rtol=0
        )
        torch.testing.assert_close(
            iou, torch.tensor([[iou_mesh]], device=device), atol=1e-2, rtol=0
        )


class TestSkewedBoxes:
    """Upstream test #11 — boxes with non-cuboid (but coplanar) face quads."""

    def test_skewed_overlap(self, device: torch.device) -> None:
        box_skew_1 = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [-2, -2, 2],
                [2, -2, 2],
                [2, 2, 2],
                [-2, 2, 2],
            ],
            dtype=torch.float32,
            device=device,
        )
        box_skew_2 = torch.tensor(
            [
                [2.015995, 0.695233, 2.152806],
                [2.832533, 0.663448, 1.576389],
                [2.675445, -0.309592, 1.407520],
                [1.858907, -0.277806, 1.983936],
                [-0.413922, 3.161758, 2.044343],
                [2.852230, 3.034615, -0.261321],
                [2.223878, -0.857545, -0.936800],
                [-1.042273, -0.730402, 1.368864],
            ],
            dtype=torch.float32,
            device=device,
        )
        vol1 = 14.000
        vol2 = 14.000005
        vol_inters = 5.431122
        iou_mesh = vol_inters / (vol1 + vol2 - vol_inters)
        vol, iou = overlap(box_skew_1[None], box_skew_2[None])
        torch.testing.assert_close(
            vol, torch.tensor([[vol_inters]], device=device), atol=1e-1, rtol=0
        )
        torch.testing.assert_close(
            iou, torch.tensor([[iou_mesh]], device=device), atol=1e-1, rtol=0
        )


class TestGhIssue992:
    """Upstream test #13 — zero-area coplanar face after intersection."""

    def test_corner_to_corner_touch(self, device: torch.device) -> None:
        ctrs = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]], device=device, dtype=torch.float32
        )
        whl = torch.tensor(
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], device=device, dtype=torch.float32
        )
        box_a = create_box(ctrs[0], whl[0])
        box_b = create_box(ctrs[1], whl[1])
        vol, _ = overlap(box_a[None], box_b[None])
        torch.testing.assert_close(
            vol, torch.tensor([[2.0]], device=device), atol=1e-3, rtol=0
        )


class TestGhIssue1082:
    """Upstream test #15 — boxes overlapping by ~91% in real coordinates."""

    def test_near_identity_iou(self, device: torch.device) -> None:
        box_a = torch.tensor(
            [
                [-2.5629019, 4.13995749, -1.76344576],
                [1.92329434, 4.28127117, -1.86155124],
                [1.86994571, 5.97489644, -1.86155124],
                [-2.61625053, 5.83358276, -1.76344576],
                [-2.53123587, 4.14095496, -0.31397536],
                [1.95496037, 4.28226864, -0.41208084],
                [1.90161174, 5.97589391, -0.41208084],
                [-2.5845845, 5.83458023, -0.31397536],
            ],
            device=device,
            dtype=torch.float32,
        )
        box_b = torch.tensor(
            [
                [-2.6256125, 4.13036357, -1.82893437],
                [1.87201008, 4.25296695, -1.82893437],
                [1.82562476, 5.95458116, -1.82893437],
                [-2.67199782, 5.83197777, -1.82893437],
                [-2.6256125, 4.13036357, -0.40095884],
                [1.87201008, 4.25296695, -0.40095884],
                [1.82562476, 5.95458116, -0.40095884],
                [-2.67199782, 5.83197777, -0.40095884],
            ],
            device=device,
            dtype=torch.float32,
        )
        _, iou = overlap(box_a[None], box_b[None])
        torch.testing.assert_close(
            iou, torch.tensor([[0.91]], device=device), atol=1e-2, rtol=0
        )


class TestGhIssue1287:
    """Upstream test #16 — identical boxes far from origin (~150 units away)."""

    def test_identical_far_from_origin(self, device: torch.device) -> None:
        corners = torch.tensor(
            [
                [-167.5847, -70.6167, -2.7927],
                [-166.7333, -72.4264, -2.7927],
                [-166.7333, -72.4264, -4.5927],
                [-167.5847, -70.6167, -4.5927],
                [-163.0605, -68.4880, -2.7927],
                [-162.2090, -70.2977, -2.7927],
                [-162.2090, -70.2977, -4.5927],
                [-163.0605, -68.4880, -4.5927],
            ],
            device=device,
            dtype=torch.float32,
        )
        _, iou = overlap(corners[None], corners[None])
        torch.testing.assert_close(
            iou, torch.tensor([[1.0]], device=device), atol=1e-2, rtol=0
        )

    def test_almost_identical_far_from_origin(self, device: torch.device) -> None:
        # Upstream test #17 — corners differ by ~1e-5; IoU should still be ~1.
        box_a = torch.tensor(
            [
                [-33.94158, -4.51639, 0.96941],
                [-34.67156, -2.65437, 0.96941],
                [-34.67156, -2.65437, -0.95367],
                [-33.94158, -4.51639, -0.95367],
                [-38.75954, -6.40521, 0.96941],
                [-39.48952, -4.54319, 0.96941],
                [-39.48952, -4.54319, -0.95367],
                [-38.75954, -6.40521, -0.95367],
            ],
            device=device,
            dtype=torch.float32,
        )
        box_b = torch.tensor(
            [
                [-33.94159, -4.51638, 0.96939],
                [-34.67158, -2.65437, 0.96939],
                [-34.67158, -2.65437, -0.95368],
                [-33.94159, -4.51638, -0.95368],
                [-38.75954, -6.40523, 0.96939],
                [-39.48953, -4.54321, 0.96939],
                [-39.48953, -4.54321, -0.95368],
                [-38.75954, -6.40523, -0.95368],
            ],
            device=device,
            dtype=torch.float32,
        )
        _, iou = overlap(box_a[None], box_b[None])
        torch.testing.assert_close(
            iou, torch.tensor([[1.0]], device=device), atol=1e-2, rtol=0
        )


class TestDisjoint:
    """Upstream test #9 — non-overlapping boxes."""

    def test_far_apart(self, device: torch.device) -> None:
        box1 = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        box2 = box1 + torch.tensor([[0.0, 100.0, 0.0]], device=device)
        vol, iou = overlap(box1[None], box2[None])
        torch.testing.assert_close(vol, torch.tensor([[0.0]], device=device))
        torch.testing.assert_close(iou, torch.tensor([[0.0]], device=device))


class TestAutogradWiring:
    """End-to-end smoke test for the ``register_autograd`` wiring."""

    def test_iou_backward_runs(self, device: torch.device) -> None:
        box1 = torch.tensor(
            UNIT_BOX, dtype=torch.float32, device=device, requires_grad=True
        )
        box2 = (
            torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
            + torch.tensor([[0.3, 0.0, 0.0]], device=device)
        ).requires_grad_(True)
        _, iou, _, _ = torch.ops.vision3d.iou_box3d(box1[None], box2[None])
        loss = iou.sum()
        loss.backward()
        assert box1.grad is not None
        assert box1.grad.shape == box1.shape
        assert box2.grad is not None
        assert box2.grad.shape == box2.shape

    def test_vol_backward_runs(self, device: torch.device) -> None:
        box1 = torch.tensor(
            UNIT_BOX, dtype=torch.float32, device=device, requires_grad=True
        )
        box2 = (
            torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
            + torch.tensor([[0.3, 0.0, 0.0]], device=device)
        ).requires_grad_(True)
        vol, _, _, _ = torch.ops.vision3d.iou_box3d(box1[None], box2[None])
        vol.sum().backward()
        assert box1.grad is not None
        assert box2.grad is not None


class TestAnalyticBackward:
    """Analytic backward vs. finite differences.

    Tests use box parameters (chained through ``box3d_corners``) and
    rotated boxes that avoid the coplanar-faces edge case where the
    forward's dedup attributes the full face area to one box (which makes
    the analytic gradient a valid one-sided subgradient but not the
    centered FD average).
    """

    def test_grad_matches_fd_rotated_pair(self, device: torch.device) -> None:
        from vision3d.ops._box3d_corners import box3d_corners
        from vision3d.tensors import BoundingBox3DFormat

        fmt = BoundingBox3DFormat.XYZLWHY

        def iou_at(b1: Tensor, b2: Tensor) -> Tensor:
            c1 = box3d_corners(b1, fmt).to(torch.float32)
            c2 = box3d_corners(b2, fmt).to(torch.float32)
            _, iou, _, _ = torch.ops.vision3d.iou_box3d(c1, c2)
            return iou

        b1 = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        b2 = torch.tensor(
            [[0.4, 0.2, 0.1, 1.0, 1.0, 1.0, -0.2]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        iou = iou_at(b1, b2)
        iou.sum().backward()
        assert b1.grad is not None
        assert b2.grad is not None
        analytic_b1 = b1.grad.clone()
        analytic_b2 = b2.grad.clone()

        eps = 1e-2

        def fd_grad(boxes: Tensor, other: Tensor, perturb_first: bool) -> Tensor:
            g = torch.zeros_like(boxes)
            base = boxes.detach()
            other = other.detach()
            for j in range(boxes.shape[1]):
                bp = base.clone()
                bm = base.clone()
                bp[0, j] += eps
                bm[0, j] -= eps
                a, b = (bp, other) if perturb_first else (other, bp)
                am, bm_ = (bm, other) if perturb_first else (other, bm)
                ip = iou_at(a, b).item()
                im = iou_at(am, bm_).item()
                g[0, j] = (ip - im) / (2 * eps)
            return g

        fd_b1 = fd_grad(b1, b2, perturb_first=True)
        fd_b2 = fd_grad(b2, b1, perturb_first=False)

        torch.testing.assert_close(analytic_b1, fd_b1, atol=2e-3, rtol=0)
        torch.testing.assert_close(analytic_b2, fd_b2, atol=2e-3, rtol=0)

    def test_grad_zero_for_disjoint(self, device: torch.device) -> None:
        # Boxes far apart: face_area is zero everywhere, gradients should be 0.
        box1 = torch.tensor(
            UNIT_BOX, dtype=torch.float32, device=device, requires_grad=True
        )
        box2 = (
            torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
            + torch.tensor([[100.0, 0.0, 0.0]], device=device)
        ).requires_grad_(True)
        _, iou, _, _ = torch.ops.vision3d.iou_box3d(box1[None], box2[None])
        iou.sum().backward()
        assert box1.grad is not None
        assert box2.grad is not None
        torch.testing.assert_close(box1.grad, torch.zeros_like(box1.grad))
        torch.testing.assert_close(box2.grad, torch.zeros_like(box2.grad))

    def test_grad_matches_fd_coplanar_config(self, device: torch.device) -> None:
        """gradcheck-tight even at exactly-coplanar configurations.

        Boxes share four side faces (parallel offset in z). The analytic
        backward gives the centered subgradient (A/2 attribution per
        side); naive dedup would give an asymmetric (A vs 0) attribution.

        Yaw uses a wider FD eps (0.05) than the other parameters because
        the forward kernel's coplanar check has a built-in angular
        tolerance (``dEpsilon = 1e-3`` => ~2.5°): perturbations smaller
        than this gate the same coplanar branch and the asymmetric
        ordering of ``IsCoplanarTriTri`` makes ``iou(+e) != iou(-e)``
        even though the smooth-limit value is symmetric. Past the gate
        threshold, the forward becomes well-behaved.
        """
        from vision3d.ops._box3d_corners import box3d_corners
        from vision3d.tensors import BoundingBox3DFormat

        fmt = BoundingBox3DFormat.XYZLWHY

        def iou_at(b1: Tensor, b2: Tensor) -> Tensor:
            c1 = box3d_corners(b1, fmt).to(torch.float32)
            c2 = box3d_corners(b2, fmt).to(torch.float32)
            _, iou, _, _ = torch.ops.vision3d.iou_box3d(c1, c2)
            return iou

        b1 = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        b2 = torch.tensor(
            [[0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        iou = iou_at(b1, b2)
        iou.sum().backward()
        assert b1.grad is not None
        assert b2.grad is not None
        analytic_b1 = b1.grad.clone()

        # Per-parameter FD eps. Yaw uses a wider step to escape the
        # forward kernel's coplanar tolerance band (see docstring).
        eps_per_param = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 5e-2]

        def fd_grad(boxes: Tensor, other: Tensor) -> Tensor:
            g = torch.zeros_like(boxes)
            base = boxes.detach()
            other = other.detach()
            for j in range(boxes.shape[1]):
                eps = eps_per_param[j]
                bp = base.clone()
                bm = base.clone()
                bp[0, j] += eps
                bm[0, j] -= eps
                ip = iou_at(bp, other).item()
                im = iou_at(bm, other).item()
                g[0, j] = (ip - im) / (2 * eps)
            return g

        fd_b1 = fd_grad(b1, b2)
        torch.testing.assert_close(analytic_b1, fd_b1, atol=3e-3, rtol=0)

    def test_cpu_cuda_grad_parity(self) -> None:
        # CUDA backward should produce the same gradients as CPU, modulo
        # small floating-point reduction-order differences (atomicAdd).
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from vision3d.ops._box3d_corners import box3d_corners
        from vision3d.tensors import BoundingBox3DFormat

        fmt = BoundingBox3DFormat.XYZLWHY

        def iou_at(b1: Tensor, b2: Tensor) -> Tensor:
            c1 = box3d_corners(b1, fmt).to(torch.float32)
            c2 = box3d_corners(b2, fmt).to(torch.float32)
            _, iou, _, _ = torch.ops.vision3d.iou_box3d(c1, c2)
            return iou

        params1 = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3]]
        params2 = [[0.4, 0.2, 0.1, 1.0, 1.0, 1.0, -0.2]]

        b1_cpu = torch.tensor(
            params1, dtype=torch.float32, device="cpu", requires_grad=True
        )
        b2_cpu = torch.tensor(
            params2, dtype=torch.float32, device="cpu", requires_grad=True
        )
        iou_at(b1_cpu, b2_cpu).sum().backward()

        b1_cuda = torch.tensor(
            params1, dtype=torch.float32, device="cuda", requires_grad=True
        )
        b2_cuda = torch.tensor(
            params2, dtype=torch.float32, device="cuda", requires_grad=True
        )
        iou_at(b1_cuda, b2_cuda).sum().backward()

        assert b1_cpu.grad is not None
        assert b1_cuda.grad is not None
        assert b2_cpu.grad is not None
        assert b2_cuda.grad is not None
        torch.testing.assert_close(
            b1_cpu.grad, b1_cuda.grad.cpu(), atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(
            b2_cpu.grad, b2_cuda.grad.cpu(), atol=1e-4, rtol=1e-3
        )
