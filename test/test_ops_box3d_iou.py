"""Tests for box3d_iou.

Covers all four formats, including full 9-DOF XYZLWHYPR. Hand-computed
overlap values are cross-checked against the algorithm.
"""

import math

import pytest
import torch
import torch.testing

from vision3d.ops import box3d_iou
from vision3d.tensors import BoundingBox3DFormat

_IDENTITY_TOL = 5e-4
_VALUE_TOL = 2e-3


class TestBox3dIouAxisAligned:
    def test_identity_xyzlwh(self) -> None:
        boxes = torch.tensor([[0.0, 0, 0, 2, 2, 2]])
        iou = box3d_iou(boxes, boxes, BoundingBox3DFormat.XYZLWH)
        assert abs(iou.item() - 1.0) < _IDENTITY_TOL

    def test_identity_xyzxyz(self) -> None:
        boxes = torch.tensor([[-1.0, -1, -1, 1, 1, 1]])
        iou = box3d_iou(boxes, boxes, BoundingBox3DFormat.XYZXYZ)
        assert abs(iou.item() - 1.0) < _IDENTITY_TOL

    def test_disjoint(self) -> None:
        b1 = torch.tensor([[0.0, 0, 0, 1, 1, 1]])
        b2 = torch.tensor([[10.0, 10, 10, 1, 1, 1]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        assert iou.item() < 1e-5

    def test_half_overlap_x(self) -> None:
        # Unit boxes offset by 0.5 in x. Intersection vol = 0.5x1x1 = 0.5.
        # Union = 1 + 1 - 0.5 = 1.5. IoU = 1/3.
        b1 = torch.tensor([[0.0, 0, 0, 1, 1, 1]])
        b2 = torch.tensor([[0.5, 0, 0, 1, 1, 1]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        assert abs(iou.item() - 1.0 / 3.0) < _VALUE_TOL

    def test_quarter_overlap_xy(self) -> None:
        # Offset by 0.5 in both x and y -> intersection is 0.5x0.5x1 = 0.25.
        # Union = 1 + 1 - 0.25 = 1.75. IoU = 0.25/1.75 = 1/7.
        b1 = torch.tensor([[0.0, 0, 0, 1, 1, 1]])
        b2 = torch.tensor([[0.5, 0.5, 0, 1, 1, 1]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        assert abs(iou.item() - 1.0 / 7.0) < _VALUE_TOL

    def test_fully_contained(self) -> None:
        # Small box fully inside a large box.
        # inner vol = 1, outer vol = 27. IoU = 1/27.
        outer = torch.tensor([[0.0, 0, 0, 3, 3, 3]])
        inner = torch.tensor([[0.0, 0, 0, 1, 1, 1]])
        iou = box3d_iou(outer, inner, BoundingBox3DFormat.XYZLWH)
        assert abs(iou.item() - 1.0 / 27.0) < _VALUE_TOL

    def test_xyzxyz_cross_check(self) -> None:
        # Same geometry in XYZXYZ format should match XYZLWH.
        b1_lwh = torch.tensor([[0.0, 0, 0, 2, 2, 2]])
        b2_lwh = torch.tensor([[1.0, 0, 0, 2, 2, 2]])
        b1_xyz = torch.tensor([[-1.0, -1, -1, 1, 1, 1]])
        b2_xyz = torch.tensor([[0.0, -1, -1, 2, 1, 1]])
        iou_lwh = box3d_iou(b1_lwh, b2_lwh, BoundingBox3DFormat.XYZLWH)
        iou_xyz = box3d_iou(b1_xyz, b2_xyz, BoundingBox3DFormat.XYZXYZ)
        assert abs(iou_lwh.item() - iou_xyz.item()) < 1e-5


class TestBox3dIouYawRotated:
    def test_identity(self) -> None:
        b = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0.5]])
        iou = box3d_iou(b, b, BoundingBox3DFormat.XYZLWHY)
        assert abs(iou.item() - 1.0) < _IDENTITY_TOL

    def test_cube_90deg_symmetry(self) -> None:
        # 2x2x2 cube is symmetric under 90deg yaw.
        b1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0.0]])
        b2 = torch.tensor([[0.0, 0, 0, 2, 2, 2, math.pi / 2]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWHY)
        assert abs(iou.item() - 1.0) < _VALUE_TOL

    def test_non_square_90deg(self) -> None:
        # 2x1x1 box rotated 90° vs unrotated. Intersection is a 1x1x1 cube
        # (the overlap of the two rectangular footprints in the BEV plane).
        # Vol1 = Vol2 = 2. Inter = 1. Union = 3. IoU = 1/3.
        b1 = torch.tensor([[0.0, 0, 0, 2, 1, 1, 0.0]])
        b2 = torch.tensor([[0.0, 0, 0, 2, 1, 1, math.pi / 2]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWHY)
        assert abs(iou.item() - 1.0 / 3.0) < _VALUE_TOL

    def test_45deg_vs_unrotated(self) -> None:
        # Unit box vs itself rotated 45°. Intersection is a regular octagon
        # footprint extruded through Z, with a hand-computed area.
        # For unit squares, the overlap area = 2*sqrt(2) - 2 ≈ 0.828.
        # Vol1 = Vol2 = 1. Inter ≈ 0.828. Union ≈ 1.172. IoU ≈ 0.707.
        b1 = torch.tensor([[0.0, 0, 0, 1, 1, 1, 0.0]])
        b2 = torch.tensor([[0.0, 0, 0, 1, 1, 1, math.pi / 4]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWHY)
        expected = (2 * math.sqrt(2) - 2) / (2 - (2 * math.sqrt(2) - 2))
        assert abs(iou.item() - expected) < _VALUE_TOL

    def test_no_z_overlap(self) -> None:
        # Same XY, but Z ranges disjoint -> IoU = 0.
        b1 = torch.tensor([[0.0, 0, 0, 1, 1, 1, 0.0]])
        b2 = torch.tensor([[0.0, 0, 5, 1, 1, 1, 0.0]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWHY)
        assert iou.item() < 1e-4


class TestBox3dIouFull9DOF:
    def test_identity(self) -> None:
        b = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0.3, 0.2, 0.1]])
        iou = box3d_iou(b, b, BoundingBox3DFormat.XYZLWHYPR)
        assert abs(iou.item() - 1.0) < _IDENTITY_TOL

    def test_pitch_matters(self) -> None:
        # A cube rotated only in pitch should have IoU < 1 against itself
        # unrotated (unlike yaw-only approximations, where pitch is
        # silently ignored).
        b1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0, 0, 0]])
        b2 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0, 0.5, 0]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWHYPR)
        assert 0.0 < iou.item() < 1.0

    def test_roll_matters(self) -> None:
        b1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0, 0, 0]])
        b2 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0, 0, 0.5]])
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWHYPR)
        assert 0.0 < iou.item() < 1.0

    def test_pitch_and_roll_differ(self) -> None:
        b0 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0, 0, 0]])
        b_pitch = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0, 0.3, 0]])
        b_both = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0, 0.3, 0.3]])
        iou_pitch = box3d_iou(b0, b_pitch, BoundingBox3DFormat.XYZLWHYPR)
        iou_both = box3d_iou(b0, b_both, BoundingBox3DFormat.XYZLWHYPR)
        assert abs(iou_pitch.item() - iou_both.item()) > 0.01

    def test_yaw_only_matches_xyzlwhy(self) -> None:
        b1_yaw = torch.tensor([[0.0, 0, 0, 2, 1, 1, 0.3]])
        b2_yaw = torch.tensor([[0.5, 0.2, 0, 2, 1, 1, -0.1]])
        b1_full = torch.tensor([[0.0, 0, 0, 2, 1, 1, 0.3, 0, 0]])
        b2_full = torch.tensor([[0.5, 0.2, 0, 2, 1, 1, -0.1, 0, 0]])
        iou_yaw = box3d_iou(b1_yaw, b2_yaw, BoundingBox3DFormat.XYZLWHY)
        iou_full = box3d_iou(b1_full, b2_full, BoundingBox3DFormat.XYZLWHYPR)
        assert abs(iou_yaw.item() - iou_full.item()) < _VALUE_TOL


class TestBox3dIouProperties:
    def test_pairwise_shape(self) -> None:
        b1 = torch.rand(5, 6) + 1.0  # centers 1..2, dims 1..2
        b2 = torch.rand(3, 6) + 1.0
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        assert iou.shape == (5, 3)

    def test_symmetry(self) -> None:
        torch.manual_seed(0)
        b1 = torch.cat([torch.rand(4, 3) * 2, torch.rand(4, 3) + 0.5], dim=-1)
        b2 = torch.cat([torch.rand(3, 3) * 2, torch.rand(3, 3) + 0.5], dim=-1)
        iou_ab = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        iou_ba = box3d_iou(b2, b1, BoundingBox3DFormat.XYZLWH)
        torch.testing.assert_close(iou_ab, iou_ba.T, atol=1e-4, rtol=1e-4)

    def test_self_iou_diagonal_is_one(self) -> None:
        torch.manual_seed(1)
        boxes = torch.cat(
            [torch.rand(4, 3) * 2, torch.rand(4, 3) + 0.5, torch.rand(4, 1)],
            dim=-1,
        )
        iou = box3d_iou(boxes, boxes, BoundingBox3DFormat.XYZLWHY)
        diag = iou.diagonal()
        for v in diag:
            assert abs(v.item() - 1.0) < _IDENTITY_TOL

    def test_range(self) -> None:
        torch.manual_seed(2)
        b1 = torch.cat([torch.rand(6, 3) * 3, torch.rand(6, 3) + 0.5], dim=-1)
        b2 = torch.cat([torch.rand(6, 3) * 3, torch.rand(6, 3) + 0.5], dim=-1)
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        assert iou.min().item() >= 0.0
        assert iou.max().item() <= 1.0

    def test_empty_inputs(self) -> None:
        b1 = torch.zeros(0, 6)
        b2 = torch.zeros(3, 6)
        iou = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        assert iou.shape == (0, 3)

        iou = box3d_iou(b2, b1, BoundingBox3DFormat.XYZLWH)
        assert iou.shape == (3, 0)

    @pytest.mark.parametrize(
        "format",
        [
            BoundingBox3DFormat.XYZXYZ,
            BoundingBox3DFormat.XYZLWH,
            BoundingBox3DFormat.XYZLWHY,
            BoundingBox3DFormat.XYZLWHYPR,
        ],
    )
    def test_all_formats_supported(self, format: BoundingBox3DFormat) -> None:
        # 6 / 6 / 7 / 9 columns depending on format.
        k = {
            BoundingBox3DFormat.XYZXYZ: 6,
            BoundingBox3DFormat.XYZLWH: 6,
            BoundingBox3DFormat.XYZLWHY: 7,
            BoundingBox3DFormat.XYZLWHYPR: 9,
        }[format]
        if format is BoundingBox3DFormat.XYZXYZ:
            box = torch.tensor([[-1.0, -1, -1, 1, 1, 1]])
        else:
            box = torch.zeros(1, k)
            box[0, 3:6] = 2.0  # dims
        iou = box3d_iou(box, box, format)
        assert iou.shape == (1, 1)
        assert abs(iou.item() - 1.0) < _IDENTITY_TOL


class TestBox3dIouMetaRegistration:
    @staticmethod
    def _ensure_loaded() -> None:
        # Touch box3d_iou so ``vision3d.ops`` imports (and
        # ``_meta_registrations`` runs) before FakeTensorMode is entered.
        box3d_iou(
            torch.tensor([[-1.0, -1, -1, 1, 1, 1]]),
            torch.tensor([[-1.0, -1, -1, 1, 1, 1]]),
            BoundingBox3DFormat.XYZXYZ,
        )

    def test_fake_tensor_mode_propagates_shapes(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        self._ensure_loaded()
        with FakeTensorMode():
            fake_b1 = torch.empty(5, 8, 3)
            fake_b2 = torch.empty(7, 8, 3)
            vol, iou = torch.ops.vision3d.iou_box3d(fake_b1, fake_b2)
            assert vol.shape == (5, 7)
            assert iou.shape == (5, 7)
            assert vol.dtype == torch.float32
            assert iou.dtype == torch.float32

    def test_fake_tensor_output_dtype_is_float32(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        self._ensure_loaded()
        with FakeTensorMode():
            fake_b1 = torch.empty(2, 8, 3, dtype=torch.float64)
            fake_b2 = torch.empty(3, 8, 3, dtype=torch.float64)
            vol, iou = torch.ops.vision3d.iou_box3d(fake_b1, fake_b2)
            assert vol.dtype == torch.float32
            assert iou.dtype == torch.float32

    def test_fake_tensor_rejects_wrong_shape(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        self._ensure_loaded()
        with FakeTensorMode():
            # Wrong trailing dim (4 instead of 3).
            bad = torch.empty(2, 8, 4)
            ok = torch.empty(3, 8, 3)
            with pytest.raises(RuntimeError, match="boxes1 must be"):
                torch.ops.vision3d.iou_box3d(bad, ok)

    def test_torch_compile_end_to_end(self) -> None:
        self._ensure_loaded()
        compiled = torch.compile(box3d_iou, fullgraph=False, dynamic=False)
        b1 = torch.tensor([[0.0, 0, 0, 1, 1, 1]])
        b2 = torch.tensor([[0.5, 0, 0, 1, 1, 1]])
        iou_eager = box3d_iou(b1, b2, BoundingBox3DFormat.XYZLWH)
        iou_compiled = compiled(b1, b2, BoundingBox3DFormat.XYZLWH)
        torch.testing.assert_close(iou_compiled, iou_eager)
