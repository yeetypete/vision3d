"""Tests for box3d_corners op."""

import math

import torch

from vision3d.ops import box3d_corners
from vision3d.tensors import BoundingBox3DFormat


class TestBox3dCornersAxisAligned:
    def test_xyzxyz_unit_box_at_origin(self) -> None:
        boxes = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZXYZ)
        assert corners.shape == (1, 8, 3)
        # All corners should be at +-1 in each dimension
        assert corners.abs().allclose(torch.ones_like(corners))

    def test_xyzlwh_unit_box_at_origin(self) -> None:
        # center=(0,0,0), l=2, w=2, h=2  -> same as XYZXYZ(-1,-1,-1,1,1,1)
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWH)
        assert corners.shape == (1, 8, 3)
        assert corners.abs().allclose(torch.ones_like(corners))

    def test_xyzlwh_offset_center(self) -> None:
        boxes = torch.tensor([[5.0, 3.0, 1.0, 2.0, 4.0, 6.0]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWH)
        # Center at (5,3,1), half-dims (1,2,3)
        assert corners[0, :, 0].min().item() == 4.0
        assert corners[0, :, 0].max().item() == 6.0
        assert corners[0, :, 1].min().item() == 1.0
        assert corners[0, :, 1].max().item() == 5.0
        assert corners[0, :, 2].min().item() == -2.0
        assert corners[0, :, 2].max().item() == 4.0


class TestBox3dCornersRotated:
    def test_90_degree_yaw(self) -> None:
        # XYZLWHY: center=(0,0,0), l=4, w=2, h=2, yaw=pi/2
        yaw = math.pi / 2
        boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 2.0, yaw]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWHY)

        # After 90-deg yaw, length (X) maps to Y, width (Y) maps to -X
        # Half-dims: hl=2, hw=1, hh=1
        # X range should be [-1, 1] (was Y range before rotation)
        # Y range should be [-2, 2] (was X range before rotation)
        assert corners[0, :, 0].min().isclose(torch.tensor(-1.0), atol=1e-5)
        assert corners[0, :, 0].max().isclose(torch.tensor(1.0), atol=1e-5)
        assert corners[0, :, 1].min().isclose(torch.tensor(-2.0), atol=1e-5)
        assert corners[0, :, 1].max().isclose(torch.tensor(2.0), atol=1e-5)
        # Z unchanged
        assert corners[0, :, 2].min().isclose(torch.tensor(-1.0), atol=1e-5)
        assert corners[0, :, 2].max().isclose(torch.tensor(1.0), atol=1e-5)

    def test_zero_yaw_matches_xyzlwh(self) -> None:
        boxes_lwh = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        boxes_lwhy = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]])
        c1 = box3d_corners(boxes_lwh, BoundingBox3DFormat.XYZLWH)
        c2 = box3d_corners(boxes_lwhy, BoundingBox3DFormat.XYZLWHY)
        assert torch.allclose(c1, c2, atol=1e-6)

    def test_xyzlwhypr_zero_pitch_roll_matches_yaw_only(self) -> None:
        yaw = 0.5
        boxes_y = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, yaw]])
        boxes_ypr = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, yaw, 0.0, 0.0]])
        c1 = box3d_corners(boxes_y, BoundingBox3DFormat.XYZLWHY)
        c2 = box3d_corners(boxes_ypr, BoundingBox3DFormat.XYZLWHYPR)
        assert torch.allclose(c1, c2, atol=1e-6)

    def test_xyzlwhypr_with_pitch_differs_from_yaw_only(self) -> None:
        yaw = 0.5
        boxes_y = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, yaw]])
        boxes_ypr = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, yaw, 0.3, 0.0]])
        c1 = box3d_corners(boxes_y, BoundingBox3DFormat.XYZLWHY)
        c2 = box3d_corners(boxes_ypr, BoundingBox3DFormat.XYZLWHYPR)
        # With non-zero pitch, corners should differ
        assert not torch.allclose(c1, c2, atol=1e-4)
        # But center should still be the same
        assert torch.allclose(c1.mean(dim=1), c2.mean(dim=1), atol=1e-5)


class TestBox3dCornersBatch:
    def test_batch_shape(self) -> None:
        boxes = torch.rand(5, 7)
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWHY)
        assert corners.shape == (5, 8, 3)

    def test_empty_batch(self) -> None:
        boxes = torch.zeros(0, 6)
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWH)
        assert corners.shape == (0, 8, 3)


class TestBox3dCornersProperties:
    def test_center_of_corners_equals_box_center(self) -> None:
        boxes = torch.tensor([[3.0, 7.0, -2.0, 4.0, 6.0, 8.0, 1.0]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWHY)
        center = corners[0].mean(dim=0)
        assert center[0].isclose(torch.tensor(3.0), atol=1e-5)
        assert center[1].isclose(torch.tensor(7.0), atol=1e-5)
        assert center[2].isclose(torch.tensor(-2.0), atol=1e-5)
