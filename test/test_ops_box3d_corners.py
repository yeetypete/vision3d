"""Tests for box3d_corners op."""

import math

import pytest
import torch

from vision3d.ops import box3d_corners
from vision3d.tensors import BoundingBox3DFormat

ALL_FORMATS = [
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWHYPR,
]


# Format-specific tests
class TestAxisAligned:
    def test_xyzxyz_unit_box_at_origin(self) -> None:
        boxes = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZXYZ)
        assert corners.shape == (1, 8, 3)
        assert corners.abs().allclose(torch.ones_like(corners))

    def test_xyzlwh_unit_box_at_origin(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWH)
        assert corners.shape == (1, 8, 3)
        assert corners.abs().allclose(torch.ones_like(corners))

    def test_xyzlwh_offset_center(self) -> None:
        boxes = torch.tensor([[5.0, 3.0, 1.0, 2.0, 4.0, 6.0]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWH)
        assert corners[0, :, 0].min().item() == 4.0
        assert corners[0, :, 0].max().item() == 6.0
        assert corners[0, :, 1].min().item() == 1.0
        assert corners[0, :, 1].max().item() == 5.0
        assert corners[0, :, 2].min().item() == -2.0
        assert corners[0, :, 2].max().item() == 4.0


class TestRotated:
    def test_90_degree_yaw(self) -> None:
        yaw = math.pi / 2
        boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 2.0, yaw]])
        corners = box3d_corners(boxes, BoundingBox3DFormat.XYZLWHY)

        assert corners[0, :, 0].min().isclose(torch.tensor(-1.0), atol=1e-5)
        assert corners[0, :, 0].max().isclose(torch.tensor(1.0), atol=1e-5)
        assert corners[0, :, 1].min().isclose(torch.tensor(-2.0), atol=1e-5)
        assert corners[0, :, 1].max().isclose(torch.tensor(2.0), atol=1e-5)
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
        assert not torch.allclose(c1, c2, atol=1e-4)
        assert torch.allclose(c1.mean(dim=1), c2.mean(dim=1), atol=1e-5)


# Properties that hold across all formats
class TestProperties:
    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_output_shape(self, fmt: BoundingBox3DFormat) -> None:
        n_cols = {
            BoundingBox3DFormat.XYZXYZ: 6,
            BoundingBox3DFormat.XYZLWH: 6,
            BoundingBox3DFormat.XYZLWHY: 7,
            BoundingBox3DFormat.XYZLWHYPR: 9,
        }
        boxes = torch.rand(5, n_cols[fmt])
        corners = box3d_corners(boxes, fmt)
        assert corners.shape == (5, 8, 3)

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_empty_batch(self, fmt: BoundingBox3DFormat) -> None:
        n_cols = {
            BoundingBox3DFormat.XYZXYZ: 6,
            BoundingBox3DFormat.XYZLWH: 6,
            BoundingBox3DFormat.XYZLWHY: 7,
            BoundingBox3DFormat.XYZLWHYPR: 9,
        }
        boxes = torch.zeros(0, n_cols[fmt])
        corners = box3d_corners(boxes, fmt)
        assert corners.shape == (0, 8, 3)

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_center_of_corners_equals_box_center(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        from common_utils import make_bounding_boxes_3d

        boxes = make_bounding_boxes_3d(format=fmt, num_boxes=3)
        corners = box3d_corners(boxes.as_subclass(torch.Tensor), fmt)
        # Mean of 8 corners = box center for any rotation
        computed_centers = corners.mean(dim=1)
        if fmt is BoundingBox3DFormat.XYZXYZ:
            expected = (boxes[:, :3] + boxes[:, 3:6]) / 2
        else:
            expected = boxes[:, :3]
        assert torch.allclose(
            computed_centers, expected.as_subclass(torch.Tensor), atol=1e-5
        )
