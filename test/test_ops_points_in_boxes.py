"""Tests for points_in_boxes_3d op."""

import math

import pytest
import torch

from vision3d.ops import points_in_boxes_3d, points_in_boxes_3d_indices
from vision3d.tensors import BoundingBox3DFormat


class TestPointsInBoxes3D:
    def test_point_inside(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0]])
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask[0, 0].item() is True

    def test_point_outside(self) -> None:
        points = torch.tensor([[5.0, 0.0, 0.0]])
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask[0, 0].item() is False

    def test_point_on_boundary(self) -> None:
        points = torch.tensor([[1.0, 0.0, 0.0]])
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask[0, 0].item() is True

    def test_yaw_rotation(self) -> None:
        # Box centered at origin, l=4, w=2, h=2, rotated 90 degrees
        # After rotation, length aligns with Y axis
        boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 2.0, math.pi / 2]])
        # Point at (0, 1.5, 0): inside the rotated box (within length along Y)
        # Point at (1.5, 0, 0): outside (width along X is only 2, but rotated)
        points = torch.tensor([[0.0, 1.5, 0.0], [1.5, 0.0, 0.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask[0, 0].item() is True  # inside rotated box
        assert mask[1, 0].item() is False  # outside rotated box

    def test_multiple_boxes(self) -> None:
        points = torch.tensor([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask.shape == (2, 2)
        assert mask[0, 0].item() is True  # point 0 in box 0
        assert mask[0, 1].item() is False  # point 0 not in box 1
        assert mask[1, 0].item() is False  # point 1 not in box 0
        assert mask[1, 1].item() is True  # point 1 in box 1

    def test_empty_boxes(self) -> None:
        points = torch.rand(10, 3)
        boxes = torch.zeros(0, 7)
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask.shape == (10, 0)

    def test_empty_points(self) -> None:
        points = torch.zeros(0, 3)
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask.shape == (0, 1)

    def test_extra_features_ignored(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.8]])
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask[0, 0].item() is True

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype: torch.dtype) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=dtype)
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert mask.dtype == torch.bool

    def test_xyzxyz_format(self) -> None:
        points = torch.tensor([[0.5, 0.5, 0.5]])
        boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZXYZ)
        assert mask[0, 0].item() is True

    def test_xyzlwh_format(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0]])
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWH)
        assert mask[0, 0].item() is True

    def test_z_check(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        inside = torch.tensor([[0.0, 0.0, 0.5]])
        outside = torch.tensor([[0.0, 0.0, 1.5]])
        assert points_in_boxes_3d(inside, boxes, BoundingBox3DFormat.XYZLWHY)[0, 0]
        assert not points_in_boxes_3d(outside, boxes, BoundingBox3DFormat.XYZLWHY)[0, 0]


class TestPointsInBoxes3DIndices:
    def test_basic(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [99.0, 0.0, 0.0]])
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )
        indices = points_in_boxes_3d_indices(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert indices[0].item() == 0
        assert indices[1].item() == 1
        assert indices[2].item() == -1

    def test_overlapping_boxes_first_wins(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0]])
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )
        indices = points_in_boxes_3d_indices(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert indices[0].item() == 0  # first box wins

    def test_no_boxes(self) -> None:
        points = torch.rand(5, 3)
        boxes = torch.zeros(0, 7)
        indices = points_in_boxes_3d_indices(points, boxes, BoundingBox3DFormat.XYZLWHY)
        assert (indices == -1).all()

    def test_consistent_with_mask(self) -> None:
        points = torch.rand(50, 3) * 10 - 5
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0],
                [3.0, 3.0, 0.0, 2.0, 2.0, 2.0, 0.3],
            ]
        )
        mask = points_in_boxes_3d(points, boxes, BoundingBox3DFormat.XYZLWHY)
        indices = points_in_boxes_3d_indices(points, boxes, BoundingBox3DFormat.XYZLWHY)
        for i in range(len(points)):
            if indices[i] == -1:
                assert not mask[i].any()
            else:
                assert mask[i, indices[i]]
