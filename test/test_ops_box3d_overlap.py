"""Tests for the box3d_overlap SAT op."""

import math

import torch

from vision3d.ops import box3d_overlap
from vision3d.tensors import BoundingBox3DFormat


class TestBox3DOverlap:
    def test_identical_boxes_overlap(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        assert bool(box3d_overlap(boxes, boxes, BoundingBox3DFormat.XYZLWHY)[0, 0])

    def test_disjoint_boxes_do_not_overlap(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        b2 = torch.tensor([[100.0, 0.0, 0.0, 2.0, 2.0, 2.0, math.pi / 4]])
        assert not bool(box3d_overlap(b1, b2, BoundingBox3DFormat.XYZLWHY)[0, 0])

    def test_rotation_direction(self) -> None:
        # A long thin box centered at (2, 2) sweeps into the unit cube at the
        # origin at +45 degrees but points away from it at -45 degrees, so the
        # overlap result pins down the sign of the box rotation.
        fmt = BoundingBox3DFormat.XYZLWHY
        cube = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])  # [-1, 1]^3
        plus = torch.tensor([[2.0, 2.0, 0.0, 6.0, 0.6, 2.0, math.pi / 4]])
        minus = torch.tensor([[2.0, 2.0, 0.0, 6.0, 0.6, 2.0, -math.pi / 4]])
        assert bool(box3d_overlap(cube, plus, fmt)[0, 0]) is True
        assert bool(box3d_overlap(cube, minus, fmt)[0, 0]) is False

    def test_9dof_rotation(self) -> None:
        # A box tilted about pitch, stacked in z, still overlaps because the
        # tilt brings a corner down into the lower box.
        fmt = BoundingBox3DFormat.XYZLWHYPR
        lower = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]])
        tilted = torch.tensor([[0.0, 0.0, 1.4, 2.0, 2.0, 2.0, 0.0, math.pi / 4, 0.0]])
        assert bool(box3d_overlap(lower, tilted, fmt)[0, 0]) is True

    def test_pairwise_shape(self) -> None:
        b1 = torch.zeros(3, 7)
        b1[:, 3:6] = 2.0
        b2 = torch.zeros(5, 7)
        b2[:, 3:6] = 2.0
        out = box3d_overlap(b1, b2, BoundingBox3DFormat.XYZLWHY)
        assert out.shape == (3, 5)

    def test_mixed_dtype(self) -> None:
        # float32 and float64 box sets must not raise a dtype mismatch.
        b1 = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=torch.float32)
        b2 = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=torch.float64)
        assert bool(box3d_overlap(b1, b2, BoundingBox3DFormat.XYZLWHY)[0, 0])
