"""Tests for box3d_iou_bev op."""

import math

import torch

from vision3d.ops import box3d_iou_bev, box3d_overlap_bev
from vision3d.tensors import BoundingBox3DFormat


class TestBox3dIouBev:
    def test_identical_boxes(self) -> None:
        boxes = torch.tensor([[0.0, 0, 0, 4, 2, 2, 0]])
        iou = box3d_iou_bev(boxes, boxes, BoundingBox3DFormat.XYZLWHY)
        torch.testing.assert_close(iou, torch.tensor([[1.0]]))

    def test_non_overlapping(self) -> None:
        boxes1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0]])
        boxes2 = torch.tensor([[10.0, 0, 0, 2, 2, 2, 0]])
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert iou[0, 0].item() == 0.0

    def test_partial_overlap(self) -> None:
        # Box 1: center (0,0), l=2, w=2 → BEV [-1,1] x [-1,1], area=4
        # Box 2: center (1,0), l=2, w=2 → BEV [0,2] x [-1,1], area=4
        # Intersection: [0,1] x [-1,1] = 1*2 = 2
        # Union: 4+4-2 = 6
        # IoU: 2/6 = 1/3
        boxes1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0]])
        boxes2 = torch.tensor([[1.0, 0, 0, 2, 2, 2, 0]])
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        torch.testing.assert_close(iou, torch.tensor([[1.0 / 3]]), atol=1e-5, rtol=1e-5)

    def test_symmetry(self) -> None:
        boxes1 = torch.tensor([[0.0, 0, 0, 4, 2, 2, 0.3], [3.0, 0, 0, 2, 2, 2, 0]])
        boxes2 = torch.tensor([[1.0, 1, 0, 3, 3, 2, -0.2]])
        iou_12 = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        iou_21 = box3d_iou_bev(boxes2, boxes1, BoundingBox3DFormat.XYZLWHY)
        torch.testing.assert_close(iou_12, iou_21.T)

    def test_output_shape(self) -> None:
        boxes1 = torch.rand(3, 7)
        boxes2 = torch.rand(5, 7)
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert iou.shape == (3, 5)

    def test_empty_boxes(self) -> None:
        boxes1 = torch.zeros(0, 7)
        boxes2 = torch.rand(3, 7)
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert iou.shape == (0, 3)

    def test_xyzxyz_format(self) -> None:
        # Box: [0,0,0] to [2,2,2], BEV area = 4
        # Same box → IoU = 1
        boxes = torch.tensor([[0.0, 0, 0, 2, 2, 2]])
        iou = box3d_iou_bev(boxes, boxes, BoundingBox3DFormat.XYZXYZ)
        torch.testing.assert_close(iou, torch.tensor([[1.0]]))

    def test_xyzlwh_format(self) -> None:
        boxes = torch.tensor([[0.0, 0, 0, 2, 2, 2]])
        iou = box3d_iou_bev(boxes, boxes, BoundingBox3DFormat.XYZLWH)
        torch.testing.assert_close(iou, torch.tensor([[1.0]]))

    def test_rotated_90_degrees(self) -> None:
        # Box1: 4x2 at yaw=0, Box2: 4x2 at yaw=90°
        # Both centered at origin. Intersection is a 2x2 square.
        # Each area=8, intersection=4, union=12, IoU=1/3
        boxes1 = torch.tensor([[0.0, 0, 0, 4, 2, 2, 0]])
        boxes2 = torch.tensor([[0.0, 0, 0, 4, 2, 2, math.pi / 2]])
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        torch.testing.assert_close(iou, torch.tensor([[1.0 / 3]]), atol=1e-4, rtol=1e-4)

    def test_self_iou_rotated(self) -> None:
        # Same box at 45° should have IoU=1 with itself
        boxes = torch.tensor([[0.0, 0, 0, 4, 2, 2, math.pi / 4]])
        iou = box3d_iou_bev(boxes, boxes, BoundingBox3DFormat.XYZLWHY)
        torch.testing.assert_close(iou, torch.tensor([[1.0]]), atol=1e-4, rtol=1e-4)

    def test_no_overlap_rotated(self) -> None:
        # Two rotated boxes far apart
        boxes1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0.5]])
        boxes2 = torch.tensor([[20.0, 0, 0, 2, 2, 2, -0.3]])
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert iou[0, 0].item() == 0.0

    def test_values_in_range(self) -> None:
        boxes1 = torch.rand(10, 7)
        boxes1[:, 3:6] = boxes1[:, 3:6].abs() + 0.1  # positive dims
        boxes2 = torch.rand(8, 7)
        boxes2[:, 3:6] = boxes2[:, 3:6].abs() + 0.1
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert (iou >= 0).all()
        assert (iou <= 1).all()


class TestBox3dOverlapBev:
    def test_overlap(self) -> None:
        boxes1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0]])
        boxes2 = torch.tensor([[0.5, 0, 0, 2, 2, 2, 0]])
        overlap = box3d_overlap_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert overlap[0, 0].item() is True

    def test_no_overlap(self) -> None:
        boxes1 = torch.tensor([[0.0, 0, 0, 2, 2, 2, 0]])
        boxes2 = torch.tensor([[10.0, 0, 0, 2, 2, 2, 0]])
        overlap = box3d_overlap_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert overlap[0, 0].item() is False

    def test_consistent_with_iou(self) -> None:
        boxes1 = torch.rand(5, 7)
        boxes1[:, 3:6] = boxes1[:, 3:6].abs() + 0.1
        boxes2 = torch.rand(3, 7)
        boxes2[:, 3:6] = boxes2[:, 3:6].abs() + 0.1
        iou = box3d_iou_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        overlap = box3d_overlap_bev(boxes1, boxes2, BoundingBox3DFormat.XYZLWHY)
        assert (overlap == (iou > 0)).all()
