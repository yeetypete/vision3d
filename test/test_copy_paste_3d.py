"""Tests for CopyPaste3D transform."""

from typing import Any

import torch
from common_utils import make_bounding_boxes_3d

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, PointCloud3D
from vision3d.transforms import CopyPaste3D


def _make_batch(
    batch_size: int = 2,
    num_points_per_box: int = 20,
    num_boxes: int = 3,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    # Create batch with points guaranteed to be inside the boxes.
    inputs = []
    targets = []
    for _ in range(batch_size):
        boxes = make_bounding_boxes_3d(
            format=BoundingBox3DFormat.XYZLWHY, num_boxes=num_boxes
        )
        raw = boxes.as_subclass(torch.Tensor)

        # Generate points inside each box
        all_points = []
        for j in range(num_boxes):
            cx, cy, cz = raw[j, 0], raw[j, 1], raw[j, 2]
            l, w, h = raw[j, 3], raw[j, 4], raw[j, 5]
            local = (torch.rand(num_points_per_box, 3) - 0.5) * torch.tensor([l, w, h])
            local[:, 0] += cx
            local[:, 1] += cy
            local[:, 2] += cz
            all_points.append(local)

        points = torch.cat(all_points)
        inp = {"points": PointCloud3D(points)}
        tgt = {
            "boxes": boxes,
            "labels": torch.arange(num_boxes, dtype=torch.long),
            "class_names": ["Car"] * num_boxes,
        }
        inputs.append(inp)
        targets.append(tgt)
    return tuple(inputs), tuple(targets)


class TestCopyPaste3DFirstBatch:
    def test_first_batch_populates_database(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
        )
        inputs, targets = _make_batch(batch_size=2, num_boxes=3)
        cp(inputs, targets)

        # Database should be populated after first batch
        assert len(cp._database["Car"]) > 0

    def test_database_populated(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        inputs, targets = _make_batch(batch_size=2, num_boxes=3)
        cp(inputs, targets)
        assert len(cp._database["Car"]) > 0


class TestCopyPaste3DPasting:
    def test_second_batch_pastes(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch1)

        batch2 = _make_batch(batch_size=2, num_boxes=2)
        _, out_targets = cp(*batch2)

        original_count = 2
        any_pasted = any(
            out_targets[i]["boxes"].shape[0] > original_count for i in range(2)
        )
        assert any_pasted

    def test_box_count_increases(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=5)
        cp(*batch1)

        batch2 = _make_batch(batch_size=1, num_boxes=2)
        _, out_targets = cp(*batch2)
        assert out_targets[0]["boxes"].shape[0] >= 2

    def test_class_names_updated(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch1)

        batch2 = _make_batch(batch_size=1, num_boxes=1)
        _, out_targets = cp(*batch2)

        n_boxes = out_targets[0]["boxes"].shape[0]
        assert len(out_targets[0]["class_names"]) == n_boxes


class TestCopyPaste3DTypes:
    def test_preserves_point_cloud_type(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch1)

        batch2 = _make_batch(batch_size=1, num_boxes=1)
        out_inputs, _ = cp(*batch2)
        assert isinstance(out_inputs[0]["points"], PointCloud3D)

    def test_preserves_bounding_boxes_type(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch1)

        batch2 = _make_batch(batch_size=1, num_boxes=1)
        _, out_targets = cp(*batch2)
        assert isinstance(out_targets[0]["boxes"], BoundingBoxes3D)
        assert out_targets[0]["boxes"].format == BoundingBox3DFormat.XYZLWHY


class TestCopyPaste3DProbability:
    def test_p_zero_no_paste(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            p=0.0,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch1)

        batch2 = _make_batch(batch_size=1, num_boxes=2)
        _, out_targets = cp(*batch2)
        assert out_targets[0]["boxes"].shape[0] == 2


class TestCopyPaste3DDatabase:
    def test_database_grows(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch1)
        size_after_1 = len(cp._database["Car"])

        batch2 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch2)
        size_after_2 = len(cp._database["Car"])
        assert size_after_2 > size_after_1

    def test_max_database_size(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
            max_database_size=5,
        )
        for _ in range(10):
            batch = _make_batch(batch_size=2, num_boxes=3)
            cp(*batch)
        assert len(cp._database["Car"]) <= 5

    def test_min_points_filter(self) -> None:
        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=9999,
        )
        batch = _make_batch(batch_size=2, num_boxes=3, num_points_per_box=2)
        cp(*batch)
        assert len(cp._database["Car"]) == 0


class TestCopyPaste3DCollision:
    def test_no_overlap_with_existing(self) -> None:
        from vision3d.ops import box3d_overlap_bev

        cp = CopyPaste3D(
            target_counts={"Car": 10},
            min_points=1,
        )
        batch1 = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch1)

        batch2 = _make_batch(batch_size=1, num_boxes=2)
        _, out_targets = cp(*batch2)

        boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        if boxes.shape[0] > 1:
            overlap = box3d_overlap_bev(boxes, boxes, BoundingBox3DFormat.XYZLWHY)
            overlap.fill_diagonal_(False)
            assert not overlap.any()
