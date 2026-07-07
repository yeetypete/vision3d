"""Tests for ObjectMinPointsFilter."""

from typing import Any

import pytest
import torch
from common_utils import box_at

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, PointCloud3D
from vision3d.transforms import ObjectMinPointsFilter

_ALL_FORMATS = [
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWHYPR,
]


def _sample_with_counts(
    fmt: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an ``(inputs, targets)`` sample with known per-box point counts.

    Boxes are unit cubes (side 2) centered at x = 0, 10, 20. Point counts:
    box 0 -> 3, box 1 -> 1, box 2 -> 0. Two background points sit far away.

    Returns:
        ``(inputs, targets)`` dicts.
    """
    coords = [
        [0.0, 0, 0, 1],  # box 0
        [0.3, 0, 0, 1],  # box 0
        [-0.4, 0, 0, 1],  # box 0
        [10.0, 0, 0, 1],  # box 1
        [100.0, 0, 0, 1],  # background
        [0.0, 100, 0, 1],  # background
    ]
    points = PointCloud3D(torch.tensor(coords))
    boxes = BoundingBoxes3D(
        torch.tensor(
            [box_at(0, 0, fmt=fmt), box_at(10, 0, fmt=fmt), box_at(20, 0, fmt=fmt)]
        ),
        format=fmt,
    )
    labels = torch.tensor([0, 1, 2])
    return {"points": points}, {"boxes": boxes, "labels": labels}


class TestBoxFiltering:
    def test_keeps_boxes_meeting_threshold(self) -> None:
        inputs, targets = _sample_with_counts()
        _, out = ObjectMinPointsFilter(min_points=2)(inputs, targets)
        assert out["boxes"].shape[0] == 1
        assert out["labels"].tolist() == [0]

    def test_min_points_one_keeps_nonempty_boxes(self) -> None:
        inputs, targets = _sample_with_counts()
        _, out = ObjectMinPointsFilter(min_points=1)(inputs, targets)
        assert out["boxes"].shape[0] == 2
        assert out["labels"].tolist() == [0, 1]

    def test_min_points_zero_keeps_all(self) -> None:
        inputs, targets = _sample_with_counts()
        _, out = ObjectMinPointsFilter(min_points=0)(inputs, targets)
        assert out["boxes"].shape[0] == 3
        assert out["labels"].tolist() == [0, 1, 2]

    def test_high_threshold_drops_all(self) -> None:
        inputs, targets = _sample_with_counts()
        _, out = ObjectMinPointsFilter(min_points=99)(inputs, targets)
        assert out["boxes"].shape[0] == 0
        assert out["labels"].shape[0] == 0

    def test_overlap_point_counts_for_each_box(self) -> None:
        # A single point inside two overlapping boxes is counted once for
        # each box (membership, not first-box assignment), so at
        # min_points=1 both boxes survive.
        fmt = BoundingBox3DFormat.XYZLWHY
        points = PointCloud3D(torch.tensor([[0.5, 0.0, 0.0, 1.0]]))
        boxes = BoundingBoxes3D(
            torch.tensor([box_at(0, 0, fmt=fmt), box_at(1, 0, fmt=fmt)]), format=fmt
        )
        inputs = {"points": points}
        targets = {"boxes": boxes, "labels": torch.tensor([0, 1])}
        _, out = ObjectMinPointsFilter(min_points=1)(inputs, targets)
        assert out["boxes"].shape[0] == 2
        assert out["labels"].tolist() == [0, 1]

    def test_points_pass_through_unchanged(self) -> None:
        inputs, targets = _sample_with_counts()
        out_inputs, _ = ObjectMinPointsFilter(min_points=2)(inputs, targets)
        assert out_inputs["points"].shape[0] == 6

    def test_preserves_box_type_and_format(self) -> None:
        inputs, targets = _sample_with_counts()
        _, out = ObjectMinPointsFilter(min_points=2)(inputs, targets)
        assert isinstance(out["boxes"], BoundingBoxes3D)
        assert out["boxes"].format == BoundingBox3DFormat.XYZLWHYPR

    @pytest.mark.parametrize("fmt", _ALL_FORMATS)
    def test_format_agnostic(self, fmt: BoundingBox3DFormat) -> None:
        inputs, targets = _sample_with_counts(fmt=fmt)
        _, out = ObjectMinPointsFilter(min_points=2)(inputs, targets)
        assert out["boxes"].shape[0] == 1
        assert out["labels"].tolist() == [0]


class TestSingleDictSample:
    def test_filters_boxes_in_single_dict(self) -> None:
        inputs, targets = _sample_with_counts()
        sample = {**inputs, **targets}
        out = ObjectMinPointsFilter(min_points=2)(sample)
        assert out["boxes"].shape[0] == 1
        assert out["points"].shape[0] == 6
        assert out["labels"].tolist() == [0]


class TestEdgeCases:
    def test_no_points_keeps_all_boxes(self) -> None:
        _, targets = _sample_with_counts()
        inputs: dict[str, Any] = {}
        _, out = ObjectMinPointsFilter(min_points=5)(inputs, targets)
        assert out["boxes"].shape[0] == 3

    def test_empty_point_cloud_drops_all_boxes(self) -> None:
        _, targets = _sample_with_counts()
        inputs = {"points": PointCloud3D(torch.zeros(0, 4))}
        _, out = ObjectMinPointsFilter(min_points=1)(inputs, targets)
        assert out["boxes"].shape[0] == 0

    def test_no_boxes_key(self) -> None:
        inputs = {"points": PointCloud3D(torch.tensor([[0.0, 0, 0, 1]]))}
        targets: dict[str, Any] = {}
        _, out = ObjectMinPointsFilter(min_points=1)(inputs, targets)
        assert "boxes" not in out

    def test_empty_boxes(self) -> None:
        inputs = {"points": PointCloud3D(torch.tensor([[0.0, 0, 0, 1]]))}
        targets = {
            "boxes": BoundingBoxes3D(
                torch.zeros(0, 9), format=BoundingBox3DFormat.XYZLWHYPR
            ),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        _, out = ObjectMinPointsFilter(min_points=1)(inputs, targets)
        assert out["boxes"].shape[0] == 0

    def test_other_entries_pass_through(self) -> None:
        inputs, targets = _sample_with_counts()
        inputs["images"] = torch.randn(2, 3, 8, 8)
        out_inputs, _ = ObjectMinPointsFilter(min_points=2)(inputs, targets)
        assert torch.equal(out_inputs["images"], inputs["images"])


class TestValidation:
    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ObjectMinPointsFilter(min_points=-1)

    def test_non_int_raises(self) -> None:
        with pytest.raises(TypeError, match="non-negative"):
            ObjectMinPointsFilter(min_points=1.5)  # type: ignore[arg-type]

    def test_bool_raises(self) -> None:
        with pytest.raises(TypeError, match="non-negative"):
            ObjectMinPointsFilter(min_points=True)  # type: ignore[arg-type]
