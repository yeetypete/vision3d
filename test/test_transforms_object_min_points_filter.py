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
    @pytest.mark.parametrize(
        ("min_points", "expected_labels"),
        [(0, [0, 1, 2]), (1, [0, 1]), (2, [0]), (99, [])],
    )
    def test_threshold(self, min_points: int, expected_labels: list[int]) -> None:
        # Per-box counts: box 0 -> 3, box 1 -> 1, box 2 -> 0.
        inputs, targets = _sample_with_counts()
        _, out = ObjectMinPointsFilter(min_points=min_points)(inputs, targets)
        assert out["labels"].tolist() == expected_labels
        assert out["boxes"].shape[0] == len(expected_labels)

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

    def test_mixed_point_box_dtype(self) -> None:
        # float64 points (e.g. loaded from a numpy float64 array) with
        # float32 boxes must not raise a dtype mismatch.
        inputs, targets = _sample_with_counts()
        pts = inputs["points"].as_subclass(torch.Tensor).double()
        inputs["points"] = PointCloud3D(pts)
        targets["boxes"] = BoundingBoxes3D(
            targets["boxes"].as_subclass(torch.Tensor).float(),
            format=targets["boxes"].format,
        )
        _, out = ObjectMinPointsFilter(min_points=2)(inputs, targets)
        assert out["boxes"].shape[0] == 1
        assert out["labels"].tolist() == [0]

    def test_pair_inputs_dict_not_aliased(self) -> None:
        # The returned inputs dict is a fresh copy (matches RangeFilter3D),
        # so downstream mutation cannot corrupt the caller's sample.
        inputs, targets = _sample_with_counts()
        out_inputs, _ = ObjectMinPointsFilter(min_points=2)(inputs, targets)
        assert out_inputs is not inputs


class TestStructureAgnostic:
    def test_non_standard_dict_keys(self) -> None:
        # Boxes and points live under non-"boxes"/"points" keys; they are
        # located by leaf type, not by key name.
        inputs, targets = _sample_with_counts()
        sample = {
            "lidar": inputs["points"],
            "gt_boxes": targets["boxes"],
            "labels": targets["labels"],
            "frame_id": 7,
        }
        out = ObjectMinPointsFilter(min_points=1)(sample)
        assert out["lidar"].shape[0] == 6
        assert out["gt_boxes"].shape[0] == 2
        assert out["labels"].tolist() == [0, 1]
        assert out["frame_id"] == 7

    def test_multiple_box_sets_raises(self) -> None:
        inputs, targets = _sample_with_counts()
        targets = {
            "boxes": targets["boxes"],
            "pred_boxes": BoundingBoxes3D(
                targets["boxes"].as_subclass(torch.Tensor).clone(),
                format=targets["boxes"].format,
            ),
            "labels": targets["labels"],
        }
        with pytest.raises(ValueError, match="multiple BoundingBoxes3D"):
            ObjectMinPointsFilter(min_points=1)(inputs, targets)

    def test_multiple_point_clouds_raises(self) -> None:
        inputs, targets = _sample_with_counts()
        inputs = {
            "points": inputs["points"],
            "extra_points": PointCloud3D(
                inputs["points"].as_subclass(torch.Tensor).clone()
            ),
        }
        with pytest.raises(ValueError, match="multiple PointCloud3D"):
            ObjectMinPointsFilter(min_points=1)(inputs, targets)


class TestLabelsGetter:
    def test_default_getter_is_case_insensitive(self) -> None:
        inputs, targets = _sample_with_counts()
        targets = {"boxes": targets["boxes"], "Labels": targets["labels"]}
        _, out = ObjectMinPointsFilter(min_points=1)(inputs, targets)
        assert out["Labels"].tolist() == [0, 1]

    def test_custom_labels_getter(self) -> None:
        inputs, targets = _sample_with_counts()
        targets = {"boxes": targets["boxes"], "gt_labels": targets["labels"]}
        f = ObjectMinPointsFilter(
            min_points=1, labels_getter=lambda s: s[1]["gt_labels"]
        )
        _, out = f(inputs, targets)
        assert out["gt_labels"].tolist() == [0, 1]

    def test_custom_getter_returning_tuple_of_tensors(self) -> None:
        inputs, targets = _sample_with_counts()
        targets = {
            "boxes": targets["boxes"],
            "labels": targets["labels"],
            "attributes": torch.tensor([10, 11, 12]),
        }
        f = ObjectMinPointsFilter(
            min_points=1,
            labels_getter=lambda s: (s[1]["labels"], s[1]["attributes"]),
        )
        _, out = f(inputs, targets)
        assert out["boxes"].shape[0] == 2
        assert out["labels"].tolist() == [0, 1]
        assert out["attributes"].tolist() == [10, 11]

    def test_none_getter_filters_boxes_but_not_labels(self) -> None:
        inputs, targets = _sample_with_counts()
        f = ObjectMinPointsFilter(min_points=1, labels_getter=None)
        _, out = f(inputs, targets)
        assert out["boxes"].shape[0] == 2
        assert out["labels"].shape[0] == 3

    def test_default_getter_missing_labels_with_boxes_raises(self) -> None:
        inputs, targets = _sample_with_counts()
        targets = {"boxes": targets["boxes"]}
        with pytest.raises(ValueError, match="could not find a labels tensor"):
            ObjectMinPointsFilter(min_points=1)(inputs, targets)

    def test_getter_returning_copy_raises(self) -> None:
        inputs, targets = _sample_with_counts()
        f = ObjectMinPointsFilter(
            min_points=1, labels_getter=lambda s: s[1]["labels"].clone()
        )
        with pytest.raises(ValueError, match="leaves of the sample"):
            f(inputs, targets)

    def test_mismatched_label_length_raises(self) -> None:
        inputs, targets = _sample_with_counts()
        targets = {"boxes": targets["boxes"], "labels": torch.tensor([0, 1])}
        with pytest.raises(ValueError, match="labels must be per-box"):
            ObjectMinPointsFilter(min_points=1)(inputs, targets)

    def test_invalid_labels_getter_raises(self) -> None:
        bad_getter: Any = 123
        with pytest.raises(ValueError, match="labels_getter"):
            ObjectMinPointsFilter(min_points=1, labels_getter=bad_getter)


class TestValidation:
    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ObjectMinPointsFilter(min_points=-1)

    def test_non_int_raises(self) -> None:
        bad_min_points: Any = 1.5
        with pytest.raises(TypeError, match="non-negative"):
            ObjectMinPointsFilter(min_points=bad_min_points)

    def test_bool_raises(self) -> None:
        bad_min_points: Any = True
        with pytest.raises(TypeError, match="non-negative"):
            ObjectMinPointsFilter(min_points=bad_min_points)
