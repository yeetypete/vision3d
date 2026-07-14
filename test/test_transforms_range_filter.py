"""Tests for RangeFilter3D."""

from typing import Any

import pytest
import torch
from common_utils import box_at

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, PointCloud3D
from vision3d.transforms import RangeFilter3D

_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)

_ALL_FORMATS = [
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWHYPR,
]


def _make_two_dict_sample(
    fmt: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    points = PointCloud3D(
        torch.tensor(
            [
                [0.0, 0, 0, 1],  # in range
                [5.0, 5, 0, 1],  # in range
                [50.0, 0, 0, 1],  # out of range
                [0.0, 0, 10, 1],  # out of range (z)
            ]
        )
    )
    boxes = BoundingBoxes3D(
        torch.tensor(
            [
                box_at(0, 0, fmt=fmt),  # in range
                box_at(5, 5, fmt=fmt),  # in range
                box_at(50, 0, fmt=fmt),  # out of range
            ]
        ),
        format=fmt,
    )
    labels = torch.tensor([0, 1, 2])
    inputs = {"points": points}
    targets = {"boxes": boxes, "labels": labels}
    return inputs, targets


def _make_single_dict_sample(
    fmt: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
) -> dict[str, Any]:
    points = PointCloud3D(
        torch.tensor(
            [
                [0.0, 0, 0, 1],
                [50.0, 0, 0, 1],
            ]
        )
    )
    boxes = BoundingBoxes3D(
        torch.tensor(
            [
                box_at(0, 0, fmt=fmt),
                box_at(50, 0, fmt=fmt),
            ]
        ),
        format=fmt,
    )
    return {"points": points, "boxes": boxes, "labels": torch.tensor([0, 1])}


class TestPointFiltering:
    def test_keeps_points_inside_range(self) -> None:
        inputs, targets = _make_two_dict_sample()
        f = RangeFilter3D(point_cloud_range=_RANGE)
        out_inputs, _ = f(inputs, targets)
        assert out_inputs["points"].shape[0] == 2

    def test_removes_points_outside_range(self) -> None:
        points = PointCloud3D(torch.tensor([[50.0, 0, 0, 1], [0, 50, 0, 1]]))
        inputs = {"points": points}
        targets = {
            "boxes": BoundingBoxes3D(
                torch.zeros(0, 9), format=BoundingBox3DFormat.XYZLWHYPR
            ),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        out_inputs, _ = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_inputs["points"].shape[0] == 0

    def test_preserves_features(self) -> None:
        inputs, targets = _make_two_dict_sample()
        out_inputs, _ = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_inputs["points"].shape[1] == 4

    def test_preserves_type(self) -> None:
        inputs, targets = _make_two_dict_sample()
        out_inputs, _ = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert isinstance(out_inputs["points"], PointCloud3D)


class TestBoxFiltering:
    def test_keeps_boxes_with_center_in_range(self) -> None:
        inputs, targets = _make_two_dict_sample()
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_targets["boxes"].shape[0] == 2

    def test_labels_filtered_in_sync(self) -> None:
        inputs, targets = _make_two_dict_sample()
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_targets["labels"].shape[0] == 2
        assert out_targets["labels"].tolist() == [0, 1]

    def test_preserves_box_type(self) -> None:
        inputs, targets = _make_two_dict_sample()
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert isinstance(out_targets["boxes"], BoundingBoxes3D)

    def test_preserves_format(self) -> None:
        inputs, targets = _make_two_dict_sample()
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_targets["boxes"].format == BoundingBox3DFormat.XYZLWHYPR

    @pytest.mark.parametrize("fmt", _ALL_FORMATS)
    def test_format_agnostic(self, fmt: BoundingBox3DFormat) -> None:
        inputs, targets = _make_two_dict_sample(fmt=fmt)
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_targets["boxes"].shape[0] == 2
        assert out_targets["labels"].tolist() == [0, 1]


class TestSingleDictSample:
    def test_filters_both_points_and_boxes(self) -> None:
        sample = _make_single_dict_sample()
        out = RangeFilter3D(point_cloud_range=_RANGE)(sample)
        assert out["points"].shape[0] == 1
        assert out["boxes"].shape[0] == 1
        assert out["labels"].tolist() == [0]


class TestEdgeCases:
    def test_all_in_range(self) -> None:
        inputs, targets = _make_two_dict_sample()
        wide = (-100.0, -100.0, -100.0, 100.0, 100.0, 100.0)
        out_inputs, out_targets = RangeFilter3D(point_cloud_range=wide)(inputs, targets)
        assert out_inputs["points"].shape[0] == 4
        assert out_targets["boxes"].shape[0] == 3

    def test_all_out_of_range(self) -> None:
        inputs, targets = _make_two_dict_sample()
        tiny = (99.0, 99.0, 99.0, 100.0, 100.0, 100.0)
        out_inputs, out_targets = RangeFilter3D(point_cloud_range=tiny)(inputs, targets)
        assert out_inputs["points"].shape[0] == 0
        assert out_targets["boxes"].shape[0] == 0
        assert out_targets["labels"].shape[0] == 0

    def test_no_points_key(self) -> None:
        targets = {
            "boxes": BoundingBoxes3D(
                torch.tensor([box_at(0, 0, fmt=BoundingBox3DFormat.XYZLWHYPR)]),
                format=BoundingBox3DFormat.XYZLWHYPR,
            ),
            "labels": torch.tensor([0]),
        }
        inputs: dict[str, torch.Tensor] = {}
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_targets["boxes"].shape[0] == 1

    def test_no_boxes_key(self) -> None:
        points = PointCloud3D(torch.tensor([[0.0, 0, 0, 1]]))
        inputs = {"points": points}
        targets: dict[str, torch.Tensor] = {}
        out_inputs, _ = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_inputs["points"].shape[0] == 1

    def test_camera_data_passthrough(self) -> None:
        inputs: dict[str, Any] = dict(_make_two_dict_sample()[0])
        targets = _make_two_dict_sample()[1]
        inputs["images"] = torch.randn(2, 3, 16, 24)
        inputs["intrinsics"] = torch.eye(3).unsqueeze(0)
        out_inputs, _ = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert torch.equal(out_inputs["images"], inputs["images"])
        assert torch.equal(out_inputs["intrinsics"], inputs["intrinsics"])

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="6 elements"):
            RangeFilter3D(point_cloud_range=(0.0, 0.0, 0.0))

    def test_no_boxes_passes_through_non_point_leaves(self) -> None:
        points = PointCloud3D(torch.tensor([[0.0, 0, 0, 1], [50.0, 0, 0, 1]]))
        image = torch.randn(2, 3, 16, 24)
        sample = {"points": points, "images": image, "frame_id": 7}
        out = RangeFilter3D(point_cloud_range=_RANGE)(sample)
        assert out["points"].shape[0] == 1
        assert torch.equal(out["images"], image)
        assert out["frame_id"] == 7

    def test_multiple_box_sets_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {
            "boxes": targets["boxes"],
            "pred_boxes": BoundingBoxes3D(
                targets["boxes"].as_subclass(torch.Tensor).clone(),
                format=targets["boxes"].format,
            ),
            "labels": targets["labels"],
        }
        with pytest.raises(ValueError, match="multiple BoundingBoxes3D"):
            RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)


class TestLabelsGetter:
    def test_default_getter_is_case_insensitive(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {"boxes": targets["boxes"], "Labels": targets["labels"]}
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_targets["Labels"].tolist() == [0, 1]

    def test_custom_labels_getter(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {"boxes": targets["boxes"], "gt_labels": targets["labels"]}
        f = RangeFilter3D(
            point_cloud_range=_RANGE, labels_getter=lambda s: s[1]["gt_labels"]
        )
        _, out_targets = f(inputs, targets)
        assert out_targets["gt_labels"].tolist() == [0, 1]

    def test_custom_getter_returning_tuple_of_tensors(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {
            "boxes": targets["boxes"],
            "labels": targets["labels"],
            "attributes": torch.tensor([10, 11, 12]),
        }
        f = RangeFilter3D(
            point_cloud_range=_RANGE,
            labels_getter=lambda s: (s[1]["labels"], s[1]["attributes"]),
        )
        _, out_targets = f(inputs, targets)
        assert out_targets["boxes"].shape[0] == 2
        assert out_targets["labels"].tolist() == [0, 1]
        assert out_targets["attributes"].tolist() == [10, 11]

    def test_none_getter_filters_boxes_but_not_labels(self) -> None:
        inputs, targets = _make_two_dict_sample()
        f = RangeFilter3D(point_cloud_range=_RANGE, labels_getter=None)
        _, out_targets = f(inputs, targets)
        assert out_targets["boxes"].shape[0] == 2
        assert out_targets["labels"].shape[0] == 3

    def test_non_standard_dict_structure(self) -> None:
        inputs, targets = _make_two_dict_sample()
        sample = {
            "lidar": inputs["points"],
            "gt_boxes": targets["boxes"],
            "labels": targets["labels"],
            "frame_id": 7,
        }
        out = RangeFilter3D(point_cloud_range=_RANGE)(sample)
        assert out["lidar"].shape[0] == 2
        assert out["gt_boxes"].shape[0] == 2
        assert out["labels"].tolist() == [0, 1]
        assert out["frame_id"] == 7

    def test_invalid_labels_getter_raises(self) -> None:
        bad_getter: Any = 123
        with pytest.raises(ValueError, match="labels_getter"):
            RangeFilter3D(point_cloud_range=_RANGE, labels_getter=bad_getter)

    def test_getter_returning_copy_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        f = RangeFilter3D(
            point_cloud_range=_RANGE,
            labels_getter=lambda s: s[1]["labels"].clone(),
        )
        with pytest.raises(ValueError, match="leaves of the sample"):
            f(inputs, targets)

    def test_mismatched_label_length_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {"boxes": targets["boxes"], "labels": torch.tensor([0, 1])}
        with pytest.raises(ValueError, match="labels must be per-box"):
            RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)

    def test_default_getter_non_tensor_labels_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {"boxes": targets["boxes"], "labels": ["a", "b", "c"]}
        with pytest.raises(ValueError, match="not a tensor"):
            RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)

    def test_default_getter_missing_labels_with_boxes_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {"boxes": targets["boxes"]}
        with pytest.raises(ValueError, match="could not find a labels tensor"):
            RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)

    def test_default_getter_missing_labels_without_boxes_ok(self) -> None:
        points = PointCloud3D(torch.tensor([[0.0, 0, 0, 1], [50.0, 0, 0, 1]]))
        sample = {"points": points, "frame_id": 7}
        out = RangeFilter3D(point_cloud_range=_RANGE)(sample)
        assert out["points"].shape[0] == 1
        assert out["frame_id"] == 7

    def test_custom_getter_returning_non_tensor_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        f = RangeFilter3D(
            point_cloud_range=_RANGE, labels_getter=lambda s: "not a tensor"
        )
        with pytest.raises(ValueError, match="tuple/list of"):
            f(inputs, targets)

    def test_custom_getter_returning_list_of_non_tensors_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        f = RangeFilter3D(point_cloud_range=_RANGE, labels_getter=lambda s: [1, 2, 3])
        with pytest.raises(ValueError, match="tuple/list of"):
            f(inputs, targets)

    def test_scalar_label_tensor_raises(self) -> None:
        inputs, targets = _make_two_dict_sample()
        targets = {"boxes": targets["boxes"], "labels": torch.tensor(0)}
        with pytest.raises(ValueError, match="0-dim"):
            RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
