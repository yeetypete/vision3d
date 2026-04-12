"""Tests for RangeFilter3D."""

from typing import Any

import pytest
import torch

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, PointCloud3D
from vision3d.transforms import RangeFilter3D

_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)

_ALL_FORMATS = [
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWHYPR,
]


def _box_at(
    cx: float, cy: float, cz: float = 0.0, *, fmt: BoundingBox3DFormat
) -> list[float]:
    if fmt == BoundingBox3DFormat.XYZXYZ:
        return [cx - 1.0, cy - 1.0, cz - 1.0, cx + 1.0, cy + 1.0, cz + 1.0]
    if fmt == BoundingBox3DFormat.XYZLWH:
        return [cx, cy, cz, 2.0, 2.0, 2.0]
    if fmt == BoundingBox3DFormat.XYZLWHY:
        return [cx, cy, cz, 2.0, 2.0, 2.0, 0.0]
    if fmt == BoundingBox3DFormat.XYZLWHYPR:
        return [cx, cy, cz, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    raise ValueError(fmt)


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
                _box_at(0, 0, fmt=fmt),  # in range
                _box_at(5, 5, fmt=fmt),  # in range
                _box_at(50, 0, fmt=fmt),  # out of range
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
                _box_at(0, 0, fmt=fmt),
                _box_at(50, 0, fmt=fmt),
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
                torch.tensor([_box_at(0, 0, fmt=BoundingBox3DFormat.XYZLWHYPR)]),
                format=BoundingBox3DFormat.XYZLWHYPR,
            ),
            "labels": torch.tensor([0]),
        }
        inputs: dict[str, torch.Tensor[Any]] = {}
        _, out_targets = RangeFilter3D(point_cloud_range=_RANGE)(inputs, targets)
        assert out_targets["boxes"].shape[0] == 1

    def test_no_boxes_key(self) -> None:
        points = PointCloud3D(torch.tensor([[0.0, 0, 0, 1]]))
        inputs = {"points": points}
        targets: dict[str, torch.Tensor[Any]] = {}
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
