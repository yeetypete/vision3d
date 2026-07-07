"""Tests for the ObjectPointsSample transform."""

from typing import Any

import pytest
import torch
from common_utils import check_transform, make_fusion_sample

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, PointCloud3D
from vision3d.transforms import ObjectPointsSample

FMT = BoundingBox3DFormat.XYZLWHY

# Point layout in the controlled sample below.
_N_OBJ0 = 8
_N_OBJ1 = 6
_N_BG = 5


def _controlled_sample() -> dict[str, Any]:
    """Build a sample with a known number of points inside each of two boxes.

    Box 0 (label 0) sits at x=10 with ``_N_OBJ0`` interior points; box 1
    (label 1) at x=20 with ``_N_OBJ1``; ``_N_BG`` background points sit far
    away. Each point carries a unique id in its feature column so survivors
    can be identified after thinning.

    Returns:
        ``{"points", "boxes", "labels"}`` sample dict.
    """
    boxes = BoundingBoxes3D(
        torch.tensor(
            [
                [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [20.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]
        ),
        format=FMT,
    )
    # Interior points within +/-0.4 of each center (box half-extent is 1.0).
    xyz0 = torch.tensor([10.0, 0.0, 0.0]) + (torch.rand(_N_OBJ0, 3) - 0.5) * 0.8
    xyz1 = torch.tensor([20.0, 0.0, 0.0]) + (torch.rand(_N_OBJ1, 3) - 0.5) * 0.8
    xyz_bg = torch.tensor([100.0, 100.0, 100.0]) + torch.rand(_N_BG, 3)
    xyz = torch.cat([xyz0, xyz1, xyz_bg])
    ids = torch.arange(xyz.shape[0], dtype=torch.float32).unsqueeze(1)
    points = PointCloud3D(torch.cat([xyz, ids], dim=1))
    return {"points": points, "boxes": boxes, "labels": torch.tensor([0, 1])}


def _bg_ids() -> set[int]:
    return set(range(_N_OBJ0 + _N_OBJ1, _N_OBJ0 + _N_OBJ1 + _N_BG))


def _surviving_ids(points: PointCloud3D) -> set[int]:
    return {int(v) for v in points[:, -1].tolist()}


class TestObjectPointsSampleValidation:
    def test_requires_a_mode(self) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            ObjectPointsSample()

    def test_modes_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            ObjectPointsSample(keep=5, keep_ratio=0.5)

    def test_negative_keep(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ObjectPointsSample(keep=-1)

    def test_keep_min_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="min must not exceed max"):
            ObjectPointsSample(keep=(5, 2))

    def test_keep_ratio_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            ObjectPointsSample(keep_ratio=1.5)

    def test_keep_ratio_min_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="min must not exceed max"):
            ObjectPointsSample(keep_ratio=(0.8, 0.2))

    def test_bad_p_object(self) -> None:
        with pytest.raises(ValueError, match="p_object"):
            ObjectPointsSample(keep=5, p_object=1.5)

    def test_bad_p(self) -> None:
        with pytest.raises(ValueError, match="`p`"):
            ObjectPointsSample(keep=5, p=2.0)

    def test_bool_keep_rejected(self) -> None:
        with pytest.raises(TypeError, match="int"):
            ObjectPointsSample(keep=True)

    def test_bad_labels_getter(self) -> None:
        with pytest.raises(ValueError, match="labels_getter"):
            ObjectPointsSample(keep=5, labels=[0], labels_getter="bogus")

    def test_labels_getter_none_with_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="labels_getter"):
            ObjectPointsSample(keep=5, labels=[0], labels_getter=None)


class TestObjectPointsSampleConvention:
    def test_transform(self) -> None:
        check_transform(ObjectPointsSample(keep=5, p=1.0), make_fusion_sample())

    def test_p_zero_is_identity(self) -> None:
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=0, p=0.0)(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_boxes_and_labels_untouched(self) -> None:
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=0, p=1.0)(sample)
        assert out["boxes"] is sample["boxes"]
        assert out["labels"] is sample["labels"]


class TestObjectPointsSampleBehavior:
    def test_keep_zero_removes_all_object_points(self) -> None:
        torch.manual_seed(0)
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=0, p=1.0, p_object=1.0)(sample)
        assert _surviving_ids(out["points"]) == _bg_ids()

    def test_fixed_keep_caps_each_object(self) -> None:
        torch.manual_seed(0)
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=3, p=1.0, p_object=1.0)(sample)
        # 3 kept per object + all background.
        assert out["points"].shape[0] == 3 + 3 + _N_BG
        assert _bg_ids() <= _surviving_ids(out["points"])

    def test_keep_larger_than_count_is_noop(self) -> None:
        torch.manual_seed(0)
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=1000, p=1.0, p_object=1.0)(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_keep_ratio(self) -> None:
        torch.manual_seed(0)
        sample = _controlled_sample()
        out = ObjectPointsSample(keep_ratio=0.5, p=1.0, p_object=1.0)(sample)
        # round(8*0.5)=4, round(6*0.5)=3, + background.
        assert out["points"].shape[0] == 4 + 3 + _N_BG

    def test_survivors_keep_original_order(self) -> None:
        torch.manual_seed(0)
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=2, p=1.0, p_object=1.0)(sample)
        ids = out["points"][:, -1].tolist()
        assert ids == sorted(ids)

    def test_label_filter_only_thins_selected_class(self) -> None:
        torch.manual_seed(0)
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=0, labels=[0], p=1.0, p_object=1.0)(sample)
        survivors = _surviving_ids(out["points"])
        obj1_ids = set(range(_N_OBJ0, _N_OBJ0 + _N_OBJ1))
        # Class 1 fully retained, class 0 fully removed.
        assert obj1_ids <= survivors
        assert survivors == obj1_ids | _bg_ids()

    def test_label_filter_without_labels_raises(self) -> None:
        sample = _controlled_sample()
        del sample["labels"]
        with pytest.raises(TypeError, match="label tensor"):
            ObjectPointsSample(keep=0, labels=[0], p=1.0)(sample)

    def test_labels_getter_callable(self) -> None:
        torch.manual_seed(0)
        base = _controlled_sample()
        # Labels stored under a non-standard key; a callable getter finds them.
        sample = {
            "points": base["points"],
            "boxes": base["boxes"],
            "gt": {"class_ids": base["labels"]},
        }
        out = ObjectPointsSample(
            keep=0,
            labels=[0],
            p=1.0,
            p_object=1.0,
            labels_getter=lambda s: s["gt"]["class_ids"],
        )(sample)
        obj1_ids = set(range(_N_OBJ0, _N_OBJ0 + _N_OBJ1))
        assert _surviving_ids(out["points"]) == obj1_ids | _bg_ids()

    def test_overlapping_boxes_thinned_with_eligible_box(self) -> None:
        # A point inside both a filtered-out box (lower index) and a targeted
        # box must still be thinned with the targeted box, not escape via the
        # first-box-wins tie-break.
        torch.manual_seed(0)
        boxes = BoundingBoxes3D(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0],  # box0 (label 0)
                    [1.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0],  # box1 (label 1), overlaps
                ]
            ),
            format=FMT,
        )
        # Ten points in the shared overlap region (inside both boxes).
        xs = torch.linspace(-0.9, 1.9, 10)
        xyz = torch.stack([xs, torch.zeros(10), torch.zeros(10)], dim=1)
        ids = torch.arange(10, dtype=torch.float32).unsqueeze(1)
        points = PointCloud3D(torch.cat([xyz, ids], dim=1))
        sample = {"points": points, "boxes": boxes, "labels": torch.tensor([0, 1])}
        out = ObjectPointsSample(keep=2, labels=[1], p=1.0, p_object=1.0)(sample)
        # Targeting label 1 down to 2 points removes 8 of the overlap points.
        assert out["points"].shape[0] == 2

    def test_label_filter_without_labels_raises_before_p_gate(self) -> None:
        # p=0 would skip the transform; the check must still fire.
        sample = _controlled_sample()
        del sample["labels"]
        with pytest.raises(TypeError, match="label tensor"):
            ObjectPointsSample(keep=0, labels=[0], p=0.0)(sample)

    def test_labels_resolved_by_key_over_decoy(self) -> None:
        torch.manual_seed(0)
        base = _controlled_sample()
        # Decoy integer per-box tensor precedes "labels" in the dict.
        sample = {
            "points": base["points"],
            "boxes": base["boxes"],
            "num_lidar_pts": torch.tensor([7, 7]),
            "labels": base["labels"],
        }
        out = ObjectPointsSample(keep=0, labels=[0], p=1.0, p_object=1.0)(sample)
        survivors = _surviving_ids(out["points"])
        obj1_ids = set(range(_N_OBJ0, _N_OBJ0 + _N_OBJ1))
        # Class 0 removed via "labels"; the decoy would have thinned nothing.
        assert survivors == obj1_ids | _bg_ids()

    def test_float_per_box_tensor_not_treated_as_labels(self) -> None:
        sample = _controlled_sample()
        sample["labels"] = torch.tensor([0.0, 1.0])
        with pytest.raises(TypeError, match="label tensor"):
            ObjectPointsSample(keep=0, labels=[0], p=1.0)(sample)

    def test_p_object_zero_is_noop(self) -> None:
        torch.manual_seed(0)
        sample = _controlled_sample()
        out = ObjectPointsSample(keep=0, p=1.0, p_object=0.0)(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_empty_boxes_passthrough(self) -> None:
        sample = _controlled_sample()
        sample["boxes"] = BoundingBoxes3D(torch.zeros(0, 7), format=FMT)
        sample["labels"] = torch.zeros(0, dtype=torch.long)
        out = ObjectPointsSample(keep=0, p=1.0)(sample)
        assert out["points"] is sample["points"]

    def test_keep_range_samples_per_object_within_bounds(self) -> None:
        torch.manual_seed(0)
        counts0, counts1 = set(), set()
        for _ in range(50):
            out = ObjectPointsSample(keep=(2, 5), p=1.0, p_object=1.0)(
                _controlled_sample()
            )
            ids = _surviving_ids(out["points"])
            counts0.add(len(ids & set(range(_N_OBJ0))))
            counts1.add(len(ids & set(range(_N_OBJ0, _N_OBJ0 + _N_OBJ1))))
        # Every draw within [2, 5]; both objects have >= 5 points.
        assert counts0
        assert counts1
        assert all(2 <= c <= 5 for c in counts0)
        assert all(2 <= c <= 5 for c in counts1)
        # The range is actually sampled, not pinned to a single value.
        assert len(counts0) > 1

    def test_keep_ratio_range_samples_per_object_within_bounds(self) -> None:
        torch.manual_seed(0)
        lo, hi = 0.25, 0.75
        counts0 = set()
        for _ in range(50):
            out = ObjectPointsSample(keep_ratio=(lo, hi), p=1.0, p_object=1.0)(
                _controlled_sample()
            )
            ids = _surviving_ids(out["points"])
            counts0.add(len(ids & set(range(_N_OBJ0))))
        # round(ratio * 8) is monotonic in ratio, so counts lie in the
        # rounded endpoint band, and the range is genuinely sampled.
        assert counts0
        assert all(round(lo * _N_OBJ0) <= c <= round(hi * _N_OBJ0) for c in counts0)
        assert len(counts0) > 1
