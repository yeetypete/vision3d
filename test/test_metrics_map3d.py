"""Tests for MeanAveragePrecision3D."""

import pytest
import torch

from vision3d.metrics import (
    APInterpolation,
    MeanAveragePrecision3D,
    Prediction3D,
    Target3D,
)
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D

CAR = 0
PED = 1

_AP_TOL = 1e-5

_ALL_FORMATS = [
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWHYPR,
]

_NUM_COLS = {
    BoundingBox3DFormat.XYZXYZ: 6,
    BoundingBox3DFormat.XYZLWH: 6,
    BoundingBox3DFormat.XYZLWHY: 7,
    BoundingBox3DFormat.XYZLWHYPR: 9,
}


def _box_at(
    cx: float,
    cy: float,
    cz: float = 0.0,
    *,
    fmt: BoundingBox3DFormat,
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


def _boxes(rows: list[list[float]], fmt: BoundingBox3DFormat) -> BoundingBoxes3D:
    return BoundingBoxes3D(torch.tensor(rows, dtype=torch.float32), format=fmt)


def _empty_boxes(fmt: BoundingBox3DFormat) -> BoundingBoxes3D:
    return BoundingBoxes3D(torch.zeros((0, _NUM_COLS[fmt])), format=fmt)


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestPerfectPrediction:
    def test_single_frame_perfect_ap_is_one(self, fmt: BoundingBox3DFormat) -> None:
        gt = _boxes(
            [_box_at(0, 0, fmt=fmt), _box_at(10, 0, fmt=fmt), _box_at(-10, 0, fmt=fmt)],
            fmt,
        )
        pred = {
            "boxes": gt,
            "scores": torch.tensor([0.9, 0.8, 0.95]),
            "labels": torch.tensor([CAR, CAR, CAR]),
        }
        tgt = {"boxes": gt, "labels": torch.tensor([CAR, CAR, CAR])}

        m = MeanAveragePrecision3D(class_ids=[CAR])
        m.update([pred], [tgt])
        r = m.compute()
        assert r["mAP"] == pytest.approx(1.0, abs=_AP_TOL)
        assert r["mAP_per_class"][CAR] == pytest.approx(1.0, abs=_AP_TOL)

    def test_perfect_prediction_every_interpolation(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt), _box_at(10, 0, fmt=fmt)], fmt)
        pred = {
            "boxes": gt,
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([CAR, CAR]),
        }
        tgt = {"boxes": gt, "labels": torch.tensor([CAR, CAR])}

        for interp in APInterpolation:
            m = MeanAveragePrecision3D(class_ids=[CAR], ap_interpolation=interp)
            m.update([pred], [tgt])
            assert m.compute()["mAP"] == pytest.approx(1.0, abs=_AP_TOL), interp


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestNoMatches:
    def test_all_predictions_far_from_gt(self, fmt: BoundingBox3DFormat) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        pred = _boxes([_box_at(100, 100, fmt=fmt)], fmt)
        m = MeanAveragePrecision3D(class_ids=[CAR])
        m.update(
            [
                {
                    "boxes": pred,
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([CAR]),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR])}],
        )
        assert m.compute()["mAP"] == pytest.approx(0.0, abs=_AP_TOL)

    def test_empty_predictions_nonempty_gt(self, fmt: BoundingBox3DFormat) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        empty = _empty_boxes(fmt)
        m = MeanAveragePrecision3D(class_ids=[CAR])
        m.update(
            [
                {
                    "boxes": empty,
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR])}],
        )
        assert m.compute()["mAP"] == pytest.approx(0.0, abs=_AP_TOL)

    def test_empty_gt_nonempty_predictions(self, fmt: BoundingBox3DFormat) -> None:
        pred = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        empty = _empty_boxes(fmt)
        m = MeanAveragePrecision3D(class_ids=[CAR])
        m.update(
            [
                {
                    "boxes": pred,
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([CAR]),
                }
            ],
            [
                {
                    "boxes": empty,
                    "labels": torch.zeros(0, dtype=torch.long),
                }
            ],
        )
        # No GTs for CAR in any frame -> AP undefined -> -1 sentinel.
        assert m.compute()["mAP"] == -1.0


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestPartialMatches:
    def test_one_tp_one_fp(self, fmt: BoundingBox3DFormat) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        pred = _boxes([_box_at(0, 0, fmt=fmt), _box_at(100, 100, fmt=fmt)], fmt)
        m = MeanAveragePrecision3D(
            class_ids=[CAR],
            iou_thresholds=(0.5,),
            ap_interpolation=APInterpolation.ALL_POINTS,
        )
        m.update(
            [
                {
                    "boxes": pred,
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([CAR, CAR]),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR])}],
        )
        # Precision curve: after high-score TP -> (1/1=1.0, recall=1.0).
        # After low-score FP -> still recall=1.0, precision=0.5 < 1.0,
        # right-envelope pulls precision at recall=1.0 back up to 1.0.
        assert m.compute()["mAP"] == pytest.approx(1.0, abs=_AP_TOL)

    def test_one_tp_and_one_missed_gt_ap_half(self, fmt: BoundingBox3DFormat) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt), _box_at(10, 0, fmt=fmt)], fmt)
        pred = _boxes([_box_at(0, 0, fmt=fmt)], fmt)  # only hits first GT
        m = MeanAveragePrecision3D(
            class_ids=[CAR],
            iou_thresholds=(0.5,),
            ap_interpolation=APInterpolation.ALL_POINTS,
        )
        m.update(
            [
                {
                    "boxes": pred,
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([CAR]),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR, CAR])}],
        )
        # Max achievable recall = 0.5. Under VOC07 all-points, the area
        # is precision=1.0 over recall [0, 0.5] -> 0.5.
        assert m.compute()["mAP"] == pytest.approx(0.5, abs=_AP_TOL)


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestMultiClass:
    def test_per_class_breakdown(self, fmt: BoundingBox3DFormat) -> None:
        boxes = _boxes([_box_at(0, 0, fmt=fmt), _box_at(5, 0, fmt=fmt)], fmt)
        pred = {
            "boxes": boxes,
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([CAR, PED]),
        }
        tgt = {"boxes": boxes, "labels": torch.tensor([CAR, PED])}

        m = MeanAveragePrecision3D(class_ids=[CAR, PED])
        m.update([pred], [tgt])
        r = m.compute()
        assert r["mAP_per_class"][CAR] == pytest.approx(1.0, abs=_AP_TOL)
        assert r["mAP_per_class"][PED] == pytest.approx(1.0, abs=_AP_TOL)


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestIoUThresholdSensitivity:
    def test_loose_threshold_matches_strict_does_not(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        # Shift the prediction by half a box side. With box size 2x2x2
        # and shift 1.0, the overlap box is 1x2x2 -> vol=4 vs union=12
        # -> IoU ~ 0.33. Passes 0.3 threshold, fails 0.5.
        pred = _boxes([_box_at(1.0, 0.0, fmt=fmt)], fmt)
        m = MeanAveragePrecision3D(
            class_ids=[CAR],
            iou_thresholds=(0.3, 0.5),
            ap_interpolation=APInterpolation.ALL_POINTS,
        )
        m.update(
            [
                {
                    "boxes": pred,
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([CAR]),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR])}],
        )
        r = m.compute()
        assert r["AP_per_iou"][0.3] == pytest.approx(1.0, abs=_AP_TOL)
        assert r["AP_per_iou"][0.5] == pytest.approx(0.0, abs=_AP_TOL)


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestRangeBins:
    def test_range_bin_breakdown(self, fmt: BoundingBox3DFormat) -> None:
        gt = _boxes([_box_at(5, 0, fmt=fmt), _box_at(40, 0, fmt=fmt)], fmt)
        pred = gt
        m = MeanAveragePrecision3D(
            class_ids=[CAR],
            range_bins=((0.0, 30.0), (30.0, 100.0)),
        )
        m.update(
            [
                {
                    "boxes": pred,
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([CAR, CAR]),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR, CAR])}],
        )
        r = m.compute()
        assert "AP_per_range" in r
        per_range = r["AP_per_range"]
        assert (0.0, 30.0) in per_range
        assert (30.0, 100.0) in per_range
        assert per_range[(0.0, 30.0)] == pytest.approx(1.0, abs=_AP_TOL)
        assert per_range[(30.0, 100.0)] == pytest.approx(1.0, abs=_AP_TOL)

    def test_range_bin_isolates_misses(self, fmt: BoundingBox3DFormat) -> None:
        # Only a near-range GT exists; far bin has no GTs -> AP undefined.
        gt = _boxes([_box_at(5, 0, fmt=fmt)], fmt)
        m = MeanAveragePrecision3D(
            class_ids=[CAR],
            range_bins=((0.0, 30.0), (30.0, 100.0)),
        )
        m.update(
            [
                {
                    "boxes": gt,
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([CAR]),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR])}],
        )
        r = m.compute()
        assert "AP_per_range" in r
        per_range = r["AP_per_range"]
        assert per_range[(0.0, 30.0)] == pytest.approx(1.0, abs=_AP_TOL)
        assert per_range[(30.0, 100.0)] == -1.0


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestMultiFrameAccumulation:
    def test_state_accumulates_across_update_calls(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        pred = {
            "boxes": gt,
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([CAR]),
        }
        tgt = {"boxes": gt, "labels": torch.tensor([CAR])}

        m = MeanAveragePrecision3D(class_ids=[CAR])
        for _ in range(3):
            m.update([pred], [tgt])
        assert m.compute()["mAP"] == pytest.approx(1.0, abs=_AP_TOL)

    def test_list_of_frames_matches_sequential_updates(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        pred_dict = {
            "boxes": gt,
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([CAR]),
        }
        tgt_dict = {"boxes": gt, "labels": torch.tensor([CAR])}

        m_batched = MeanAveragePrecision3D(class_ids=[CAR])
        m_batched.update(
            [pred_dict, pred_dict, pred_dict], [tgt_dict, tgt_dict, tgt_dict]
        )

        m_sequential = MeanAveragePrecision3D(class_ids=[CAR])
        for _ in range(3):
            m_sequential.update([pred_dict], [tgt_dict])

        assert m_batched.compute()["mAP"] == pytest.approx(
            m_sequential.compute()["mAP"], abs=1e-8
        )


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestReset:
    def test_reset_clears_state(self, fmt: BoundingBox3DFormat) -> None:
        gt = _boxes([_box_at(0, 0, fmt=fmt)], fmt)
        m = MeanAveragePrecision3D(class_ids=[CAR])
        m.update(
            [
                {
                    "boxes": gt,
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([CAR]),
                }
            ],
            [{"boxes": gt, "labels": torch.tensor([CAR])}],
        )
        m.reset()
        assert m.compute()["mAP"] == -1.0


class TestInputValidation:
    def test_length_mismatch_raises(self) -> None:
        m = MeanAveragePrecision3D(class_ids=[CAR])
        fmt = BoundingBox3DFormat.XYZLWHY
        pred: Prediction3D = {
            "boxes": _empty_boxes(fmt),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        target: Target3D = {
            "boxes": _empty_boxes(fmt),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        with pytest.raises(ValueError, match="same length"):
            m.update([pred], [target, target])

    def test_empty_class_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="class_ids"):
            MeanAveragePrecision3D(class_ids=[])

    def test_empty_iou_thresholds_raises(self) -> None:
        with pytest.raises(ValueError, match="iou_thresholds"):
            MeanAveragePrecision3D(class_ids=[CAR], iou_thresholds=())
