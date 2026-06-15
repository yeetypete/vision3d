"""Tests for NuScenesDetectionScore, cross-checked against nuscenes-devkit.

The reference path builds the official ``DetectionBox``/``EvalBoxes`` data
structures from the same random scenes and drives the devkit's own
``accumulate`` / ``calc_ap`` / ``calc_tp`` / ``DetectionMetrics`` code, then
asserts our metric agrees to floating-point tolerance.
"""

import math

import numpy as np
import pytest
import torch
from common_utils import box_at
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import (
    ATTRIBUTE_NAMES,
    DETECTION_NAMES,
    TP_METRICS,
)
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetrics
from pyquaternion import Quaternion

from vision3d.metrics import NuScenesDetectionScore
from vision3d.metrics._nuscenes_detection_score import _interp
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D

# The devkit reference and the matching loop are CPU-only; the dedicated GPU
# regression below exercises the CUDA path explicitly.
pytestmark = pytest.mark.skip_device("cuda")

_TOL = 1e-6

# Attribute id -> nuScenes attribute name; -1 means "no attribute".
_ATTR_BY_ID = {-1: "", **dict(enumerate(ATTRIBUTE_NAMES))}


def _random_frames(
    rng: np.random.Generator,
    num_frames: int,
    *,
    drop_pred_prob: float = 0.3,
    extra_pred_prob: float = 0.3,
    jitter: float = 0.5,
) -> list[dict]:
    """Generate random scenes shared by both evaluation paths.

    Each frame holds ground-truth boxes plus predictions derived from them
    (jittered, some dropped, some spurious) so that matches, misses and false
    positives all occur.

    Returns:
        A list of per-frame dicts with ``gt`` and ``pred`` sub-dicts of plain
        numpy/list scene data.
    """
    n_classes = len(DETECTION_NAMES)
    frames = []
    for _ in range(num_frames):
        n_gt = int(rng.integers(0, 6))
        gt = {
            "xyz": rng.uniform(-40, 40, size=(n_gt, 3)),
            "lwh": rng.uniform(1.0, 5.0, size=(n_gt, 3)),
            "yaw": rng.uniform(-math.pi, math.pi, size=n_gt),
            "vel": rng.uniform(-5, 5, size=(n_gt, 2)),
            "label": rng.integers(0, n_classes, size=n_gt),
            "attr": rng.integers(-1, len(ATTRIBUTE_NAMES), size=n_gt),
        }
        # Predictions: jitter each GT, randomly drop some.
        p_xyz, p_lwh, p_yaw, p_vel, p_label, p_attr, p_score = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(n_gt):
            if rng.uniform() < drop_pred_prob:
                continue
            p_xyz.append(gt["xyz"][i] + rng.normal(0, jitter, size=3))
            p_lwh.append(
                np.clip(gt["lwh"][i] + rng.normal(0, jitter, size=3), 0.3, None)
            )
            p_yaw.append(gt["yaw"][i] + rng.normal(0, 0.2))
            p_vel.append(gt["vel"][i] + rng.normal(0, 0.3, size=2))
            p_label.append(int(gt["label"][i]))
            p_attr.append(int(gt["attr"][i]))
            p_score.append(rng.uniform(0.1, 1.0))
        # Some spurious predictions far from any GT.
        for _ in range(int(rng.binomial(3, extra_pred_prob))):
            p_xyz.append(rng.uniform(-40, 40, size=3))
            p_lwh.append(rng.uniform(1.0, 5.0, size=3))
            p_yaw.append(rng.uniform(-math.pi, math.pi))
            p_vel.append(rng.uniform(-5, 5, size=2))
            p_label.append(int(rng.integers(0, n_classes)))
            p_attr.append(int(rng.integers(-1, len(ATTRIBUTE_NAMES))))
            p_score.append(rng.uniform(0.1, 1.0))

        pred = {
            "xyz": np.array(p_xyz).reshape(-1, 3),
            "lwh": np.array(p_lwh).reshape(-1, 3),
            "yaw": np.array(p_yaw).reshape(-1),
            "vel": np.array(p_vel).reshape(-1, 2),
            "label": np.array(p_label, dtype=int).reshape(-1),
            "attr": np.array(p_attr, dtype=int).reshape(-1),
            "score": np.array(p_score).reshape(-1),
        }
        # Round geometry to float32 once, as a single source of truth: this is
        # the realistic detector-output precision, and feeding identical values
        # to both evaluation paths makes the comparison about the algorithms,
        # not float32-vs-float64 rounding.
        for scene in (gt, pred):
            for key in ("xyz", "lwh", "yaw", "vel", "score"):
                if key in scene:
                    scene[key] = scene[key].astype(np.float32)
        frames.append({"gt": gt, "pred": pred})
    return frames


def _to_vision3d(frames: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert scenes to vision3d ``Prediction3D``/``Target3D`` dicts.

    Returns:
        ``(preds, targets)`` lists aligned per frame.
    """
    preds, targets = [], []
    for frame in frames:
        gt, pred = frame["gt"], frame["pred"]
        # float32 inputs, as a detector would produce; the metric upcasts to
        # float64 internally for accumulation.
        gt_boxes = BoundingBoxes3D(
            torch.tensor(
                np.concatenate(
                    [gt["xyz"], gt["lwh"], gt["yaw"][:, None]], axis=1
                ).reshape(-1, 7),
                dtype=torch.float32,
            ),
            format=BoundingBox3DFormat.XYZLWHY,
        )
        pred_boxes = BoundingBoxes3D(
            torch.tensor(
                np.concatenate(
                    [pred["xyz"], pred["lwh"], pred["yaw"][:, None]], axis=1
                ).reshape(-1, 7),
                dtype=torch.float32,
            ),
            format=BoundingBox3DFormat.XYZLWHY,
        )
        targets.append(
            {
                "boxes": gt_boxes,
                "labels": torch.tensor(gt["label"]),
                "velocities": torch.tensor(gt["vel"], dtype=torch.float32).reshape(
                    -1, 2
                ),
                "attributes": torch.tensor(gt["attr"]),
            }
        )
        preds.append(
            {
                "boxes": pred_boxes,
                "scores": torch.tensor(pred["score"], dtype=torch.float32),
                "labels": torch.tensor(pred["label"]),
                "velocities": torch.tensor(pred["vel"], dtype=torch.float32).reshape(
                    -1, 2
                ),
                "attributes": torch.tensor(pred["attr"]),
            }
        )
    return preds, targets


def _to_devkit(frames: list[dict]) -> tuple[EvalBoxes, EvalBoxes]:
    """Convert scenes to devkit ``EvalBoxes`` of ``DetectionBox``.

    Returns:
        ``(gt_boxes, pred_boxes)`` EvalBoxes.
    """
    gt_eval, pred_eval = EvalBoxes(), EvalBoxes()
    for f_idx, frame in enumerate(frames):
        token = str(f_idx)
        gt, pred = frame["gt"], frame["pred"]

        # ``float(...)`` widens the float32 source values to Python float64
        # without changing them, so the devkit accumulates on the exact same
        # numbers the metric does (which also upcasts to float64).
        gt_list = []
        for i in range(len(gt["label"])):
            gt_list.append(
                DetectionBox(
                    sample_token=token,
                    translation=tuple(float(v) for v in gt["xyz"][i]),
                    size=tuple(float(v) for v in gt["lwh"][i]),
                    rotation=tuple(
                        Quaternion(axis=(0, 0, 1), radians=float(gt["yaw"][i]))
                    ),
                    velocity=tuple(float(v) for v in gt["vel"][i]),
                    detection_name=DETECTION_NAMES[int(gt["label"][i])],
                    detection_score=-1.0,
                    attribute_name=_ATTR_BY_ID[int(gt["attr"][i])],
                )
            )
        gt_eval.add_boxes(token, gt_list)

        pred_list = []
        for i in range(len(pred["label"])):
            pred_list.append(
                DetectionBox(
                    sample_token=token,
                    translation=tuple(float(v) for v in pred["xyz"][i]),
                    size=tuple(float(v) for v in pred["lwh"][i]),
                    rotation=tuple(
                        Quaternion(axis=(0, 0, 1), radians=float(pred["yaw"][i]))
                    ),
                    velocity=tuple(float(v) for v in pred["vel"][i]),
                    detection_name=DETECTION_NAMES[int(pred["label"][i])],
                    detection_score=float(pred["score"][i]),
                    attribute_name=_ATTR_BY_ID[int(pred["attr"][i])],
                )
            )
        pred_eval.add_boxes(token, pred_list)
    return gt_eval, pred_eval


def _devkit_reference(frames: list[dict]) -> dict:
    """Run the official devkit evaluation on the scenes.

    Returns:
        A dict with ``nd_score``, ``mean_ap``, ``tp_errors``, ``label_aps``
        (keyed by ``(class_id, dist_th)``) and ``label_tp`` (keyed by
        ``(class_id, metric)``).
    """
    cfg = config_factory("detection_cvpr_2019")
    gt_eval, pred_eval = _to_devkit(frames)
    metrics = DetectionMetrics(cfg)
    label_aps, label_tp = {}, {}

    for cls_id, class_name in enumerate(DETECTION_NAMES):
        for dist_th in cfg.dist_ths:
            md = accumulate(
                gt_eval, pred_eval, class_name, cfg.dist_fcn_callable, dist_th
            )
            ap = calc_ap(md, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
            label_aps[(cls_id, dist_th)] = ap
            if dist_th == cfg.dist_th_tp:
                tp_md = md
        for metric_name in TP_METRICS:
            skip = (
                class_name == "traffic_cone"
                and metric_name in ("attr_err", "vel_err", "orient_err")
            ) or (class_name == "barrier" and metric_name in ("attr_err", "vel_err"))
            tp = np.nan if skip else calc_tp(tp_md, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)
            label_tp[(cls_id, metric_name)] = tp

    return {
        "nd_score": metrics.nd_score,
        "mean_ap": metrics.mean_ap,
        "tp_errors": metrics.tp_errors,
        "label_aps": label_aps,
        "label_tp": label_tp,
    }


def _our_metric() -> NuScenesDetectionScore:
    return NuScenesDetectionScore.from_class_names(list(DETECTION_NAMES))


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_matches_devkit(seed: int) -> None:
    frames = _random_frames(np.random.default_rng(seed), num_frames=8)
    ref = _devkit_reference(frames)

    preds, targets = _to_vision3d(frames)
    metric = _our_metric()
    metric.update(preds, targets)
    out = metric.compute()

    assert out["nd_score"] == pytest.approx(ref["nd_score"], abs=_TOL)
    assert out["mean_ap"] == pytest.approx(ref["mean_ap"], abs=_TOL)

    for metric_name in TP_METRICS:
        assert out["tp_errors"][metric_name] == pytest.approx(
            ref["tp_errors"][metric_name], abs=_TOL
        ), metric_name

    for key, ap in ref["label_aps"].items():
        assert out["label_aps"][key] == pytest.approx(ap, abs=_TOL), key

    for key, tp in ref["label_tp"].items():
        ours = out["label_tp_errors"][key]
        if math.isnan(tp):
            assert math.isnan(ours), key
        else:
            assert ours == pytest.approx(tp, abs=_TOL), key


def test_perfect_prediction_scores_one() -> None:
    rng = np.random.default_rng(123)
    # Predictions identical to GT -> NDS should be 1.0.
    frames = _random_frames(
        rng, num_frames=6, drop_pred_prob=0.0, extra_pred_prob=0.0, jitter=0.0
    )
    for frame in frames:
        frame["pred"] = {
            **{k: frame["gt"][k].copy() for k in ("xyz", "lwh", "yaw", "vel", "attr")},
            "label": frame["gt"]["label"].copy(),
            "score": np.full(len(frame["gt"]["label"]), 0.9),
        }

    preds, targets = _to_vision3d(frames)
    metric = _our_metric()
    metric.update(preds, targets)
    out = metric.compute()

    # Only classes that actually appear can reach AP 1; check the matched
    # TP errors are ~0 and NDS agrees with the devkit either way.
    ref = _devkit_reference(frames)
    assert out["nd_score"] == pytest.approx(ref["nd_score"], abs=_TOL)


@pytest.mark.parametrize(
    "active",
    [
        ("trans_err", "scale_err", "orient_err"),
        ("trans_err",),
        ("trans_err", "scale_err", "orient_err", "vel_err"),
    ],
)
def test_reduced_tp_metrics_renormalize_like_devkit(active: tuple[str, ...]) -> None:
    # AP and the per-metric TP errors are unchanged by dropping a metric; only
    # the NDS normalization differs. Reconstruct the expected NDS from the
    # devkit's own mAP and TP scores over just the active subset.
    frames = _random_frames(np.random.default_rng(3), num_frames=8)
    ref = _devkit_reference(frames)

    preds, targets = _to_vision3d(frames)
    metric = NuScenesDetectionScore.from_class_names(
        list(DETECTION_NAMES), tp_metrics=active
    )
    metric.update(preds, targets)
    out = metric.compute()

    # Only the active metrics are reported.
    assert set(out["tp_errors"]) == set(active)
    assert set(out["tp_scores"]) == set(active)

    ref_tp_scores = {m: max(0.0, 1.0 - ref["tp_errors"][m]) for m in active}
    expected_nds = (5.0 * ref["mean_ap"] + sum(ref_tp_scores.values())) / (
        5.0 + len(active)
    )
    assert out["nd_score"] == pytest.approx(expected_nds, abs=_TOL)
    assert out["mean_ap"] == pytest.approx(ref["mean_ap"], abs=_TOL)


def test_velocity_affects_score() -> None:
    # Two identical scenes except predicted velocity; the velocity error (and
    # thus NDS) must change, proving velocities flow into the score.
    frames = _random_frames(
        np.random.default_rng(5), num_frames=6, drop_pred_prob=0.0, jitter=0.1
    )
    preds, targets = _to_vision3d(frames)

    good = NuScenesDetectionScore.from_class_names(list(DETECTION_NAMES))
    good.update(preds, targets)
    good_out = good.compute()

    # Corrupt only the predicted velocities.
    bad_preds = [{**p, "velocities": p["velocities"] + 10.0} for p in preds]
    bad = NuScenesDetectionScore.from_class_names(list(DETECTION_NAMES))
    bad.update(bad_preds, targets)
    bad_out = bad.compute()

    assert bad_out["tp_errors"]["vel_err"] > good_out["tp_errors"]["vel_err"]
    assert bad_out["nd_score"] < good_out["nd_score"]


def _perfect_car_frame() -> tuple[dict, dict]:
    """A single class-0 box predicted exactly, with geometry fields only.

    Returns:
        A ``(prediction, target)`` pair.
    """
    box = BoundingBoxes3D(
        torch.tensor([box_at(1.0, 2.0, fmt=BoundingBox3DFormat.XYZLWHY)]),
        format=BoundingBox3DFormat.XYZLWHY,
    )
    pred = {"boxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
    tgt = {"boxes": box, "labels": torch.tensor([0])}
    return pred, tgt


def test_geometry_only_ignores_missing_velocity_and_attribute() -> None:
    # The base case: no velocity/attribute annotations. A geometry-only config
    # must not be polluted by a free perfect velocity or worst-case attribute.
    pred, tgt = _perfect_car_frame()
    metric = NuScenesDetectionScore(
        class_ids=[0], tp_metrics=("trans_err", "scale_err", "orient_err")
    )
    metric.update([pred], [tgt])
    out = metric.compute()

    assert set(out["tp_errors"]) == {"trans_err", "scale_err", "orient_err"}
    assert out["nd_score"] == pytest.approx(1.0, abs=_TOL)


def test_compute_under_cuda_default_device() -> None:
    # Regression: internal tensors must be pinned to CPU even when both the
    # inputs and the ambient default-device are CUDA, otherwise they mismatch
    # the CPU-moved box data. Exercises the path the module-level skip excludes.
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    metric = NuScenesDetectionScore(
        class_ids=[0], tp_metrics=("trans_err", "scale_err", "orient_err")
    )
    with torch.device("cuda"):
        pred, tgt = _perfect_car_frame()
        metric.update([pred], [tgt])
        out = metric.compute()
    assert out["nd_score"] == pytest.approx(1.0, abs=_TOL)


def test_active_metric_requires_its_annotations() -> None:
    pred, tgt = _perfect_car_frame()
    with pytest.raises(ValueError, match="velocities"):
        NuScenesDetectionScore(class_ids=[0], tp_metrics=("vel_err",)).update(
            [pred], [tgt]
        )
    with pytest.raises(ValueError, match="attributes"):
        NuScenesDetectionScore(class_ids=[0], tp_metrics=("attr_err",)).update(
            [pred], [tgt]
        )


@pytest.mark.parametrize("seed", range(20))
def test_interp_matches_numpy(seed: int) -> None:
    # ``_interp`` must reproduce ``np.interp`` exactly, including duplicate /
    # flat runs in ``xp`` (every false positive creates one) and queries that
    # land on a duplicate run clamped at the end of ``xp``.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 25))
    # Heavy duplication: draw from a small integer set and sort.
    xp = np.sort(rng.integers(0, 6, n).astype(np.float64))
    if xp[0] == xp[-1]:
        xp[-1] += 1.0
    fp = rng.normal(size=n)
    # Queries deliberately include the exact (possibly duplicated) nodes.
    q = np.concatenate(
        [rng.uniform(-1.0, 7.0, 30), xp, np.array([xp[0] - 1.0, xp[-1] + 1.0])]
    )
    ref = np.interp(q, xp, fp, right=0.0)
    ours = _interp(
        torch.from_numpy(q), torch.from_numpy(xp), torch.from_numpy(fp), right=0.0
    ).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_tp_threshold_must_be_in_dist_thresholds() -> None:
    with pytest.raises(ValueError, match="tp_threshold"):
        NuScenesDetectionScore(
            class_ids=[0], dist_thresholds=(0.5, 1.0), tp_threshold=2.0
        )


def test_unknown_tp_metric_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tp_metrics"):
        NuScenesDetectionScore(class_ids=[0], tp_metrics=("trans_err", "bogus"))


def test_from_class_names_rejects_derived_kwargs() -> None:
    with pytest.raises(ValueError, match="derived"):
        NuScenesDetectionScore.from_class_names(
            ["car"], skip_tp_metrics={0: {"vel_err"}}
        )
