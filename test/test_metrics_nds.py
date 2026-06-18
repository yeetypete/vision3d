"""Tests for NuScenesDetectionScore, cross-checked against nuscenes-devkit.

The reference path builds the official ``DetectionBox``/``EvalBoxes`` data
structures from the same random scenes and drives the devkit's own
``accumulate`` / ``calc_ap`` / ``calc_tp`` / ``DetectionMetrics`` code, then
asserts our metric agrees to floating-point tolerance.
"""

import math
from typing import Any

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
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetrics,
)
from pyquaternion import Quaternion

from vision3d.metrics import NuScenesDetectionScore, Prediction3D, Target3D
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D

# The devkit reference and the matching loop are CPU-only; the dedicated GPU
# regression below exercises the CUDA path explicitly.
pytestmark = pytest.mark.skip_device("cuda")

_TOL = 1e-6

# Attribute id -> nuScenes attribute name; -1 means "no attribute".
_ATTR_BY_ID = {-1: "", **dict(enumerate(ATTRIBUTE_NAMES))}

# Scenes are generated once as numpy (the shared source of truth) and then fed
# to BOTH evaluation paths: ``_to_vision3d`` converts them to torch tensors for
# the metric, ``_to_devkit`` converts the same values to nuscenes-devkit boxes
# for the reference. A "scene" is a per-frame dict of heterogeneous numpy arrays
# (box geometry, velocity, labels, ...); a frame pairs a ``gt`` and ``pred`` scene.
_Scene = dict[str, Any]
_Frame = dict[str, _Scene]


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Return a yaw-only rotation as a ``(w, x, y, z)`` quaternion tuple."""
    q = Quaternion(axis=(0, 0, 1), radians=yaw)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _random_frames(
    rng: np.random.Generator,
    num_frames: int,
    *,
    drop_pred_prob: float = 0.3,
    extra_pred_prob: float = 0.3,
    jitter: float = 0.5,
) -> list[_Frame]:
    """Generate random scenes shared by both evaluation paths.

    Each frame holds ground-truth boxes plus predictions derived from them
    (jittered, some dropped, some spurious) so that matches, misses and false
    positives all occur.

    Returns:
        A list of per-frame dicts with ``gt`` and ``pred`` sub-dicts of plain
        numpy/list scene data.
    """
    n_classes = len(DETECTION_NAMES)
    frames: list[_Frame] = []
    for _ in range(num_frames):
        n_gt = int(rng.integers(0, 6))
        gt: _Scene = {
            "xyz": rng.uniform(-40, 40, size=(n_gt, 3)),
            "lwh": rng.uniform(1.0, 5.0, size=(n_gt, 3)),
            "yaw": rng.uniform(-math.pi, math.pi, size=n_gt),
            "vel": rng.uniform(-5, 5, size=(n_gt, 2)),
            "label": rng.integers(0, n_classes, size=n_gt),
            "attr": rng.integers(-1, len(ATTRIBUTE_NAMES), size=n_gt),
        }
        # Predictions: jitter each GT, randomly drop some.
        p_xyz: list[Any] = []
        p_lwh: list[Any] = []
        p_yaw: list[Any] = []
        p_vel: list[Any] = []
        p_label: list[int] = []
        p_attr: list[int] = []
        p_score: list[float] = []
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

        pred: _Scene = {
            "xyz": np.array(p_xyz).reshape(-1, 3),
            "lwh": np.array(p_lwh).reshape(-1, 3),
            "yaw": np.array(p_yaw).reshape(-1),
            "vel": np.array(p_vel).reshape(-1, 2),
            "label": np.array(p_label, dtype=int).reshape(-1),
            "attr": np.array(p_attr, dtype=int).reshape(-1),
            "score": np.array(p_score).reshape(-1),
        }
        # Cast geometry to float32 once, as a single source of truth: this is
        # the realistic detector-output precision, and feeding identical values
        # to both evaluation paths makes the comparison about the algorithms,
        # not float32-vs-float64 casting.
        for scene in (gt, pred):
            for key in ("xyz", "lwh", "yaw", "vel", "score"):
                if key in scene:
                    scene[key] = scene[key].astype(np.float32)
        frames.append({"gt": gt, "pred": pred})
    return frames


def _to_vision3d(
    frames: list[_Frame],
) -> tuple[list[Prediction3D], list[Target3D]]:
    """Convert scenes to vision3d ``Prediction3D``/``Target3D`` dicts.

    Returns:
        ``(preds, targets)`` lists aligned per frame.
    """
    preds: list[Prediction3D] = []
    targets: list[Target3D] = []
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


def _devkit_box(token: str, scene: _Scene, i: int, *, score: float) -> DetectionBox:
    """Build one devkit ``DetectionBox`` from row ``i`` of ``scene``.

    ``tolist()``/``float(...)`` widen the float32 source values to Python
    float64 without changing them, so the devkit accumulates on the exact same
    numbers the metric does (which also upcasts to float64).

    Returns:
        The constructed ``DetectionBox``.
    """
    return DetectionBox(
        sample_token=token,
        translation=tuple(scene["xyz"][i].tolist()),
        size=tuple(scene["lwh"][i].tolist()),
        rotation=_yaw_quat(float(scene["yaw"][i])),
        velocity=tuple(scene["vel"][i].tolist()),
        detection_name=DETECTION_NAMES[int(scene["label"][i])],
        detection_score=score,
        attribute_name=_ATTR_BY_ID[int(scene["attr"][i])],
    )


def _to_devkit(frames: list[_Frame]) -> tuple[EvalBoxes, EvalBoxes]:
    """Convert scenes to devkit ``EvalBoxes`` of ``DetectionBox``.

    Returns:
        ``(gt_boxes, pred_boxes)`` EvalBoxes.
    """
    gt_eval, pred_eval = EvalBoxes(), EvalBoxes()
    for f_idx, frame in enumerate(frames):
        token = str(f_idx)
        gt, pred = frame["gt"], frame["pred"]
        gt_eval.add_boxes(
            token,
            [_devkit_box(token, gt, i, score=-1.0) for i in range(len(gt["label"]))],
        )
        pred_eval.add_boxes(
            token,
            [
                _devkit_box(token, pred, i, score=float(pred["score"][i]))
                for i in range(len(pred["label"]))
            ],
        )
    return gt_eval, pred_eval


def _devkit_reference(frames: list[_Frame]) -> dict[str, Any]:
    """Run the official devkit evaluation on the scenes.

    Returns:
        A dict with ``nd_score``, ``mean_ap``, ``tp_errors``, ``label_aps``
        (keyed by ``(class_id, dist_th)``) and ``label_tp`` (keyed by
        ``(class_id, metric)``).
    """
    cfg = config_factory("detection_cvpr_2019")
    assert isinstance(cfg, DetectionConfig)
    gt_eval, pred_eval = _to_devkit(frames)
    metrics = DetectionMetrics(cfg)
    label_aps: dict[tuple[int, float], float] = {}
    label_tp: dict[tuple[int, str], float] = {}

    for cls_id, class_name in enumerate(DETECTION_NAMES):
        tp_md = None
        for dist_th in cfg.dist_ths:
            md = accumulate(
                gt_eval, pred_eval, class_name, cfg.dist_fcn_callable, dist_th
            )
            ap = calc_ap(md, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
            label_aps[(cls_id, dist_th)] = ap
            if dist_th == cfg.dist_th_tp:
                tp_md = md
        assert tp_md is not None
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
    # Every class present and predicted exactly (zero error, valid attributes
    # and velocities): each AP is 1 and every TP error is 0, so the AP and all
    # TP scores -- and hence the NDS -- are exactly 1.
    rng = np.random.default_rng(123)
    n_classes = len(DETECTION_NAMES)
    gt: _Scene = {
        "xyz": rng.uniform(-40, 40, size=(n_classes, 3)).astype(np.float32),
        "lwh": rng.uniform(1.0, 5.0, size=(n_classes, 3)).astype(np.float32),
        "yaw": rng.uniform(-math.pi, math.pi, size=n_classes).astype(np.float32),
        "vel": rng.uniform(-5, 5, size=(n_classes, 2)).astype(np.float32),
        "label": np.arange(n_classes),
        "attr": np.zeros(n_classes, dtype=int),
    }
    pred: _Scene = {
        **{k: gt[k].copy() for k in ("xyz", "lwh", "yaw", "vel", "label", "attr")},
        "score": np.full(n_classes, 0.9, dtype=np.float32),
    }
    preds, targets = _to_vision3d([{"gt": gt, "pred": pred}])

    metric = _our_metric()
    metric.update(preds, targets)
    out = metric.compute()

    assert out["mean_ap"] == pytest.approx(1.0, abs=_TOL)
    assert out["nd_score"] == pytest.approx(1.0, abs=_TOL)
    assert all(
        score == pytest.approx(1.0, abs=_TOL) for score in out["tp_scores"].values()
    )


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

    # Corrupt only the predicted velocities, at the scene level so the inputs
    # stay typed ``Prediction3D`` after re-conversion (``preds`` above already
    # materialized its tensors, so it is unaffected).
    for frame in frames:
        frame["pred"]["vel"] = frame["pred"]["vel"] + 10.0
    bad_preds, _ = _to_vision3d(frames)
    bad = NuScenesDetectionScore.from_class_names(list(DETECTION_NAMES))
    bad.update(bad_preds, targets)
    bad_out = bad.compute()

    assert bad_out["tp_errors"]["vel_err"] > good_out["tp_errors"]["vel_err"]
    assert bad_out["nd_score"] < good_out["nd_score"]


def _perfect_car_frame() -> tuple[Prediction3D, Target3D]:
    """A single class-0 box predicted exactly, with geometry fields only.

    Returns:
        A ``(prediction, target)`` pair.
    """
    box = BoundingBoxes3D(
        torch.tensor([box_at(1.0, 2.0, fmt=BoundingBox3DFormat.XYZLWHY)]),
        format=BoundingBox3DFormat.XYZLWHY,
    )
    pred: Prediction3D = {
        "boxes": box,
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
    }
    tgt: Target3D = {"boxes": box, "labels": torch.tensor([0])}
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


def test_tp_threshold_must_be_in_dist_thresholds() -> None:
    with pytest.raises(ValueError, match="tp_threshold"):
        NuScenesDetectionScore(
            class_ids=[0], dist_thresholds=(0.5, 1.0), tp_threshold=2.0
        )


def test_unknown_tp_metric_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tp_metrics"):
        NuScenesDetectionScore(class_ids=[0], tp_metrics=("trans_err", "bogus"))


def test_from_class_names_rejects_derived_kwargs() -> None:
    # ``orientation_periods``/``skip_tp_metrics`` are derived from the class
    # names and are not part of the ``from_class_names`` signature, so passing
    # one is a ``TypeError`` (also flagged statically by the type checker).
    with pytest.raises(TypeError, match="skip_tp_metrics"):
        NuScenesDetectionScore.from_class_names(
            ["car"],
            skip_tp_metrics={0: {"vel_err"}},  # type: ignore[call-arg]
        )


def test_compute_without_update_returns_zero() -> None:
    # With no frames every class is empty, mirroring the devkit's
    # ``no_predictions()`` (AP 0, worst-case TP errors -> TP scores 0), so the
    # aggregate score is exactly zero.
    out = NuScenesDetectionScore.from_class_names(list(DETECTION_NAMES)).compute()
    assert out["nd_score"] == 0.0
    assert out["mean_ap"] == 0.0
    assert all(score == 0.0 for score in out["tp_scores"].values())
