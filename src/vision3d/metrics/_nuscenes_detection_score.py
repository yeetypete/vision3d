r"""nuScenes Detection Score (NDS) metric.

This reproduces the official nuScenes detection evaluation
(``nuscenes.eval.detection``): detections are matched to ground truth by
2D (bird's-eye-view) center distance, average precision is integrated over
several distance thresholds, and five true-positive (TP) error metrics --
translation, scale, orientation, velocity and attribute -- are aggregated
into the single nuScenes Detection Score.

The metric is dataset-agnostic: classes are referenced by integer label and
all thresholds, weights and per-class special-cases are configurable. Use
:meth:`NuScenesDetectionScore.from_class_names` to build the metric with the
official nuScenes settings (distance thresholds ``(0.5, 1, 2, 4)`` m, TP
distance ``2`` m, ``barrier`` orientation period of :math:`\pi`, and the
attribute/velocity/orientation skips for ``barrier`` and ``traffic_cone``).
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from torch import Tensor

from vision3d.metrics._types import Prediction3D, Target3D
from vision3d.ops._points_in_boxes_3d import _extract_box_params

if TYPE_CHECKING:
    from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D

# The five true-positive error metrics, in the order nuScenes reports them.
TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")

# Number of interpolated recall steps (0 % .. 100 %), matching the devkit's
# ``DetectionMetricData.nelem``.
_NELEM = 101

# nuScenes classes special-cased in :meth:`from_class_names`: ``barrier``
# orientation is only defined up to 180 degrees, and barrier/traffic_cone
# lack a meaningful velocity/attribute (and, for cones, orientation).
_HALF_PERIOD_CLASSES = ("barrier",)
_SKIP_TP_METRICS = {
    "traffic_cone": frozenset({"attr_err", "vel_err", "orient_err"}),
    "barrier": frozenset({"attr_err", "vel_err"}),
}


class NuScenesDetectionScoreResult(TypedDict):
    """Structured result returned by :meth:`NuScenesDetectionScore.compute`.

    Attributes:
        nd_score: The nuScenes Detection Score, a weighted combination of
            ``mean_ap`` and the TP scores in ``[0, 1]``.
        mean_ap: Mean Average Precision, averaged over distance thresholds
            and classes.
        mean_dist_aps: Per-class AP, averaged over distance thresholds.
            Keyed by class ID.
        label_aps: AP per ``(class_id, distance_threshold)`` pair.
        tp_errors: Mean TP error per active metric name (a subset of
            ``trans_err``, ``scale_err``, ``orient_err``, ``vel_err``,
            ``attr_err``), averaged over classes (ignoring skipped classes).
        tp_scores: Per-metric score ``max(0, 1 - error)``, for the active
            metrics only.
        label_tp_errors: TP error per ``(class_id, metric_name)`` pair over
            the active metrics; ``nan`` for skipped class/metric
            combinations.
    """

    nd_score: float
    mean_ap: float
    mean_dist_aps: dict[int, float]
    label_aps: dict[tuple[int, float], float]
    tp_errors: dict[str, float]
    tp_scores: dict[str, float]
    label_tp_errors: dict[tuple[int, str], float]


@dataclass
class _BoxData:
    """Per-frame box attributes as numpy arrays.

    Attributes:
        center: ``[N, 2]`` bird's-eye-view (xy) centers.
        size: ``[N, 3]`` box extents.
        yaw: ``[N]`` yaw angles in radians.
        velocity: ``[N, 2]`` ground-plane velocities.
        attribute: ``[N]`` integer attribute labels (``-1`` for none).
        label: ``[N]`` integer class labels.
        score: ``[N]`` confidence scores (empty for ground truth).
    """

    center: np.ndarray
    size: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    attribute: np.ndarray
    label: np.ndarray
    score: np.ndarray


class NuScenesDetectionScore:
    r"""nuScenes Detection Score (NDS) metric.

    Predictions are matched to ground truth greedily by ascending 2D center
    distance (within each frame, one ground truth per prediction), pooled
    across all frames and sorted by descending score. Average precision is
    integrated per distance threshold with the nuScenes recall/precision
    clipping, and the five TP error metrics are measured at ``tp_threshold``.

    Args:
        class_ids: Integer class IDs to score. Predictions and ground
            truths with labels outside this set are ignored.
        dist_thresholds: BEV center-distance match thresholds in meters
            over which AP is averaged. Default ``(0.5, 1.0, 2.0, 4.0)``.
        tp_threshold: The single distance threshold at which the TP error
            metrics are computed. Must be one of ``dist_thresholds``.
            Default ``2.0``.
        min_recall: Recall below which precision is discarded before
            integrating AP and TP errors. Default ``0.1``.
        min_precision: Precision floor subtracted before integrating AP.
            Default ``0.1``.
        mean_ap_weight: Weight of mAP relative to the TP scores in the NDS.
            Default ``5.0``.
        tp_metrics: The TP error metrics that participate in the score, a
            subset of ``("trans_err", "scale_err", "orient_err", "vel_err",
            "attr_err")``. The NDS is normalized over ``mean_ap_weight``
            plus the number of active TP metrics, so dropping a metric
            cleanly removes it rather than scoring it as a perfect (or
            worst-case) constant. Default: all five (the nuScenes setting).
            Drop ``"vel_err"``/``"attr_err"`` for datasets without velocity
            or attribute annotations. ``"vel_err"`` requires per-box
            ``velocities`` and ``"attr_err"`` requires ``attributes`` on
            every frame passed to :meth:`update`.
        orientation_periods: Optional ``{class_id: period}`` overriding the
            orientation periodicity (radians) used for the orientation
            error. Defaults to :math:`2\pi` for every class.
        skip_tp_metrics: Optional ``{class_id: set_of_metric_names}`` forcing
            the listed TP metrics to ``nan`` for that class (and thus
            excluding it from those metrics' class averages).

    Raises:
        ValueError: If ``class_ids`` or ``dist_thresholds`` is empty, if
            ``tp_threshold`` is not in ``dist_thresholds``, or if
            ``tp_metrics`` is empty or names an unknown metric.
    """

    def __init__(
        self,
        class_ids: list[int],
        dist_thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        tp_threshold: float = 2.0,
        min_recall: float = 0.1,
        min_precision: float = 0.1,
        mean_ap_weight: float = 5.0,
        tp_metrics: tuple[str, ...] = TP_METRICS,
        orientation_periods: dict[int, float] | None = None,
        skip_tp_metrics: dict[int, set[str]] | None = None,
    ) -> None:
        if not class_ids:
            msg = "class_ids must be non-empty"
            raise ValueError(msg)
        if not dist_thresholds:
            msg = "dist_thresholds must be non-empty"
            raise ValueError(msg)
        if tp_threshold not in dist_thresholds:
            msg = (
                f"tp_threshold {tp_threshold} must be one of "
                f"dist_thresholds {dist_thresholds}"
            )
            raise ValueError(msg)
        if not tp_metrics:
            msg = "tp_metrics must be non-empty"
            raise ValueError(msg)
        unknown = [m for m in tp_metrics if m not in TP_METRICS]
        if unknown:
            msg = f"unknown tp_metrics {unknown}; valid names are {TP_METRICS}"
            raise ValueError(msg)
        self.class_ids = list(class_ids)
        self.dist_thresholds = tuple(dist_thresholds)
        self.tp_threshold = tp_threshold
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.mean_ap_weight = mean_ap_weight
        self.tp_metrics = tuple(tp_metrics)
        self.orientation_periods = dict(orientation_periods or {})
        self.skip_tp_metrics = {c: set(m) for c, m in (skip_tp_metrics or {}).items()}
        self._frames: list[tuple[_BoxData, _BoxData]] = []

    @classmethod
    def from_class_names(
        cls,
        class_names: list[str],
        **kwargs: object,
    ) -> "NuScenesDetectionScore":
        """Build the metric from class names using the nuScenes presets.

        Class IDs are assigned by position (``class_names[i] -> i``). For the
        official nuScenes classes this wires up the ``barrier`` half-period
        orientation and the ``barrier``/``traffic_cone`` TP-metric skips.

        Args:
            class_names: Ordered class names; their indices become the
                integer class IDs scored by the metric.
            **kwargs: Forwarded to :class:`NuScenesDetectionScore` (e.g.
                ``dist_thresholds``). ``orientation_periods`` and
                ``skip_tp_metrics`` are derived from the names and must not
                be passed here.

        Returns:
            A configured :class:`NuScenesDetectionScore`.

        Raises:
            ValueError: If ``orientation_periods`` or ``skip_tp_metrics`` is
                passed in ``kwargs`` (they are derived from the names).
        """
        forbidden = {"orientation_periods", "skip_tp_metrics"} & kwargs.keys()
        if forbidden:
            msg = f"{sorted(forbidden)} are derived from class_names"
            raise ValueError(msg)
        orientation_periods = {
            i: math.pi
            for i, name in enumerate(class_names)
            if name in _HALF_PERIOD_CLASSES
        }
        skip_tp_metrics = {
            i: set(_SKIP_TP_METRICS[name])
            for i, name in enumerate(class_names)
            if name in _SKIP_TP_METRICS
        }
        return cls(
            class_ids=list(range(len(class_names))),
            orientation_periods=orientation_periods,
            skip_tp_metrics=skip_tp_metrics,
            **kwargs,  # type: ignore[arg-type]
        )

    def update(
        self,
        preds: list[Prediction3D],
        targets: list[Target3D],
    ) -> None:
        """Accumulate one or more frames of predictions vs ground truth.

        Args:
            preds: List of per-frame :class:`Prediction3D` dicts.
            targets: List of per-frame :class:`Target3D` dicts.

        Raises:
            ValueError: If ``preds`` and ``targets`` differ in length, or if
                an active TP metric needs annotations (``velocities`` for
                ``vel_err``, ``attributes`` for ``attr_err``) that a frame
                does not provide.
        """
        if len(preds) != len(targets):
            msg = (
                f"preds and targets must have the same length; "
                f"got {len(preds)} vs {len(targets)}"
            )
            raise ValueError(msg)
        for pred, target in zip(preds, targets):
            self._check_required_fields(pred, target)
            pred_data = _to_box_data(
                pred["boxes"],
                pred["labels"],
                pred.get("velocities"),
                pred.get("attributes"),
                pred["scores"],
            )
            gt_data = _to_box_data(
                target["boxes"],
                target["labels"],
                target.get("velocities"),
                target.get("attributes"),
                None,
            )
            self._frames.append((pred_data, gt_data))

    def _check_required_fields(self, pred: Prediction3D, target: Target3D) -> None:
        """Verify a frame carries the annotations its active metrics need.

        Raises:
            ValueError: If ``vel_err`` is active but ``velocities`` is
                missing, or ``attr_err`` is active but ``attributes`` is
                missing, on either the prediction or the target.
        """
        required = {"vel_err": "velocities", "attr_err": "attributes"}
        for metric, field in required.items():
            if metric not in self.tp_metrics:
                continue
            for role, frame in (("preds", pred), ("targets", target)):
                if field not in frame:
                    msg = (
                        f"{metric!r} is an active TP metric but {role} is "
                        f"missing {field!r}; provide it or drop {metric!r} "
                        f"from tp_metrics"
                    )
                    raise ValueError(msg)

    def compute(self) -> NuScenesDetectionScoreResult:
        """Compute the aggregated NDS and its constituent metrics.

        Returns:
            A populated :class:`NuScenesDetectionScoreResult`.
        """
        label_aps: dict[tuple[int, float], float] = {}
        # ``_MetricData`` per (class, dist_th); reused for AP and TP metrics.
        md_by_key: dict[tuple[int, float], _MetricData] = {}
        for cls in self.class_ids:
            period = self.orientation_periods.get(cls, 2.0 * math.pi)
            for dist_th in self.dist_thresholds:
                md = _accumulate(self._frames, cls, dist_th, period)
                md_by_key[(cls, dist_th)] = md
                label_aps[(cls, dist_th)] = _calc_ap(
                    md, self.min_recall, self.min_precision
                )

        label_tp_errors: dict[tuple[int, str], float] = {}
        for cls in self.class_ids:
            skip = self.skip_tp_metrics.get(cls, set())
            md = md_by_key[(cls, self.tp_threshold)]
            for metric in self.tp_metrics:
                if metric in skip:
                    label_tp_errors[(cls, metric)] = math.nan
                else:
                    label_tp_errors[(cls, metric)] = _calc_tp(
                        md, self.min_recall, metric
                    )

        mean_dist_aps = {
            cls: float(np.mean([label_aps[(cls, t)] for t in self.dist_thresholds]))
            for cls in self.class_ids
        }
        mean_ap = float(np.mean(list(mean_dist_aps.values())))

        tp_errors: dict[str, float] = {}
        tp_scores: dict[str, float] = {}
        for metric in self.tp_metrics:
            errs = [label_tp_errors[(cls, metric)] for cls in self.class_ids]
            mean_err = float(np.nanmean(errs))
            tp_errors[metric] = mean_err
            tp_scores[metric] = max(0.0, 1.0 - mean_err)

        nd_score = (self.mean_ap_weight * mean_ap + sum(tp_scores.values())) / (
            self.mean_ap_weight + len(tp_scores)
        )

        return {
            "nd_score": nd_score,
            "mean_ap": mean_ap,
            "mean_dist_aps": mean_dist_aps,
            "label_aps": label_aps,
            "tp_errors": tp_errors,
            "tp_scores": tp_scores,
            "label_tp_errors": label_tp_errors,
        }

    def reset(self) -> None:
        """Clear all accumulated frames."""
        self._frames.clear()


@dataclass
class _MetricData:
    """Interpolated per-(class, threshold) curve, mirroring the devkit.

    Each array has length :data:`_NELEM`. ``confidence`` is descending and
    ``recall`` ascending.

    Attributes:
        recall: Interpolated recall levels.
        precision: Interpolated precision at each recall level.
        confidence: Interpolated confidence at each recall level.
        trans_err: Interpolated translation error.
        scale_err: Interpolated scale error.
        orient_err: Interpolated orientation error.
        vel_err: Interpolated velocity error.
        attr_err: Interpolated attribute error.
    """

    recall: np.ndarray
    precision: np.ndarray
    confidence: np.ndarray
    trans_err: np.ndarray
    scale_err: np.ndarray
    orient_err: np.ndarray
    vel_err: np.ndarray
    attr_err: np.ndarray

    @property
    def max_recall_ind(self) -> int:
        """Index of the maximum achieved recall (last nonzero confidence)."""
        nonzero = np.nonzero(self.confidence)[0]
        return int(nonzero[-1]) if len(nonzero) else 0


def _no_predictions_md() -> _MetricData:
    """Build the metric data for a class with no matched predictions.

    Returns:
        A :class:`_MetricData` with zero precision/confidence and unit
        (worst-case) TP errors.
    """
    return _MetricData(
        recall=np.linspace(0.0, 1.0, _NELEM),
        precision=np.zeros(_NELEM),
        confidence=np.zeros(_NELEM),
        trans_err=np.ones(_NELEM),
        scale_err=np.ones(_NELEM),
        orient_err=np.ones(_NELEM),
        vel_err=np.ones(_NELEM),
        attr_err=np.ones(_NELEM),
    )


def _accumulate(
    frames: list[tuple[_BoxData, _BoxData]],
    class_id: int,
    dist_th: float,
    orientation_period: float,
) -> _MetricData:
    """Match detections to ground truth and interpolate the metric curves.

    Mirrors ``nuscenes.eval.detection.algo.accumulate`` for a single class
    and distance threshold.

    Args:
        frames: Per-frame ``(prediction, ground_truth)`` box data.
        class_id: Class to score.
        dist_th: BEV center-distance match threshold in meters.
        orientation_period: Periodicity (radians) for the orientation error.

    Returns:
        The interpolated :class:`_MetricData` for this class/threshold.
    """
    npos = sum(int(np.count_nonzero(gt.label == class_id)) for _, gt in frames)
    if npos == 0:
        return _no_predictions_md()

    # Pool predictions across frames, remembering their originating frame.
    confs: list[float] = []
    frame_of: list[int] = []
    idx_of: list[int] = []
    for f_idx, (pred, _) in enumerate(frames):
        for i in np.nonzero(pred.label == class_id)[0]:
            confs.append(float(pred.score[i]))
            frame_of.append(f_idx)
            idx_of.append(int(i))

    # Sort by ascending (conf, position) then reverse -> descending order,
    # matching the devkit's tie-breaking exactly.
    sortind = sorted(range(len(confs)), key=lambda k: (confs[k], k))[::-1]

    tp: list[int] = []
    fp: list[int] = []
    conf: list[float] = []
    match_data: dict[str, list[float]] = {m: [] for m in TP_METRICS}
    match_conf: list[float] = []

    taken: set[tuple[int, int]] = set()
    for k in sortind:
        f_idx = frame_of[k]
        p_i = idx_of[k]
        pred, gt = frames[f_idx]

        gt_idxs = np.nonzero(gt.label == class_id)[0]
        min_dist = np.inf
        match_gt = -1
        for g_i in gt_idxs:
            g_i = int(g_i)
            if (f_idx, g_i) in taken:
                continue
            dist = float(np.linalg.norm(pred.center[p_i] - gt.center[g_i]))
            if dist < min_dist:
                min_dist = dist
                match_gt = g_i

        if min_dist < dist_th:
            taken.add((f_idx, match_gt))
            tp.append(1)
            fp.append(0)
            conf.append(confs[k])

            match_data["trans_err"].append(min_dist)
            match_data["vel_err"].append(
                float(np.linalg.norm(pred.velocity[p_i] - gt.velocity[match_gt]))
            )
            match_data["scale_err"].append(
                1.0 - _scale_iou(gt.size[match_gt], pred.size[p_i])
            )
            match_data["orient_err"].append(
                _yaw_diff(gt.yaw[match_gt], pred.yaw[p_i], orientation_period)
            )
            match_data["attr_err"].append(
                1.0 - _attr_acc(gt.attribute[match_gt], pred.attribute[p_i])
            )
            match_conf.append(confs[k])
        else:
            tp.append(0)
            fp.append(1)
            conf.append(confs[k])

    if len(match_conf) == 0:
        return _no_predictions_md()

    tp_arr = np.cumsum(tp).astype(float)
    fp_arr = np.cumsum(fp).astype(float)
    conf_arr = np.array(conf)

    prec = tp_arr / (fp_arr + tp_arr)
    rec = tp_arr / float(npos)

    rec_interp = np.linspace(0.0, 1.0, _NELEM)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf_interp = np.interp(rec_interp, rec, conf_arr, right=0)

    resampled: dict[str, np.ndarray] = {}
    match_conf_arr = np.array(match_conf)
    for metric in TP_METRICS:
        tmp = _cummean(np.array(match_data[metric]))
        # ``np.interp`` needs ascending sample points; confidences descend.
        resampled[metric] = np.interp(
            conf_interp[::-1], match_conf_arr[::-1], tmp[::-1]
        )[::-1]

    return _MetricData(
        recall=rec_interp,
        precision=prec,
        confidence=conf_interp,
        trans_err=resampled["trans_err"],
        scale_err=resampled["scale_err"],
        orient_err=resampled["orient_err"],
        vel_err=resampled["vel_err"],
        attr_err=resampled["attr_err"],
    )


def _calc_ap(md: _MetricData, min_recall: float, min_precision: float) -> float:
    """Integrate average precision with recall/precision clipping.

    Mirrors ``nuscenes.eval.detection.algo.calc_ap``.

    Returns:
        The clipped, normalized average precision in ``[0, 1]``.
    """
    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1 :]
    prec -= min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def _calc_tp(md: _MetricData, min_recall: float, metric_name: str) -> float:
    """Average a TP error metric over the valid recall range.

    Mirrors ``nuscenes.eval.detection.algo.calc_tp``.

    Returns:
        The mean error, or ``1.0`` if no valid recall range exists.
    """
    first_ind = round(100 * min_recall) + 1
    last_ind = md.max_recall_ind
    if last_ind < first_ind:
        return 1.0
    return float(np.mean(getattr(md, metric_name)[first_ind : last_ind + 1]))


def _scale_iou(size_a: np.ndarray, size_b: np.ndarray) -> float:
    """Aligned (translation/rotation-invariant) 3D scale IoU.

    Returns:
        Intersection-over-union of two boxes assumed perfectly aligned.
    """
    min_dims = np.minimum(size_a, size_b)
    vol_a = float(np.prod(size_a))
    vol_b = float(np.prod(size_b))
    intersection = float(np.prod(min_dims))
    union = vol_a + vol_b - intersection
    return intersection / union


def _yaw_diff(yaw_a: float, yaw_b: float, period: float) -> float:
    """Smallest absolute yaw difference under the given periodicity.

    Returns:
        Yaw difference in ``[0, period / 2]`` radians.
    """
    diff = (yaw_a - yaw_b + period / 2) % period - period / 2
    if diff > math.pi:
        diff -= 2 * math.pi
    return abs(diff)


def _attr_acc(gt_attr: float, pred_attr: float) -> float:
    """Attribute classification accuracy for a matched pair.

    Returns:
        ``nan`` if the ground truth has no attribute (negative label),
        otherwise ``1.0`` for a match and ``0.0`` otherwise.
    """
    if gt_attr < 0:
        return math.nan
    return float(gt_attr == pred_attr)


def _cummean(x: np.ndarray) -> np.ndarray:
    """NaN-aware cumulative mean.

    Returns:
        Cumulative mean ignoring NaNs, or all-ones if every value is NaN.
    """
    if np.all(np.isnan(x)):
        return np.ones(len(x))
    sum_vals = np.nancumsum(x.astype(float))
    count_vals = np.cumsum(~np.isnan(x))
    return np.divide(
        sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0
    )


def _to_box_data(
    boxes: "BoundingBoxes3D",
    labels: Tensor,
    velocities: Tensor | None,
    attributes: Tensor | None,
    scores: Tensor | None,
) -> _BoxData:
    """Convert a frame's boxes/labels/etc. to numpy ``_BoxData``.

    Centers (xy), sizes and yaw are derived from the box parameters via
    :func:`_extract_box_params`, so every supported box format is handled.

    Returns:
        The populated :class:`_BoxData`.
    """
    fmt: BoundingBox3DFormat = boxes.format
    n = boxes.shape[0]
    # Accumulate in float64 regardless of the input dtype: detection outputs
    # are typically float32, but cumulative sums, recall interpolation and
    # near-threshold distance comparisons are sensitive to rounding, so we
    # upcast (the metric is not on any hot path).
    boxes_t = boxes.as_subclass(Tensor).detach().double()
    centers, half_dims, rot = _extract_box_params(boxes_t, fmt)
    center_xy = centers[:, :2].cpu().numpy()
    size = (2.0 * half_dims).cpu().numpy()
    # Yaw of the box's local x-axis projected into the xy plane, matching the
    # devkit's ``quaternion_yaw``.
    yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0]).cpu().numpy()

    label = labels.detach().cpu().numpy()
    if velocities is not None:
        velocity = velocities[:, :2].detach().double().cpu().numpy()
    else:
        velocity = np.zeros((n, 2), dtype=np.float64)
    if attributes is not None:
        attribute = attributes.detach().cpu().numpy()
    else:
        attribute = np.full(n, -1)
    score = (
        scores.detach().double().cpu().numpy()
        if scores is not None
        else np.empty(0, dtype=np.float64)
    )
    return _BoxData(
        center=center_xy,
        size=size,
        yaw=yaw,
        velocity=velocity,
        attribute=attribute,
        label=label,
        score=score,
    )
