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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

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
    """Per-frame box attributes as CPU float64 / int64 tensors.

    Attributes:
        center: ``[N, 2]`` bird's-eye-view (xy) centers.
        size: ``[N, 3]`` box extents.
        yaw: ``[N]`` yaw angles in radians.
        velocity: ``[N, 2]`` ground-plane velocities.
        attribute: ``[N]`` integer attribute labels (``-1`` for none).
        label: ``[N]`` integer class labels.
        score: ``[N]`` confidence scores (empty for ground truth).
    """

    center: Tensor
    size: Tensor
    yaw: Tensor
    velocity: Tensor
    attribute: Tensor
    label: Tensor
    score: Tensor


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
            # Pin internal tensors to CPU (the matching loop is sequential
            # Python) regardless of any ambient default-device context.
            with torch.device("cpu"):
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
        # Pin to CPU regardless of any ambient default-device context.
        md_by_key: dict[tuple[int, float], _MetricData] = {}
        with torch.device("cpu"):
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
            cls: _mean(label_aps[(cls, t)] for t in self.dist_thresholds)
            for cls in self.class_ids
        }
        mean_ap = _mean(mean_dist_aps.values())

        tp_errors: dict[str, float] = {}
        tp_scores: dict[str, float] = {}
        for metric in self.tp_metrics:
            errs = [label_tp_errors[(cls, metric)] for cls in self.class_ids]
            mean_err = _nanmean(errs)
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

    Each tensor has length :data:`_NELEM`. ``confidence`` is descending and
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

    recall: Tensor
    precision: Tensor
    confidence: Tensor
    trans_err: Tensor
    scale_err: Tensor
    orient_err: Tensor
    vel_err: Tensor
    attr_err: Tensor

    @property
    def max_recall_ind(self) -> int:
        """Index of the maximum achieved recall (last nonzero confidence)."""
        nonzero = torch.nonzero(self.confidence, as_tuple=False)
        return int(nonzero[-1].item()) if nonzero.numel() else 0


def _no_predictions_md() -> _MetricData:
    """Build the metric data for a class with no matched predictions.

    Returns:
        A :class:`_MetricData` with zero precision/confidence and unit
        (worst-case) TP errors.
    """
    return _MetricData(
        recall=torch.linspace(0.0, 1.0, _NELEM, dtype=torch.float64),
        precision=torch.zeros(_NELEM, dtype=torch.float64),
        confidence=torch.zeros(_NELEM, dtype=torch.float64),
        trans_err=torch.ones(_NELEM, dtype=torch.float64),
        scale_err=torch.ones(_NELEM, dtype=torch.float64),
        orient_err=torch.ones(_NELEM, dtype=torch.float64),
        vel_err=torch.ones(_NELEM, dtype=torch.float64),
        attr_err=torch.ones(_NELEM, dtype=torch.float64),
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
    npos = sum(int((gt.label == class_id).sum()) for _, gt in frames)
    if npos == 0:
        return _no_predictions_md()

    # Greedy matching by descending score decomposes into independent per-frame
    # matching: a prediction only ever matches ground truth in its own frame.
    # Precompute each frame's prediction-vs-GT distance matrix once; the greedy
    # loop then indexes rows. ``donot_use_mm_for_euclid_dist`` forces the direct
    # formula so distances match the devkit's ``np.linalg.norm``.
    gt_local: list[Tensor] = []
    pred_local: list[Tensor] = []
    dist_mats: list[Tensor | None] = []
    taken: list[Tensor] = []
    for pred, gt in frames:
        g_idx = torch.nonzero(gt.label == class_id, as_tuple=False).flatten()
        p_idx = torch.nonzero(pred.label == class_id, as_tuple=False).flatten()
        gt_local.append(g_idx)
        pred_local.append(p_idx)
        taken.append(torch.zeros(g_idx.numel(), dtype=torch.bool))
        if p_idx.numel() and g_idx.numel():
            dist_mats.append(
                torch.cdist(
                    pred.center[p_idx],
                    gt.center[g_idx],
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
            )
        else:
            dist_mats.append(None)

    confs: list[float] = []
    frame_of: list[int] = []
    row_of: list[int] = []
    for f_idx, (pred, _) in enumerate(frames):
        for row, i in enumerate(pred_local[f_idx].tolist()):
            confs.append(float(pred.score[i]))
            frame_of.append(f_idx)
            row_of.append(row)

    # Descending score, tie-broken by position, matching the devkit exactly.
    sortind = sorted(range(len(confs)), key=lambda k: (confs[k], k))[::-1]

    tp: list[int] = []
    fp: list[int] = []
    conf: list[float] = []
    match_conf: list[float] = []
    # Translation error is the match distance; the other TP errors are
    # vectorized below from the matched index triples.
    trans_errs: list[float] = []
    match_frame: list[int] = []
    match_pred: list[int] = []
    match_gt_idx: list[int] = []

    for k in sortind:
        f_idx = frame_of[k]
        dmat = dist_mats[f_idx]

        min_dist = math.inf
        match_rel = -1
        if dmat is not None:
            # ``argmin`` over untaken GTs reproduces the loop's earliest-of-ties pick.
            dists = dmat[row_of[k]].masked_fill(taken[f_idx], math.inf)
            best_dist, best_rel = torch.min(dists, dim=0)
            min_dist = float(best_dist)
            match_rel = int(best_rel)

        if min_dist < dist_th:
            taken[f_idx][match_rel] = True
            tp.append(1)
            fp.append(0)
            conf.append(confs[k])
            trans_errs.append(min_dist)
            match_frame.append(f_idx)
            match_pred.append(int(pred_local[f_idx][row_of[k]]))
            match_gt_idx.append(int(gt_local[f_idx][match_rel]))
            match_conf.append(confs[k])
        else:
            tp.append(0)
            fp.append(1)
            conf.append(confs[k])

    if len(match_conf) == 0:
        return _no_predictions_md()

    tp_cum = torch.tensor(tp, dtype=torch.float64).cumsum(0)
    fp_cum = torch.tensor(fp, dtype=torch.float64).cumsum(0)
    conf_t = torch.tensor(conf, dtype=torch.float64)

    prec = tp_cum / (fp_cum + tp_cum)
    rec = tp_cum / float(npos)

    rec_interp = torch.linspace(0.0, 1.0, _NELEM, dtype=torch.float64)
    prec_i = _interp(rec_interp, rec, prec, right=0.0)
    conf_i = _interp(rec_interp, rec, conf_t, right=0.0)

    match_errors = _match_errors(
        frames,
        trans_errs,
        match_frame,
        match_pred,
        match_gt_idx,
        orientation_period,
    )

    resampled: dict[str, Tensor] = {}
    match_conf_t = torch.tensor(match_conf, dtype=torch.float64)
    for metric in TP_METRICS:
        tmp = _cummean(match_errors[metric])
        # ``_interp`` needs ascending sample points; confidences descend.
        resampled[metric] = _interp(
            conf_i.flip(0), match_conf_t.flip(0), tmp.flip(0)
        ).flip(0)

    return _MetricData(
        recall=rec_interp,
        precision=prec_i,
        confidence=conf_i,
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
    prec = md.precision[round(100 * min_recall) + 1 :]
    prec = (prec - min_precision).clamp_min(0.0)
    return float(prec.mean()) / (1.0 - min_precision)


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
    errors: Tensor = getattr(md, metric_name)
    return float(errors[first_ind : last_ind + 1].mean())


def _interp(x: Tensor, xp: Tensor, fp: Tensor, right: float | None = None) -> Tensor:
    """1-D linear interpolation matching :func:`numpy.interp`.

    ``xp`` must be non-decreasing (duplicates allowed). Queries below
    ``xp[0]`` return ``fp[0]``; queries above ``xp[-1]`` return ``right`` if
    given, else ``fp[-1]``. On a run of equal ``xp`` the value at the last
    occurrence is used, reproducing ``numpy``'s behavior exactly.

    Returns:
        Interpolated values, shaped like ``x``.
    """
    left_val = fp[0]
    right_val = (
        fp[-1]
        if right is None
        else torch.tensor(right, dtype=fp.dtype, device=fp.device)
    )
    # ``side="right"`` counts xp <= x, so hi-1 lands on the last equal node.
    hi = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    lo = hi - 1
    x0, x1 = xp[lo], xp[hi]
    f0, f1 = fp[lo], fp[hi]
    denom = x1 - x0
    safe = torch.where(denom != 0, denom, torch.ones_like(denom))
    slope = (f1 - f0) / safe
    # ``denom == 0`` only when the query lands on a duplicate run clamped at
    # the top of ``xp``; numpy resolves that to the last node's value (``f1``).
    res = torch.where(denom != 0, f0 + slope * (x - x0), f1)
    res = torch.where(x < xp[0], left_val, res)
    return torch.where(x > xp[-1], right_val, res)


def _match_errors(
    frames: list[tuple[_BoxData, _BoxData]],
    trans_errs: list[float],
    match_frame: list[int],
    match_pred: list[int],
    match_gt: list[int],
    orientation_period: float,
) -> dict[str, Tensor]:
    """Vectorized per-true-positive TP errors, in match order.

    Translation error is the match distance, already collected. The velocity,
    scale, orientation and attribute errors are computed here in one batched
    pass per frame (the greedy match is sequential, but these errors are not),
    then scattered back into the order the matches were made.

    Args:
        frames: Per-frame ``(prediction, ground_truth)`` box data.
        trans_errs: Translation error (match distance) per true positive.
        match_frame: Originating frame index per true positive.
        match_pred: Matched prediction index (within its frame) per TP.
        match_gt: Matched ground-truth index (within its frame) per TP.
        orientation_period: Periodicity (radians) for the orientation error.

    Returns:
        ``{metric_name: [n_match] tensor}`` for every entry of
        :data:`TP_METRICS`.
    """
    n = len(trans_errs)
    out = {
        "trans_err": torch.tensor(trans_errs, dtype=torch.float64),
        "vel_err": torch.empty(n, dtype=torch.float64),
        "scale_err": torch.empty(n, dtype=torch.float64),
        "orient_err": torch.empty(n, dtype=torch.float64),
        "attr_err": torch.empty(n, dtype=torch.float64),
    }
    groups: dict[int, list[int]] = {}
    for pos, f_idx in enumerate(match_frame):
        groups.setdefault(f_idx, []).append(pos)

    half = orientation_period / 2.0
    for f_idx, positions in groups.items():
        pred, gt = frames[f_idx]
        pos_t = torch.tensor(positions)
        p_i = torch.tensor([match_pred[p] for p in positions])
        g_i = torch.tensor([match_gt[p] for p in positions])

        out["vel_err"][pos_t] = torch.linalg.norm(
            pred.velocity[p_i] - gt.velocity[g_i], dim=1
        )

        # Aligned (translation/rotation-invariant) scale IoU.
        sa, sb = gt.size[g_i], pred.size[p_i]
        inter = torch.minimum(sa, sb).prod(dim=1)
        union = sa.prod(dim=1) + sb.prod(dim=1) - inter
        out["scale_err"][pos_t] = 1.0 - inter / union

        # Smallest yaw difference under the (per-class) periodicity.
        diff = (gt.yaw[g_i] - pred.yaw[p_i] + half) % orientation_period - half
        diff = torch.where(diff > math.pi, diff - 2.0 * math.pi, diff)
        out["orient_err"][pos_t] = diff.abs()

        # Attribute error: 1 - accuracy, or NaN where GT has no attribute.
        gt_attr, pred_attr = gt.attribute[g_i], pred.attribute[p_i]
        attr_err = 1.0 - (gt_attr == pred_attr).to(torch.float64)
        out["attr_err"][pos_t] = attr_err.masked_fill(gt_attr < 0, math.nan)

    return out


def _cummean(x: Tensor) -> Tensor:
    """NaN-aware cumulative mean.

    Returns:
        Cumulative mean ignoring NaNs, or all-ones if every value is NaN.
    """
    is_nan = torch.isnan(x)
    if bool(is_nan.all()):
        return torch.ones_like(x)
    sum_vals = torch.nan_to_num(x, nan=0.0).cumsum(0)
    count = (~is_nan).cumsum(0).to(x.dtype)
    safe = torch.where(count != 0, count, torch.ones_like(count))
    return torch.where(count != 0, sum_vals / safe, torch.zeros_like(sum_vals))


def _mean(values: Iterable[float]) -> float:
    """Arithmetic mean of a finite, non-empty iterable of floats.

    Returns:
        The mean.
    """
    vals = list(values)
    return sum(vals) / len(vals)


def _nanmean(values: Iterable[float]) -> float:
    """Mean over the non-NaN entries.

    Returns:
        The mean of the non-NaN values, or ``nan`` if all are NaN.
    """
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def _to_box_data(
    boxes: "BoundingBoxes3D",
    labels: Tensor,
    velocities: Tensor | None,
    attributes: Tensor | None,
    scores: Tensor | None,
) -> _BoxData:
    """Convert a frame's boxes/labels/etc. to CPU float64 ``_BoxData`` tensors.

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
    # upcast (the metric is not on any hot path). Everything is moved to CPU
    # since the matching loop is sequential Python.
    boxes_t = boxes.as_subclass(Tensor).detach().double().cpu()
    centers, half_dims, rot = _extract_box_params(boxes_t, fmt)
    center_xy = centers[:, :2]
    size = 2.0 * half_dims
    # Yaw of the box's local x-axis projected into the xy plane, matching the
    # devkit's ``quaternion_yaw``.
    yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])

    label = labels.detach().cpu()
    if velocities is not None:
        velocity = velocities[:, :2].detach().double().cpu()
    else:
        velocity = torch.zeros((n, 2), dtype=torch.float64)
    if attributes is not None:
        attribute = attributes.detach().cpu()
    else:
        attribute = torch.full((n,), -1)
    score = (
        scores.detach().double().cpu()
        if scores is not None
        else torch.empty(0, dtype=torch.float64)
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
