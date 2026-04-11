"""3D detection mean Average Precision metric."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch
from torch import Tensor

from vision3d.ops import box3d_iou
from vision3d.ops._points_in_boxes_3d import _extract_box_params

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vision3d.metrics._types import Prediction3D, Target3D
    from vision3d.tensors import BoundingBox3DFormat


class APInterpolation(Enum):
    """AP interpolation mode."""

    R40 = "r40"
    """40-point interpolation (modern KITTI default)."""

    R11 = "r11"
    """11-point interpolation (legacy KITTI, PASCAL VOC07)."""

    R101 = "r101"
    """101-point interpolation (COCO)."""

    ALL_POINTS = "all_points"
    """VOC07 area-under-curve at every recall change."""


_RangeBin = tuple[float, float]


@dataclass(frozen=True)
class _DetectionStatsKey:
    """Key into the per-bucket accumulator dict.

    Attributes:
        class_id: Integer class ID this bucket scores.
        iou_threshold: IoU threshold at which TP/FP were decided.
        range_bin: ``(low, high)`` distance bounds in meters, or
            ``None`` when range bucketing is disabled.
    """

    class_id: int
    iou_threshold: float
    range_bin: _RangeBin | None


@dataclass
class _DetectionStats:
    """Per-bucket accumulator for AP computation.

    Attributes:
        scores: Per-frame prediction score chunks.
        is_tp: Per-frame true-positive flag chunks; ``is_tp[f][i]``
            is ``True`` iff prediction ``i`` of frame ``f`` was
            matched to a ground-truth box at this bucket's IoU
            threshold.
        num_gt: Total ground-truth boxes seen for this bucket.
    """

    scores: list[Tensor] = field(default_factory=list)
    is_tp: list[Tensor] = field(default_factory=list)
    num_gt: int = 0


class MeanAveragePrecision3DResult(TypedDict):
    """Structured result returned by :meth:`MeanAveragePrecision3D.compute`.

    Undefined slots (buckets with no ground-truth boxes accumulated)
    are reported as ``-1.0`` and callers can filter them with
    ``x >= 0``.

    Attributes:
        mAP: Overall mean AP, taken over every defined
            ``(class, iou, bin)`` bucket.
        mAP_per_class: AP per class, averaged over the other axes.
        AP_per_iou: AP per IoU threshold, averaged over the other axes.
        AP_per_class_per_iou: AP per ``(class, iou)`` pair, averaged
            over range bins (or a single value when range bucketing is
            disabled).
        AP_per_range: AP per range bin, averaged over the other axes.
            Only present when ``range_bins`` was set on the metric.
        AP_per_class_per_range: AP per ``(class, range_bin)`` pair,
            averaged over IoU thresholds. Only present when
            ``range_bins`` was set on the metric.
    """

    mAP: float
    mAP_per_class: dict[int, float]
    AP_per_iou: dict[float, float]
    AP_per_class_per_iou: dict[tuple[int, float], float]
    AP_per_range: NotRequired[dict[_RangeBin, float]]
    AP_per_class_per_range: NotRequired[dict[tuple[int, _RangeBin], float]]


class MeanAveragePrecision3D:
    """3D detection mAP metric.

    Matching is greedy by descending score, one prediction to one
    ground truth, with precision/recall accumulated globally across
    frames (KITTI convention).

    Args:
        class_ids: Integer class IDs to score. Predictions and GTs
            with labels outside this set are ignored.
        iou_thresholds: IoU thresholds to report AP at. Default
            ``(0.5, 0.7)``.
        ap_interpolation: Interpolation mode. Default :attr:`APInterpolation.R40`.
        range_bins: Optional distance bins ``[low, high)`` in meters
            from the sensor origin. When set, AP is also broken down
            per bin; boxes are bucketed by their center's distance.
    """

    def __init__(
        self,
        class_ids: list[int],
        iou_thresholds: tuple[float, ...] = (0.5, 0.7),
        ap_interpolation: APInterpolation = APInterpolation.R40,
        range_bins: tuple[_RangeBin, ...] | None = None,
    ) -> None:
        if not class_ids:
            msg = "class_ids must be non-empty"
            raise ValueError(msg)
        if not iou_thresholds:
            msg = "iou_thresholds must be non-empty"
            raise ValueError(msg)
        self.class_ids = list(class_ids)
        self.iou_thresholds = tuple(iou_thresholds)
        self.ap_interpolation = ap_interpolation
        self.range_bins = tuple(range_bins) if range_bins is not None else None
        self._state: dict[_DetectionStatsKey, _DetectionStats] = {}

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
            ValueError: If ``preds`` and ``targets`` differ in length.
        """
        if len(preds) != len(targets):
            msg = (
                f"preds and targets must have the same length; "
                f"got {len(preds)} vs {len(targets)}"
            )
            raise ValueError(msg)
        for pred, target in zip(preds, targets):
            self._update_frame(pred, target)

    def _update_frame(self, pred: Prediction3D, target: Target3D) -> None:
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        fmt: BoundingBox3DFormat = gt_boxes.format

        pred_centers, _, _ = _extract_box_params(pred_boxes, fmt)
        gt_centers, _, _ = _extract_box_params(gt_boxes, fmt)
        pred_dist = pred_centers.norm(dim=-1)
        gt_dist = gt_centers.norm(dim=-1)

        for range_bin in self.range_bins or (None,):
            if range_bin is None:
                pred_in_bin = torch.ones_like(pred_dist, dtype=torch.bool)
                gt_in_bin = torch.ones_like(gt_dist, dtype=torch.bool)
            else:
                low, high = range_bin
                pred_in_bin = (pred_dist >= low) & (pred_dist < high)
                gt_in_bin = (gt_dist >= low) & (gt_dist < high)

            for cls in self.class_ids:
                p_mask = pred_in_bin & (pred_labels == cls)
                g_mask = gt_in_bin & (gt_labels == cls)

                cls_pred_boxes = pred_boxes[p_mask]
                cls_pred_scores = pred_scores[p_mask]
                cls_gt_boxes = gt_boxes[g_mask]

                n_pred = cls_pred_boxes.shape[0]
                n_gt = cls_gt_boxes.shape[0]
                if n_pred == 0 and n_gt == 0:
                    continue

                if n_pred > 0 and n_gt > 0:
                    iou = box3d_iou(cls_pred_boxes, cls_gt_boxes, fmt)
                else:
                    iou = torch.zeros(
                        n_pred, n_gt, dtype=torch.float32, device=pred_boxes.device
                    )

                for thresh in self.iou_thresholds:
                    key = _DetectionStatsKey(cls, thresh, range_bin)
                    state = self._state.setdefault(key, _DetectionStats())
                    state.num_gt += n_gt

                    if n_pred == 0:
                        continue

                    is_tp = _greedy_match(cls_pred_scores, iou, thresh)
                    state.scores.append(cls_pred_scores.detach())
                    state.is_tp.append(is_tp)

    def compute(self) -> MeanAveragePrecision3DResult:
        """Compute the aggregated metric.

        Returns:
            Populated :class:`MeanAveragePrecision3DResult`.
        """
        ap_by_key: dict[_DetectionStatsKey, float] = {}
        for key, state in self._state.items():
            if state.num_gt == 0:
                continue
            scores = (
                torch.cat(state.scores)
                if state.scores
                else torch.empty(0, dtype=torch.float32)
            )
            is_tp = (
                torch.cat(state.is_tp)
                if state.is_tp
                else torch.empty(0, dtype=torch.bool)
            )
            ap_by_key[key] = _compute_ap(
                scores, is_tp, state.num_gt, self.ap_interpolation
            )

        result: MeanAveragePrecision3DResult = {
            "mAP": _mean_defined(ap_by_key.values()),
            "mAP_per_class": {},
            "AP_per_iou": {},
            "AP_per_class_per_iou": {},
        }

        per_class: dict[int, list[float]] = {c: [] for c in self.class_ids}
        for key, ap in ap_by_key.items():
            per_class[key.class_id].append(ap)
        result["mAP_per_class"] = {c: _mean_defined(v) for c, v in per_class.items()}

        per_iou: dict[float, list[float]] = {t: [] for t in self.iou_thresholds}
        for key, ap in ap_by_key.items():
            per_iou[key.iou_threshold].append(ap)
        result["AP_per_iou"] = {t: _mean_defined(v) for t, v in per_iou.items()}

        per_class_iou: dict[tuple[int, float], list[float]] = {}
        for key, ap in ap_by_key.items():
            per_class_iou.setdefault((key.class_id, key.iou_threshold), []).append(ap)
        result["AP_per_class_per_iou"] = {
            k: _mean_defined(v) for k, v in per_class_iou.items()
        }

        if self.range_bins is not None:
            per_range: dict[_RangeBin, list[float]] = {b: [] for b in self.range_bins}
            for key, ap in ap_by_key.items():
                if key.range_bin is not None:
                    per_range[key.range_bin].append(ap)
            result["AP_per_range"] = {b: _mean_defined(v) for b, v in per_range.items()}

            per_class_range: dict[tuple[int, _RangeBin], list[float]] = {}
            for key, ap in ap_by_key.items():
                if key.range_bin is not None:
                    per_class_range.setdefault(
                        (key.class_id, key.range_bin), []
                    ).append(ap)
            result["AP_per_class_per_range"] = {
                k: _mean_defined(v) for k, v in per_class_range.items()
            }

        return result

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._state.clear()


def _greedy_match(scores: Tensor, iou: Tensor, threshold: float) -> Tensor:
    """Greedy one-to-one matching, preds ordered by descending score.

    Args:
        scores: ``[N_pred]`` prediction confidences.
        iou: ``[N_pred, N_gt]`` IoU matrix.
        threshold: IoU threshold below which no match is made.

    Returns:
        Boolean ``[N_pred]`` mask where ``True`` means the prediction
        was assigned a GT above the threshold.
    """
    n_pred, n_gt = iou.shape
    is_tp = torch.zeros(n_pred, dtype=torch.bool, device=iou.device)
    if n_pred == 0 or n_gt == 0:
        return is_tp

    gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=iou.device)
    order = scores.argsort(descending=True).tolist()

    neg_inf = torch.full((), -1.0, dtype=iou.dtype, device=iou.device)
    for i in order:
        row = torch.where(gt_matched, neg_inf, iou[i])
        best_val, best_j = row.max(dim=0)
        if best_val.item() >= threshold:
            is_tp[i] = True
            gt_matched[best_j] = True
    return is_tp


def _compute_ap(
    scores: Tensor,
    is_tp: Tensor,
    num_gt: int,
    interpolation: APInterpolation,
) -> float:
    """Compute AP from per-prediction (score, is_tp) tensors and num_gt.

    Returns:
        AP in ``[0, 1]``, or ``-1.0`` if ``num_gt == 0`` (undefined).

    Raises:
        ValueError: If ``interpolation`` is not a known :class:`APInterpolation`.
    """
    if num_gt == 0:
        return -1.0
    if scores.numel() == 0:
        return 0.0

    scores = scores.to(torch.float32).cpu()
    is_tp_f = is_tp.to(torch.float32).cpu()

    order = scores.argsort(descending=True)
    tp_cum = is_tp_f[order].cumsum(dim=0)
    fp_cum = (1.0 - is_tp_f[order]).cumsum(dim=0)
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / num_gt

    # Right envelope: precisions[i] = max(precisions[i:]).
    precisions = precisions.flip(0).cummax(dim=0).values.flip(0)

    if interpolation == APInterpolation.R11:
        targets = torch.linspace(0.0, 1.0, 11, dtype=torch.float32)
        return _sample_ap(recalls, precisions, targets)
    if interpolation == APInterpolation.R40:
        targets = torch.arange(1, 41, dtype=torch.float32) / 40.0
        return _sample_ap(recalls, precisions, targets)
    if interpolation == APInterpolation.R101:
        targets = torch.linspace(0.0, 1.0, 101, dtype=torch.float32)
        return _sample_ap(recalls, precisions, targets)
    if interpolation == APInterpolation.ALL_POINTS:
        return _all_points_ap(recalls, precisions)
    msg = f"unknown interpolation: {interpolation}"
    raise ValueError(msg)


def _sample_ap(recalls: Tensor, precisions: Tensor, targets: Tensor) -> float:
    """Sample precision at each target recall level and average.

    Returns:
        Mean of sampled precisions at the target recall levels.
    """
    idx = torch.searchsorted(recalls, targets)
    sampled = torch.zeros_like(targets)
    valid = idx < recalls.numel()
    sampled[valid] = precisions[idx[valid]]
    return float(sampled.mean().item())


def _all_points_ap(recalls: Tensor, precisions: Tensor) -> float:
    """VOC07 area-under-curve AP at every recall change.

    Returns:
        Area under the right-enveloped precision-recall curve.
    """
    zero = torch.zeros(1, dtype=recalls.dtype)
    one = torch.ones(1, dtype=recalls.dtype)
    mrec = torch.cat([zero, recalls, one])
    mpre = torch.cat([zero, precisions, zero])
    mpre = mpre.flip(0).cummax(dim=0).values.flip(0)
    deltas = mrec[1:] - mrec[:-1]
    return float((deltas * mpre[1:]).sum().item())


def _mean_defined(values: Iterable[float]) -> float:
    """Mean over non-sentinel values, dropping ``-1`` entries.

    Returns:
        Mean of the valid entries, or ``-1.0`` if none are valid.
    """
    vs = [v for v in values if v >= 0]
    if not vs:
        return -1.0
    return sum(vs) / len(vs)
