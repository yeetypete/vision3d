"""3D non-maximum suppression."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._box3d_iou import box3d_iou

if TYPE_CHECKING:
    from vision3d.tensors import BoundingBox3DFormat


@torch.no_grad()
def nms_3d[N, K, M](
    boxes: Tensor[N, K],
    scores: Tensor[N],
    iou_threshold: float,
    format: BoundingBox3DFormat,
) -> Tensor[M]:
    """Greedy, class-agnostic non-maximum suppression on 3D bounding boxes.

    Iteratively removes lower-scoring boxes whose IoU with a
    higher-scoring box exceeds ``iou_threshold``.

    Args:
        boxes: ``[N, K]`` boxes to perform NMS on. ``K`` depends on
            ``format``.
        scores: ``[N]`` prediction confidences.
        iou_threshold: Discard any box whose IoU with a higher-scoring
            kept box is strictly greater than this value.
        format: Format of ``boxes``.

    Returns:
        ``int64`` tensor of indices into ``boxes`` that survived, sorted
        in decreasing order of score.
    """
    n = boxes.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)  # pyrefly: ignore[bad-return]

    order = scores.argsort(descending=True)
    boxes_sorted = boxes[order]
    iou = box3d_iou(boxes_sorted, boxes_sorted, format)  # [N, N]

    keep_mask = torch.ones(n, dtype=torch.bool, device=boxes.device)
    for i in range(n):
        if not keep_mask[i]:
            continue
        if i + 1 >= n:
            break
        keep_mask[i + 1 :] &= iou[i, i + 1 :] <= iou_threshold

    return order[keep_mask]


@torch.no_grad()
def batched_nms_3d[N, K, M](
    boxes: Tensor[N, K],
    scores: Tensor[N],
    idxs: Tensor[N],
    iou_threshold: float,
    format: BoundingBox3DFormat,
) -> Tensor[M]:
    """Class-aware 3D NMS: runs :func:`nms_3d` independently per class.

    Args:
        boxes: ``[N, K]`` boxes.
        scores: ``[N]`` prediction confidences.
        idxs: ``[N]`` integer class labels.
        iou_threshold: See :func:`nms_3d`.
        format: Format of ``boxes``.

    Returns:
        ``int64`` tensor of indices into ``boxes`` that survived, sorted
        in decreasing order of score.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)  # pyrefly: ignore[bad-return]

    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        in_class = torch.where(idxs == class_id)[0]
        class_keep = nms_3d(boxes[in_class], scores[in_class], iou_threshold, format)
        keep_mask[in_class[class_keep]] = True

    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].argsort(descending=True)]
