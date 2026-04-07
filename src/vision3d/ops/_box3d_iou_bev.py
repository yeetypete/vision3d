"""Bird's eye view (BEV) IoU for 3D bounding boxes."""

import torch
from torch import Tensor

from vision3d.tensors import BoundingBox3DFormat


def box3d_iou_bev(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Compute axis-aligned BEV IoU between two sets of 3D boxes.

    Projects boxes onto the XY plane and computes intersection-over-union
    using axis-aligned bounding rectangles. For rotated formats, the
    axis-aligned BEV bounding rectangle of each rotated box is used
    (conservative approximation).

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        IoU matrix ``[N, M]`` with values in ``[0, 1]``.
    """
    bev1 = _to_bev_aabb(boxes1, format)  # [N, 4]
    bev2 = _to_bev_aabb(boxes2, format)  # [M, 4]
    return _aabb_iou(bev1, bev2)


def box3d_overlap_bev(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Check BEV overlap between two sets of 3D boxes.

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        Boolean matrix ``[N, M]`` where True indicates overlap.
    """
    return box3d_iou_bev(boxes1, boxes2, format) > 0


def _to_bev_aabb(boxes: Tensor, format: BoundingBox3DFormat) -> Tensor:
    """Convert boxes to axis-aligned BEV rectangles ``[N, 4]`` as (x_min, y_min, x_max, y_max).

    For rotated boxes, computes the 4 BEV corners and takes the
    axis-aligned bounding rectangle.

    Returns:
        ``[N, 4]`` tensor of ``(x_min, y_min, x_max, y_max)``.

    Raises:
        ValueError: If ``format`` is not supported.
    """
    if format is BoundingBox3DFormat.XYZXYZ:
        return torch.stack([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]], dim=-1)

    if format is BoundingBox3DFormat.XYZLWH:
        cx, cy = boxes[:, 0], boxes[:, 1]
        hl, hw = boxes[:, 3] / 2, boxes[:, 4] / 2
        return torch.stack([cx - hl, cy - hw, cx + hl, cy + hw], dim=-1)

    if format in (BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR):
        cx, cy = boxes[:, 0], boxes[:, 1]
        hl, hw = boxes[:, 3] / 2, boxes[:, 4] / 2
        yaw = boxes[:, 6]

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        # 4 BEV corners relative to center
        dx = torch.stack([hl, hl, -hl, -hl], dim=-1)  # [N, 4]
        dy = torch.stack([hw, -hw, -hw, hw], dim=-1)  # [N, 4]

        # Rotate corners
        rx = dx * cos_y.unsqueeze(-1) - dy * sin_y.unsqueeze(-1)
        ry = dx * sin_y.unsqueeze(-1) + dy * cos_y.unsqueeze(-1)

        # Axis-aligned bounding rectangle
        x_min = cx + rx.amin(dim=-1)
        x_max = cx + rx.amax(dim=-1)
        y_min = cy + ry.amin(dim=-1)
        y_max = cy + ry.amax(dim=-1)

        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    msg = f"Unsupported format: {format}"
    raise ValueError(msg)


def _aabb_iou(bev1: Tensor, bev2: Tensor) -> Tensor:
    """Compute IoU between two sets of axis-aligned 2D rectangles.

    Args:
        bev1: ``[N, 4]`` as ``(x_min, y_min, x_max, y_max)``.
        bev2: ``[M, 4]`` as ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``[N, M]`` IoU matrix.
    """
    # [N, 1, 4] and [1, M, 4]
    b1 = bev1.unsqueeze(1)
    b2 = bev2.unsqueeze(0)

    inter_x = (
        torch.min(b1[..., 2], b2[..., 2]) - torch.max(b1[..., 0], b2[..., 0])
    ).clamp(min=0)
    inter_y = (
        torch.min(b1[..., 3], b2[..., 3]) - torch.max(b1[..., 1], b2[..., 1])
    ).clamp(min=0)
    inter_area = inter_x * inter_y

    area1 = (bev1[:, 2] - bev1[:, 0]) * (bev1[:, 3] - bev1[:, 1])  # [N]
    area2 = (bev2[:, 2] - bev2[:, 0]) * (bev2[:, 3] - bev2[:, 1])  # [M]

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    return inter_area / union.clamp(min=1e-8)
