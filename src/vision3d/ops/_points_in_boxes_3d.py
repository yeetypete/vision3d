"""Point-in-box tests for 3D bounding boxes."""

import torch
from torch import Tensor

from vision3d.tensors import BoundingBox3DFormat


def points_in_boxes_3d(
    points: Tensor,
    boxes: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Compute a boolean mask indicating which points fall inside which boxes.

    Args:
        points: Point cloud coordinates ``[N, 3+C]``. Only the first 3
            columns (x, y, z) are used.
        boxes: 3D bounding boxes ``[M, K]`` where K depends on format.
        format: Format of the bounding boxes.

    Returns:
        Boolean tensor ``[N, M]`` where entry ``(i, j)`` is True if
        point ``i`` is inside box ``j``.
    """
    centers, half_dims, yaw = _extract_box_params(boxes, format)
    return _points_in_aabb_rotated(points[:, :3], centers, half_dims, yaw)


def points_in_boxes_3d_indices(
    points: Tensor,
    boxes: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Return per-point box assignment.

    If a point is inside multiple boxes, the first (lowest index) box wins.

    Args:
        points: Point cloud coordinates ``[N, 3+C]``.
        boxes: 3D bounding boxes ``[M, K]``.
        format: Format of the bounding boxes.

    Returns:
        Integer tensor ``[N]`` with the index of the box each point
        belongs to, or ``-1`` if the point is not in any box.
    """
    mask = points_in_boxes_3d(points, boxes, format)  # [N, M]
    n = points.shape[0]
    if mask.shape[1] == 0:
        return torch.full((n,), -1, dtype=torch.long, device=points.device)
    # First True along dim=1; if none, returns M (out of bounds)
    first_box = mask.to(torch.uint8).argmax(dim=1)
    # Points not in any box: all False along dim=1
    in_any = mask.any(dim=1)
    first_box[~in_any] = -1
    return first_box


def _extract_box_params(
    boxes: Tensor, format: BoundingBox3DFormat
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract centers, half-dimensions, and yaw from boxes.

    Returns:
        Tuple of (centers ``[M, 3]``, half_dims ``[M, 3]``, yaw ``[M]``).

    Raises:
        ValueError: If ``format`` is not a supported format.
    """
    if format is BoundingBox3DFormat.XYZXYZ:
        mins = boxes[:, :3]
        maxs = boxes[:, 3:6]
        centers = (mins + maxs) / 2
        half_dims = (maxs - mins) / 2
        yaw = torch.zeros(boxes.shape[0], dtype=boxes.dtype, device=boxes.device)
    elif format is BoundingBox3DFormat.XYZLWH:
        centers = boxes[:, :3]
        half_dims = boxes[:, 3:6] / 2
        yaw = torch.zeros(boxes.shape[0], dtype=boxes.dtype, device=boxes.device)
    elif (
        format is BoundingBox3DFormat.XYZLWHY or format is BoundingBox3DFormat.XYZLWHYPR
    ):
        centers = boxes[:, :3]
        half_dims = boxes[:, 3:6] / 2
        yaw = boxes[:, 6]
        # Ignore pitch/roll — treat as yaw-only
    else:
        msg = f"Unsupported format: {format}"
        raise ValueError(msg)
    return centers, half_dims, yaw


def _points_in_aabb_rotated(
    xyz: Tensor, centers: Tensor, half_dims: Tensor, yaw: Tensor
) -> Tensor:
    """Check if points are inside yaw-rotated boxes.

    Args:
        xyz: Point positions ``[N, 3]``.
        centers: Box centers ``[M, 3]``.
        half_dims: Box half-extents ``[M, 3]`` (half_l, half_w, half_h).
        yaw: Box yaw angles ``[M]`` in radians.

    Returns:
        Boolean ``[N, M]``.
    """
    # Relative positions: [N, 1, 3] - [1, M, 3] -> [N, M, 3]
    rel = xyz.unsqueeze(1) - centers.unsqueeze(0)

    # Rotate into box local frame by -yaw (only x, y)
    cos_y = torch.cos(-yaw)  # [M]
    sin_y = torch.sin(-yaw)  # [M]

    local_x = rel[..., 0] * cos_y - rel[..., 1] * sin_y  # [N, M]
    local_y = rel[..., 0] * sin_y + rel[..., 1] * cos_y  # [N, M]
    local_z = rel[..., 2]  # [N, M]

    return (
        (local_x.abs() <= half_dims[:, 0])
        & (local_y.abs() <= half_dims[:, 1])
        & (local_z.abs() <= half_dims[:, 2])
    )
