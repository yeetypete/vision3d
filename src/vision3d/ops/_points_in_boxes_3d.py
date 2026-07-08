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

    Supports all rotation formats including full 9-DOF (yaw, pitch, roll).

    Args:
        points: Point cloud coordinates ``[N, 3+C]``. Only the first 3
            columns (x, y, z) are used.
        boxes: 3D bounding boxes ``[M, K]`` where K depends on format.
        format: Format of the bounding boxes.

    Returns:
        Boolean tensor ``[N, M]`` where entry ``(i, j)`` is True if
        point ``i`` is inside box ``j``.
    """
    if points.dtype != boxes.dtype:
        dtype = torch.promote_types(points.dtype, boxes.dtype)
        points = points.to(dtype)
        boxes = boxes.to(dtype)
    centers, half_dims, rot = extract_box3d_params(boxes, format)
    return _points_in_rotated_boxes(points[:, :3], centers, half_dims, rot)


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


def _build_rotation_matrix(
    yaw: Tensor,
    pitch: Tensor | None = None,
    roll: Tensor | None = None,
) -> Tensor:
    """Build ``[M, 3, 3]`` rotation matrices from Tait-Bryan ZY'X'' angles.

    When pitch and roll are None, builds a yaw-only Rz rotation
    (avoids unnecessary trig for the common case).

    Args:
        yaw: Yaw angles ``[M]`` in radians.
        pitch: Pitch angles ``[M]`` in radians, or None.
        roll: Roll angles ``[M]`` in radians, or None.

    Returns:
        Rotation matrices ``[M, 3, 3]``.
    """
    m = yaw.shape[0]
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    if pitch is None or roll is None:
        # Yaw-only: Rz(yaw)
        rot = torch.zeros(m, 3, 3, dtype=yaw.dtype, device=yaw.device)
        rot[:, 0, 0] = cy
        rot[:, 0, 1] = -sy
        rot[:, 1, 0] = sy
        rot[:, 1, 1] = cy
        rot[:, 2, 2] = 1.0
        return rot

    # Full Tait-Bryan ZY'X'': R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)

    rot = torch.empty(m, 3, 3, dtype=yaw.dtype, device=yaw.device)
    rot[:, 0, 0] = cy * cp
    rot[:, 0, 1] = cy * sp * sr - sy * cr
    rot[:, 0, 2] = cy * sp * cr + sy * sr
    rot[:, 1, 0] = sy * cp
    rot[:, 1, 1] = sy * sp * sr + cy * cr
    rot[:, 1, 2] = sy * sp * cr - cy * sr
    rot[:, 2, 0] = -sp
    rot[:, 2, 1] = cp * sr
    rot[:, 2, 2] = cp * cr
    return rot


def extract_box3d_params(
    boxes: Tensor, format: BoundingBox3DFormat
) -> tuple[Tensor, Tensor, Tensor]:
    """Decompose 3D boxes into centers, half-dimensions, and rotation matrices.

    Supports all box formats including full 9-DOF (yaw, pitch, roll); formats
    without rotation yield identity rotation matrices.

    Args:
        boxes: 3D bounding boxes ``[M, K]`` where ``K`` depends on ``format``.
        format: Format of the bounding boxes.

    Returns:
        ``(centers, half_dims, rot)`` where ``centers`` and ``half_dims`` are
        ``[M, 3]`` and ``rot`` is ``[M, 3, 3]``. Each ``rot`` maps local to
        world coordinates (``world = rot @ local``, matching ``box3d_corners``),
        so a box's world-frame axes are the *columns* of ``rot``. Transpose
        for the inverse (world-to-local) mapping.

    Raises:
        ValueError: If ``format`` is not a supported format.
    """
    if format is BoundingBox3DFormat.XYZXYZ:
        mins = boxes[:, :3]
        maxs = boxes[:, 3:6]
        centers = (mins + maxs) / 2
        half_dims = (maxs - mins) / 2
        yaw = torch.zeros(boxes.shape[0], dtype=boxes.dtype, device=boxes.device)
        rot = _build_rotation_matrix(yaw)
    elif format is BoundingBox3DFormat.XYZLWH:
        centers = boxes[:, :3]
        half_dims = boxes[:, 3:6] / 2
        yaw = torch.zeros(boxes.shape[0], dtype=boxes.dtype, device=boxes.device)
        rot = _build_rotation_matrix(yaw)
    elif format is BoundingBox3DFormat.XYZLWHY:
        centers = boxes[:, :3]
        half_dims = boxes[:, 3:6] / 2
        rot = _build_rotation_matrix(boxes[:, 6])
    elif format is BoundingBox3DFormat.XYZLWHYPR:
        centers = boxes[:, :3]
        half_dims = boxes[:, 3:6] / 2
        rot = _build_rotation_matrix(boxes[:, 6], boxes[:, 7], boxes[:, 8])
    else:
        msg = f"Unsupported format: {format}"
        raise ValueError(msg)
    return centers, half_dims, rot


def _points_in_rotated_boxes(
    xyz: Tensor, centers: Tensor, half_dims: Tensor, rot: Tensor
) -> Tensor:
    """Check if points are inside arbitrarily rotated boxes.

    Args:
        xyz: Point positions ``[N, 3]``.
        centers: Box centers ``[M, 3]``.
        half_dims: Box half-extents ``[M, 3]`` (half_l, half_w, half_h).
        rot: Rotation matrices ``[M, 3, 3]``.

    Returns:
        Boolean ``[N, M]``.
    """
    # Relative positions: [N, 1, 3] - [1, M, 3] -> [N, M, 3]
    rel = xyz.unsqueeze(1) - centers.unsqueeze(0)

    # Transpose to map the world-frame offset into the box's local frame
    # (see extract_box3d_params).
    rot = rot.transpose(-1, -2)
    # rel: [N, M, 3], R^T: [M, 3, 3] -> local: [N, M, 3]
    local = torch.einsum("nmk,mjk->nmj", rel, rot)

    return (local.abs() <= half_dims.unsqueeze(0)).all(dim=-1)
