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
    centers, half_dims, rot = extract_box3d_params(boxes, format)
    return _points_in_rotated_boxes(points[:, :3], centers, half_dims, rot)


def points_in_boxes_3d_indices(
    points: Tensor,
    boxes: Tensor,
    format: BoundingBox3DFormat,
    *,
    box_mask: Tensor | None = None,
) -> Tensor:
    """Return per-point box assignment.

    If a point is inside multiple boxes, the first (lowest index) box wins.

    Args:
        points: Point cloud coordinates ``[N, 3+C]``.
        boxes: 3D bounding boxes ``[M, K]``.
        format: Format of the bounding boxes.
        box_mask: Optional boolean ``[M]`` mask of eligible boxes. When
            given, points are assigned only among boxes that are ``True``;
            a point that falls solely in masked-out boxes is treated as
            belonging to no box (``-1``).

    Returns:
        Integer tensor ``[N]`` with the index of the box each point
        belongs to, or ``-1`` if the point is not in any (eligible) box.
    """
    mask = points_in_boxes_3d(points, boxes, format)  # [N, M]
    if box_mask is not None:
        mask = mask & box_mask.unsqueeze(0)
    return _first_true_index(mask)


def _first_true_index(mask: Tensor) -> Tensor:
    """Return the first True column per row of a boolean mask.

    Args:
        mask: Boolean tensor ``[N, M]``.

    Returns:
        Long tensor ``[N]`` giving the lowest column index that is True in
        each row, or ``-1`` for all-False rows.
    """
    n = mask.shape[0]
    if mask.shape[1] == 0:
        return torch.full((n,), -1, dtype=torch.long, device=mask.device)
    # First True along dim=1 (argmax returns the first max); all-False rows
    # would spuriously map to column 0, so overwrite them with -1.
    first = mask.to(torch.uint8).argmax(dim=1)
    first[~mask.any(dim=1)] = -1
    return first


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
        ``[M, 3]`` and ``rot`` is ``[M, 3, 3]``.

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

    # Rotate into box local frame by R^T (inverse rotation)
    # rel: [N, M, 3], rot^T: [M, 3, 3] -> local: [N, M, 3]
    # Einstein: local_j = rel_k * R_jk  (R^T has j,k swapped)
    local = torch.einsum("nmk,mjk->nmj", rel, rot)

    return (local.abs() <= half_dims.unsqueeze(0)).all(dim=-1)
