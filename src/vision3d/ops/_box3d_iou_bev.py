"""Bird's eye view (BEV) IoU for 3D bounding boxes."""

import torch
from torch import Tensor

from vision3d.tensors import BoundingBox3DFormat


def box3d_iou_bev(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Compute rotated BEV IoU between two sets of 3D boxes.

    Projects boxes onto the XY plane as rotated rectangles and computes
    intersection-over-union using polygon clipping.

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        IoU matrix ``[N, M]`` with values in ``[0, 1]``.
    """
    corners1 = _to_bev_corners(boxes1, format)  # [N, 4, 2]
    corners2 = _to_bev_corners(boxes2, format)  # [M, 4, 2]

    areas1 = _polygon_area(corners1)  # [N]
    areas2 = _polygon_area(corners2)  # [M]

    n, m = corners1.shape[0], corners2.shape[0]
    iou = torch.zeros(n, m, dtype=boxes1.dtype, device=boxes1.device)

    for i in range(n):
        for j in range(m):
            inter = _polygon_intersection_area(corners1[i], corners2[j])
            union = areas1[i] + areas2[j] - inter
            if union > 1e-8:
                iou[i, j] = inter / union

    return iou


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


def _to_bev_corners(boxes: Tensor, format: BoundingBox3DFormat) -> Tensor:
    """Convert boxes to BEV corner points ``[N, 4, 2]``.

    Corners are ordered counterclockwise starting from the front-right
    corner (when yaw=0).

    Returns:
        ``[N, 4, 2]`` tensor of XY corner coordinates.

    Raises:
        ValueError: If ``format`` is not supported.
    """
    if format is BoundingBox3DFormat.XYZXYZ:
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 3], boxes[:, 4]
        return torch.stack(
            [
                torch.stack([x2, y2], dim=-1),
                torch.stack([x1, y2], dim=-1),
                torch.stack([x1, y1], dim=-1),
                torch.stack([x2, y1], dim=-1),
            ],
            dim=1,
        )

    # All center+size formats
    cx, cy = boxes[:, 0], boxes[:, 1]
    hl = boxes[:, 3] / 2  # half-length (X)
    hw = boxes[:, 4] / 2  # half-width (Y)

    if format in (BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR):
        yaw = boxes[:, 6]
    elif format is BoundingBox3DFormat.XYZLWH:
        yaw = torch.zeros(boxes.shape[0], dtype=boxes.dtype, device=boxes.device)
    else:
        msg = f"Unsupported format: {format}"
        raise ValueError(msg)

    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    # Local corners: (±hl, ±hw) relative to center
    # Order: front-right, front-left, back-left, back-right
    local_x = torch.stack([hl, -hl, -hl, hl], dim=-1)  # [N, 4]
    local_y = torch.stack([hw, hw, -hw, -hw], dim=-1)  # [N, 4]

    # Rotate to scene frame
    rx = local_x * cos_y.unsqueeze(-1) - local_y * sin_y.unsqueeze(-1)
    ry = local_x * sin_y.unsqueeze(-1) + local_y * cos_y.unsqueeze(-1)

    # Translate to center
    corners_x = cx.unsqueeze(-1) + rx  # [N, 4]
    corners_y = cy.unsqueeze(-1) + ry  # [N, 4]

    return torch.stack([corners_x, corners_y], dim=-1)  # [N, 4, 2]


def _polygon_area(corners: Tensor) -> Tensor:
    """Compute area of convex polygons using the shoelace formula.

    Args:
        corners: ``[N, P, 2]`` polygon vertices.

    Returns:
        ``[N]`` areas.
    """
    x = corners[..., 0]
    y = corners[..., 1]
    # Shoelace formula
    rolled_x = torch.roll(x, -1, dims=-1)
    rolled_y = torch.roll(y, -1, dims=-1)
    return (0.5 * (x * rolled_y - rolled_x * y).sum(dim=-1)).abs()


def _polygon_intersection_area(poly1: Tensor, poly2: Tensor) -> float:
    """Compute intersection area of two convex polygons.

    Uses the Sutherland-Hodgman clipping algorithm.

    Args:
        poly1: ``[P1, 2]`` first polygon vertices (counterclockwise).
        poly2: ``[P2, 2]`` second polygon vertices (counterclockwise).

    Returns:
        Intersection area as a float.
    """
    clipped = poly1.tolist()

    for i in range(len(poly2)):
        if len(clipped) == 0:
            return 0.0

        edge_start = poly2[i].tolist()
        edge_end = poly2[(i + 1) % len(poly2)].tolist()

        new_clipped: list[list[float]] = []
        for j in range(len(clipped)):
            curr = clipped[j]
            prev = clipped[j - 1]

            curr_inside = _cross_2d(edge_start, edge_end, curr) >= 0
            prev_inside = _cross_2d(edge_start, edge_end, prev) >= 0

            if curr_inside:
                if not prev_inside:
                    intersection = _line_intersection(prev, curr, edge_start, edge_end)
                    if intersection is not None:
                        new_clipped.append(intersection)
                new_clipped.append(curr)
            elif prev_inside:
                intersection = _line_intersection(prev, curr, edge_start, edge_end)
                if intersection is not None:
                    new_clipped.append(intersection)

        clipped = new_clipped

    if len(clipped) < 3:
        return 0.0

    # Shoelace formula on the clipped polygon
    area = 0.0
    for i in range(len(clipped)):
        j = (i + 1) % len(clipped)
        area += clipped[i][0] * clipped[j][1]
        area -= clipped[j][0] * clipped[i][1]
    return abs(area) / 2.0


def _cross_2d(o: list[float], a: list[float], b: list[float]) -> float:
    """2D cross product of vectors OA and OB.

    Returns:
        Positive if B is left of OA (counterclockwise), negative if right.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _line_intersection(
    p1: list[float],
    p2: list[float],
    p3: list[float],
    p4: list[float],
) -> list[float] | None:
    """Compute intersection point of line segments p1-p2 and p3-p4.

    Returns:
        ``[x, y]`` intersection point, or None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
