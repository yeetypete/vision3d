"""Bird's eye view (BEV) IoU for 3D bounding boxes."""

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch import Tensor

from ._box3d_corners import box3d_corners

if TYPE_CHECKING:
    from vision3d.tensors import BoundingBox3DFormat


def box3d_iou_bev(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Compute rotated BEV IoU between two sets of 3D boxes.

    Projects boxes onto the XY plane and computes intersection-over-union
    using :func:`cv2.intersectConvexConvex`.  Supports all rotation
    formats including full 9-DOF.

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        IoU matrix ``[N, M]`` with values in ``[0, 1]``.
    """
    polys1 = _to_bev_polygons(boxes1, format)
    polys2 = _to_bev_polygons(boxes2, format)
    areas1 = [cv2.contourArea(p) for p in polys1]
    areas2 = [cv2.contourArea(p) for p in polys2]

    n, m = len(polys1), len(polys2)
    iou = torch.zeros(n, m, dtype=boxes1.dtype, device=boxes1.device)

    for i in range(n):
        for j in range(m):
            inter_area, _pts = cv2.intersectConvexConvex(polys1[i], polys2[j])
            if inter_area <= 0:
                continue
            union = areas1[i] + areas2[j] - inter_area
            if union > 1e-8:
                iou[i, j] = inter_area / union

    return iou


def box3d_overlap_bev(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Check BEV overlap between two sets of 3D boxes.

    Supports all rotation formats including full 9-DOF.

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        Boolean matrix ``[N, M]`` where True indicates overlap.
    """
    polys1 = _to_bev_polygons(boxes1, format)
    polys2 = _to_bev_polygons(boxes2, format)

    n, m = len(polys1), len(polys2)
    overlap = torch.zeros(n, m, dtype=torch.bool, device=boxes1.device)

    for i in range(n):
        for j in range(m):
            inter_area, _pts = cv2.intersectConvexConvex(polys1[i], polys2[j])
            if inter_area > 0:
                overlap[i, j] = True

    return overlap


def _to_bev_polygons(boxes: Tensor, format: BoundingBox3DFormat) -> list[np.ndarray]:
    """Convert boxes to BEV convex hull polygons for OpenCV.

    Projects the 8 3D corners onto the XY plane and computes the convex
    hull, producing a polygon that correctly represents the BEV footprint
    even for pitched/rolled boxes.

    Args:
        boxes: ``[M, K]`` boxes.
        format: Box format.

    Returns:
        List of ``M`` numpy arrays, each ``[V, 1, 2]`` (OpenCV contour
        format) with float32 dtype.
    """
    if boxes.shape[0] == 0:
        return []
    corners = box3d_corners(boxes, format)  # [M, 8, 3]
    bev_xy = corners[:, :, :2].detach().cpu().numpy()  # [M, 8, 2]

    polys: list[np.ndarray] = []
    for i in range(bev_xy.shape[0]):
        pts = bev_xy[i].astype(np.float32)
        hull = cv2.convexHull(pts)  # [V, 1, 2]
        polys.append(hull)
    return polys
