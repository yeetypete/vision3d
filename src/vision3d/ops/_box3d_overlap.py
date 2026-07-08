"""3D oriented bounding box overlap using the Separating Axis Theorem."""

import torch
from torch import Tensor

from vision3d.tensors import BoundingBox3DFormat

from ._points_in_boxes_3d import extract_box3d_params


def box3d_overlap(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Check 3D overlap between two sets of oriented bounding boxes.

    Uses the Separating Axis Theorem (SAT) with 15 potential separating
    axes (3 face normals per box + 9 edge cross products).

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        Boolean matrix ``[N, M]`` where True indicates overlap.
    """
    if boxes1.dtype != boxes2.dtype:
        dtype = torch.promote_types(boxes1.dtype, boxes2.dtype)
        boxes1 = boxes1.to(dtype)
        boxes2 = boxes2.to(dtype)

    centers1, half1, rot1 = extract_box3d_params(boxes1, format)
    centers2, half2, rot2 = extract_box3d_params(boxes2, format)

    # Transpose so box axes are rows (see extract_box3d_params), as the
    # projections below expect.
    rot1 = rot1.transpose(-1, -2)
    rot2 = rot2.transpose(-1, -2)

    # Pairwise center difference: [N, M, 3]
    diff = centers2.unsqueeze(0) - centers1.unsqueeze(1)

    # Projections of diff onto each box's world-frame axes (rows of the
    # transposed rot are the box axes).
    # dot1[n, m, i] = diff[n,m] . axis_i(box1)
    dot1 = torch.einsum("nmk,nik->nmi", diff, rot1)  # [N, M, 3]
    # dot2[n, m, j] = diff[n,m] . axis_j(box2)
    dot2 = torch.einsum("nmk,mjk->nmj", diff, rot2)  # [N, M, 3]
    # c[n, m, i, j] = axis_i(box1) . axis_j(box2)
    c = torch.einsum("nik,mjk->nmij", rot1, rot2)  # [N, M, 3, 3]
    abs_c = c.abs()

    overlap = torch.ones(
        centers1.shape[0], centers2.shape[0], dtype=torch.bool, device=boxes1.device
    )

    # Face normals of box1 (axes i=0,1,2)
    for i in range(3):
        d = dot1[:, :, i].abs()
        r1 = half1[:, i].unsqueeze(1)
        r2 = (abs_c[:, :, i, :] * half2.unsqueeze(0)).sum(dim=-1)
        overlap &= d <= r1 + r2

    # Face normals of box2 (axes j=0,1,2)
    for j in range(3):
        d = dot2[:, :, j].abs()
        r1 = (abs_c[:, :, :, j] * half1.unsqueeze(1)).sum(dim=-1)
        r2 = half2[:, j].unsqueeze(0)
        overlap &= d <= r1 + r2

    # Edge cross products: rot1[:,:,i] x rot2[:,:,j]
    # For axis a_i x b_j, the projections simplify using the c matrix.
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            # d = |diff . (a_i x b_j)| = |dot1_i2 * c_i1j - dot1_i1 * c_i2j|
            #   (using triple product expansion)
            d = (
                dot1[:, :, i1] * c[:, :, i2, j] - dot1[:, :, i2] * c[:, :, i1, j]
            ).abs()
            r1 = (
                half1[:, i1].unsqueeze(1) * abs_c[:, :, i2, j]
                + half1[:, i2].unsqueeze(1) * abs_c[:, :, i1, j]
            )
            r2 = (
                half2[:, j1].unsqueeze(0) * abs_c[:, :, i, j2]
                + half2[:, j2].unsqueeze(0) * abs_c[:, :, i, j1]
            )
            overlap &= d <= r1 + r2

    return overlap
