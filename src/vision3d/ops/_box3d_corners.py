"""Compute 3D bounding box corners."""

import torch
from torch import Tensor

from vision3d.tensors import BoundingBox3DFormat

from ._points_in_boxes_3d import _extract_box_params


def box3d_corners(
    boxes: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    r"""Compute the 8 world-space corners of 3D bounding boxes.

    Supports all rotation formats including full 9-DOF (yaw, pitch, roll).

    Corner ordering::

        4 -------- 5       z  x
        |\         |\      |  /
        | 7 -------| 6     | /
        | |        | |     |/
        0 |--------1 |     +------ y
         \|         \|
          3 -------- 2

    Bottom face (z-): {0, 1, 2, 3}.  Top face (z+): {4, 5, 6, 7}.

    Args:
        boxes: 3D bounding boxes ``[N, K]``.
        format: Format of the bounding boxes.

    Returns:
        Corner coordinates ``[N, 8, 3]``.
    """
    centers, half_dims, rot = _extract_box_params(boxes, format)

    # 8 sign combinations for (x, y, z) offsets
    # Order: (-x,-y,-z), (+x,-y,-z), (+x,+y,-z), (-x,+y,-z),
    #        (-x,-y,+z), (+x,-y,+z), (+x,+y,+z), (-x,+y,+z)
    signs = torch.tensor(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=boxes.dtype,
        device=boxes.device,
    )  # [8, 3]

    # Local corners before rotation: [N, 8, 3]
    local = half_dims.unsqueeze(1) * signs.unsqueeze(0)

    # Rotate by full rotation matrix: [N, 3, 3] @ [N, 3, 8] -> [N, 3, 8] -> [N, 8, 3]
    rotated = torch.einsum("nij,nkj->nki", rot, local)

    return rotated + centers.unsqueeze(1)
