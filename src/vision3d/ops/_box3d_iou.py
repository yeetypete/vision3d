"""3D oriented bounding box IoU.

Python wrapper around PyTorch3D's ``box3d_overlap``. C++ and
CUDA sources live under ``src/vision3d/ops/csrc/iou_box3d/``.
"""

import torch
from torch import Tensor

from vision3d import _extension  # noqa: F401  # loads ``_C`` into torch.ops
from vision3d.ops import _meta_registrations  # noqa: F401  # registers fake kernels
from vision3d.tensors import BoundingBox3DFormat

from ._box3d_corners import box3d_corners


@torch.no_grad()
def box3d_iou(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Compute the pairwise intersection-over-union of 3D ``boxes1`` and ``boxes2``.

    ``iou = vol / (vol1 + vol2 - vol)``, where ``vol`` is the volume of
    the intersecting convex polyhedron and ``vol1``, ``vol2`` are the
    volumes of the two input boxes.

    The same algorithm handles every supported box format — including
    full 9-DOF orientation (``XYZLWHYPR``) — because the clipping step
    operates on the 8 box corners regardless of how they were produced.

    Note: This function is not differentiable.

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        ``[N, M]`` matrix of IoU values in ``[0, 1]``.
    """
    corners1 = box3d_corners(boxes1, format).to(torch.float32)  # [N, 8, 3]
    corners2 = box3d_corners(boxes2, format).to(torch.float32)  # [M, 8, 3]
    _, iou, _, _ = torch.ops.vision3d.iou_box3d(corners1, corners2)
    return iou
