import math

import torch

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D


def make_bounding_boxes_3d(
    *,
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZWHDYPR,
    num_boxes: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> BoundingBoxes3D:
    """Generate random, valid 3D bounding boxes for testing.

    Produces non-degenerate boxes with positive dimensions and rotation angles
    in (-pi, +pi].

    Returns:
        BoundingBoxes3D with shape ``[num_boxes, K]`` where K depends on format.

    Raises:
        ValueError: If ``format`` is not a supported :class:`BoundingBox3DFormat`.
    """
    dtype = dtype or torch.float32

    # Random positive dimensions
    w = torch.rand(num_boxes) * 10 + 1
    h = torch.rand(num_boxes) * 10 + 1
    d = torch.rand(num_boxes) * 10 + 1

    # Random center positions
    cx = torch.rand(num_boxes) * 100 - 50
    cy = torch.rand(num_boxes) * 100 - 50
    cz = torch.rand(num_boxes) * 100 - 50

    # Random rotation angles in (-pi, +pi]
    yaw = torch.rand(num_boxes) * 2 * math.pi - math.pi
    pitch = torch.rand(num_boxes) * 2 * math.pi - math.pi
    roll = torch.rand(num_boxes) * 2 * math.pi - math.pi

    if format is BoundingBox3DFormat.XYZXYZ:
        x1 = cx - w / 2
        y1 = cy - h / 2
        z1 = cz - d / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        z2 = cz + d / 2
        parts = (x1, y1, z1, x2, y2, z2)
    elif format is BoundingBox3DFormat.XYZWHD:
        parts = (cx, cy, cz, w, h, d)
    elif format is BoundingBox3DFormat.XYZWHDY:
        parts = (cx, cy, cz, w, h, d, yaw)
    elif format is BoundingBox3DFormat.XYZWHDYPR:
        parts = (cx, cy, cz, w, h, d, yaw, pitch, roll)
    else:
        raise ValueError(f"Format {format} is not supported")

    data = torch.stack(parts, dim=-1).to(dtype=dtype, device=device)
    return BoundingBoxes3D(data, format=format)
