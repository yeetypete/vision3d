"""Conversion primitives between 3D bounding box formats."""

import torch
from torch import Tensor


def _xyzxyz_to_xyzwhd(boxes: Tensor) -> Tensor:
    x1, y1, z1, x2, y2, z2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cz = (z1 + z2) / 2
    w = x2 - x1
    h = y2 - y1
    d = z2 - z1
    return torch.stack((cx, cy, cz, w, h, d), dim=-1)


def _xyzwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
    cx, cy, cz, w, h, d = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    z1 = cz - d / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    z2 = cz + d / 2
    return torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)
