"""Conversion primitives between 3D bounding box formats."""

import torch
from torch import Tensor


def _xyzxyz_to_xyzlwh(boxes: Tensor) -> Tensor:
    x1, y1, z1, x2, y2, z2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cz = (z1 + z2) / 2
    l = x2 - x1
    w = y2 - y1
    h = z2 - z1
    return torch.stack((cx, cy, cz, l, w, h), dim=-1)


def _xyzlwh_to_xyzxyz(boxes: Tensor) -> Tensor:
    cx, cy, cz, l, w, h = boxes.unbind(-1)
    x1 = cx - l / 2
    y1 = cy - w / 2
    z1 = cz - h / 2
    x2 = cx + l / 2
    y2 = cy + w / 2
    z2 = cz + h / 2
    return torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)
