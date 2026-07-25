"""Meta (fake tensor) registrations for vision3d custom ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor


@torch.library.register_fake("vision3d::iou_box3d")
def _meta_iou_box3d[N: IntVar, M: IntVar](
    boxes1: Tensor[[N, 8, 3]], boxes2: Tensor[[M, 8, 3]]
) -> tuple[Tensor[[N, M]], Tensor[[N, M]]]:
    torch._check(
        boxes1.dim() == 3 and boxes1.size(1) == 8 and boxes1.size(2) == 3,
        lambda: f"boxes1 must be (N, 8, 3), got {tuple(boxes1.shape)}",
    )
    torch._check(
        boxes2.dim() == 3 and boxes2.size(1) == 8 and boxes2.size(2) == 3,
        lambda: f"boxes2 must be (M, 8, 3), got {tuple(boxes2.shape)}",
    )
    n = boxes1.size(0)
    m = boxes2.size(0)
    # float32 regardless of input dtype (matches PyTorch3D's kernel).
    vol = boxes1.new_empty((n, m), dtype=torch.float32)
    iou = boxes1.new_empty((n, m), dtype=torch.float32)
    return vol, iou


@torch.library.register_fake("vision3d::voxelize")
def _meta_voxelize[N: IntVar, C: IntVar, MPV: IntVar](
    points: Tensor[[N, C]],
    point_cloud_range: list[float],
    voxel_size: list[float],
    max_points_per_voxel: Int[MPV],
    max_voxels: int | None,
) -> tuple[Tensor[[int, MPV, C]], Tensor[[int, 3]], Tensor[[int]]]:
    torch._check(
        points.dim() == 2,
        lambda: f"points must be 2D [N, C], got {tuple(points.shape)}",
    )
    torch._check(len(point_cloud_range) == 6, lambda: "point_cloud_range size != 6")
    torch._check(len(voxel_size) == 3, lambda: "voxel_size size != 3")
    c = points.size(1)
    voxels = points.new_empty((0, max_points_per_voxel, c))
    coords = points.new_empty((0, 3), dtype=torch.int64)
    num_points = points.new_empty((0,), dtype=torch.int64)
    return voxels, coords, num_points
