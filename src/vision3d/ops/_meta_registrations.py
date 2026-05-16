"""Meta (fake tensor) registrations for vision3d custom ops."""

import torch
from torch import Tensor


@torch.library.register_fake("vision3d::iou_box3d")
def _meta_iou_box3d(
    boxes1: Tensor, boxes2: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    # Per-input-plane state used by the analytic backward (planes 0..5 from
    # boxes1, planes 6..11 from boxes2). face_area_centroid is area-weighted
    # (i.e. area * centroid) to keep the zero-area case divide-free.
    face_area = boxes1.new_empty((n, m, 12), dtype=torch.float32)
    face_area_centroid = boxes1.new_empty((n, m, 12, 3), dtype=torch.float32)
    return vol, iou, face_area, face_area_centroid


@torch.library.register_fake("vision3d::iou_box3d_backward")
def _meta_iou_box3d_backward(
    boxes1: Tensor,
    boxes2: Tensor,
    vol: Tensor,
    face_area: Tensor,
    face_area_centroid: Tensor,
    grad_vol: Tensor,
    grad_iou: Tensor,
) -> tuple[Tensor, Tensor]:
    torch._check(
        boxes1.dim() == 3 and boxes1.size(1) == 8 and boxes1.size(2) == 3,
        lambda: f"boxes1 must be (N, 8, 3), got {tuple(boxes1.shape)}",
    )
    torch._check(
        boxes2.dim() == 3 and boxes2.size(1) == 8 and boxes2.size(2) == 3,
        lambda: f"boxes2 must be (M, 8, 3), got {tuple(boxes2.shape)}",
    )
    grad_boxes1 = boxes1.new_empty((boxes1.size(0), 8, 3), dtype=torch.float32)
    grad_boxes2 = boxes2.new_empty((boxes2.size(0), 8, 3), dtype=torch.float32)
    return grad_boxes1, grad_boxes2
