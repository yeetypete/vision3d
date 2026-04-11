"""Meta (fake tensor) registrations for vision3d custom ops.

Registers shape-only implementations of ops exposed through the
``vision3d`` TORCH_LIBRARY namespace. PyTorch's tracer (``torch.compile``,
``torch.export``, ``torch.fx``) calls these during symbolic tracing instead
of the real CPU/CUDA kernels.
"""

import torch


@torch.library.register_fake("vision3d::iou_box3d")
def _meta_iou_box3d(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
