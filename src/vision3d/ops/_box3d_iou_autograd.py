"""Autograd wiring for ``vision3d::iou_box3d``.

Uses :func:`torch.library.register_autograd`, the recommended path for
stable-ABI custom ops (see the PyTorch tutorial "Custom C++ and CUDA
Operators"). The actual gradient math lives in the C++/CUDA
``iou_box3d_backward`` op; this module only saves the forward state
needed by the backward and routes the call.
"""

from typing import Any

import torch
from torch import Tensor


def _setup_context(ctx: Any, inputs: tuple[Tensor, Tensor], output: Any) -> None:
    """Save the forward state required by ``_backward``.

    Args:
        ctx: Autograd context provided by :func:`torch.library.register_autograd`.
        inputs: ``(boxes1, boxes2)`` from the forward call.
        output: ``(vol, iou, face_area, face_area_centroid)`` from the forward.
    """
    boxes1, boxes2 = inputs
    vol, _iou, face_area, face_area_centroid = output
    ctx.save_for_backward(boxes1, boxes2, vol, face_area, face_area_centroid)


def _backward(
    ctx: Any,
    grad_vol: Tensor,
    grad_iou: Tensor,
    _grad_face_area: Tensor,
    _grad_face_area_centroid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute grads on boxes1/boxes2 by delegating to the C++ backward op.

    ``face_area`` and ``face_area_centroid`` are internal state of the
    forward; their grad slots are ignored (they should not appear in user
    loss expressions).

    Args:
        ctx: Autograd context populated by :func:`_setup_context`.
        grad_vol: Upstream gradient on the ``vol`` output.
        grad_iou: Upstream gradient on the ``iou`` output.
        _grad_face_area: Ignored; ``face_area`` is internal state.
        _grad_face_area_centroid: Ignored; ``face_area_centroid`` is internal.

    Returns:
        ``(grad_boxes1, grad_boxes2)``.
    """
    boxes1, boxes2, vol, face_area, face_area_centroid = ctx.saved_tensors
    grad_boxes1, grad_boxes2 = torch.ops.vision3d.iou_box3d_backward(
        boxes1,
        boxes2,
        vol,
        face_area,
        face_area_centroid,
        grad_vol.contiguous(),
        grad_iou.contiguous(),
    )
    return grad_boxes1, grad_boxes2


torch.library.register_autograd(
    "vision3d::iou_box3d",
    _backward,
    setup_context=_setup_context,
)
