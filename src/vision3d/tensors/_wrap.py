from typing import Any

from torch import Tensor
from torchvision.tv_tensors import TVTensor

from ._bounding_boxes_3d import BoundingBoxes3D


def wrap(
    wrappee: Tensor,
    *,
    like: TVTensor,
    **kwargs: Any,
) -> TVTensor:
    """Convert a :class:`~torch.Tensor` into the same TVTensor subclass as ``like``.

    If ``like`` is a :class:`~vision3d.tensors.BoundingBoxes3D`, the ``format``
    of ``like`` is assigned to ``wrappee`` unless overridden via ``kwargs``.

    Args:
        wrappee: The tensor to convert.
        like: The reference. ``wrappee`` will be converted into the same
            subclass as ``like``.
        kwargs: Can contain ``"format"`` if ``like`` is a
            :class:`~vision3d.tensors.BoundingBoxes3D`. Ignored otherwise.

    Returns:
        A TVTensor of the same subclass as ``like``.
    """
    if isinstance(like, BoundingBoxes3D):
        return type(like)._wrap(
            wrappee,
            format=kwargs.get("format", like.format),
        )
    else:
        return wrappee.as_subclass(type(like))
