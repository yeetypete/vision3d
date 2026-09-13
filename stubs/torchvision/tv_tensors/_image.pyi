"""Shape-aware stub for :class:`torchvision.tv_tensors.Image`."""

from typing import Any, Self

import torch
from shape_extensions import IntTuple

from ._tv_tensor import TVTensor

class Image[Shape: IntTuple = IntTuple](TVTensor[Shape]):
    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self: ...
    def __repr__(self, *, tensor_contents: Any = None) -> str: ...
