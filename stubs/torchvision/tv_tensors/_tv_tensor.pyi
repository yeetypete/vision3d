"""Shape-aware stub for :class:`torchvision.tv_tensors.TVTensor`."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import torch
from shape_extensions import IntTuple
from torch import Tensor

class TVTensor[Shape: IntTuple = IntTuple](Tensor[Shape]):
    @staticmethod
    def _to_tensor(
        data: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Tensor: ...
    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Tensor: ...
    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: tuple[type[Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Tensor: ...
    def _make_repr(self, **kwargs: Any) -> str: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
