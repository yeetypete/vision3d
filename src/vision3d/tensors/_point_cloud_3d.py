from typing import TYPE_CHECKING, override

import torch
from torchvision.tv_tensors import TVTensor

if TYPE_CHECKING:
    from typing import Any, Self


class PointCloud3D(TVTensor):
    """:class:`torch.Tensor` subclass for 3D point clouds with shape ``[N, 3+C]``.

    The first 3 columns are ``(x, y, z)`` coordinates. Additional columns are
    per-point features (e.g. intensity, color, normals).

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type. If omitted, will be
            inferred from ``data``.
        device (torch.device, optional): Desired device. If omitted and ``data``
            is a :class:`torch.Tensor`, the device is taken from it. Otherwise,
            the point cloud is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations.
            If omitted and ``data`` is a :class:`torch.Tensor`, the value is
            taken from it. Otherwise, defaults to ``False``.
    """

    # This stub exists so pyrefly sees the correct constructor signature instead of torch.Tensor.__init__.
    def __init__(
        self,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None: ...

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D tensor [N, 3+C], got {tensor.ndim}D")
        if tensor.shape[-1] < 3:
            raise ValueError(
                f"Expected at least 3 columns (x, y, z), got {tensor.shape[-1]}"
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(
                f"Point clouds require floating point tensors, got {tensor.dtype}."
            )
        return tensor.as_subclass(cls)

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()
