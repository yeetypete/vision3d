from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Self, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import TVTensor


class Cylinder3DFormat(Enum):
    """Coordinate format of a 3D cylinder.

    Cylinders are upright (their axis is parallel to **Z**) and rotationally
    symmetric about that axis, so no heading is stored.

    Dimension convention:

    * ``r`` (radius) = extent in the **XY** plane
    * ``h`` (height) = extent along **Z** (vertical axis, ``dz``)

    Available formats are:

    * ``XYZRH``: center position, radius, and height;
      cx, cy, cz (center), r, h. 5 values.
    """

    XYZRH = "XYZRH"


class Cylinders3D(TVTensor):
    """:class:`~torch.Tensor` subclass for upright 3D cylinders with shape ``[N, K]``.

    Where ``N`` is the number of cylinders and ``K`` depends on the format
    (5 for ``XYZRH``). Cylinders are aligned with the **Z** axis.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format (Cylinder3DFormat, str): Format of the 3D cylinder.
        dtype (torch.dtype, optional): Desired data type of the cylinder. If
            omitted, will be inferred from ``data``.
        device (torch.device, optional): Desired device of the cylinder. If
            omitted and ``data`` is a :class:`~torch.Tensor`, the device is taken
            from it. Otherwise, the cylinder is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations
            on the cylinder. If omitted and ``data`` is a :class:`~torch.Tensor`,
            the value is taken from it. Otherwise, defaults to ``False``.
    """

    format: Cylinder3DFormat

    # This stub exists so pyrefly sees the correct constructor signature instead of torch.Tensor.__init__.
    def __init__(
        self,
        data: Any,
        *,
        format: Cylinder3DFormat | str,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None: ...

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        format: Cylinder3DFormat | str,
        check_dims: bool = True,
    ) -> Self:
        if check_dims:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                raise ValueError(f"Expected a 1D or 2D tensor, got {tensor.ndim}D")

        if isinstance(format, str):
            format = Cylinder3DFormat[format.upper()]

        cylinders = tensor.as_subclass(cls)
        cylinders.format = format
        return cylinders

    @classmethod
    def wrap(cls, tensor: Tensor, like: "Cylinders3D", **kwargs: Any) -> Self:
        return cls._wrap(tensor, format=kwargs.get("format", like.format))

    def __new__(
        cls,
        data: Any,
        *,
        format: Cylinder3DFormat | str,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        if isinstance(format, str):
            format = Cylinder3DFormat[format.upper()]
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        if not torch.is_floating_point(tensor):
            raise ValueError(
                f"3D cylinders require floating point tensors, got {tensor.dtype}."
            )
        return cls._wrap(tensor, format=format)

    @classmethod
    @override
    def _wrap_output(
        cls,
        output: Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        # Metadata is lost after __torch_function__ calls. Restore it by taking
        # the format from the first Cylinders3D in the args. This is correct
        # in most cases; when it isn't (e.g. mixing formats in one operation),
        # it's likely a mis-use we don't guard against.
        flat_params, _ = tree_flatten(
            tuple(args) + (tuple(kwargs.values()) if kwargs else ())
        )
        first_cyl_from_args = next(x for x in flat_params if isinstance(x, Cylinders3D))
        format = first_cyl_from_args.format

        if isinstance(output, Tensor) and not isinstance(output, Cylinders3D):
            output = cls._wrap(output, format=format, check_dims=False)
        elif isinstance(output, (tuple, list)):
            # This branch exists for chunk() and unbind()
            output = type(output)(
                cls._wrap(part, format=format, check_dims=False) for part in output
            )
        return output

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(format=self.format)
