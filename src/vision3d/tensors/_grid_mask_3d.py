"""Dense label grid TVTensor placed in 3D space by an affine transform."""

from collections.abc import Mapping, Sequence
from typing import Any, Self, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import TVTensor

_HOMOGENEOUS_ROW = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)


def _as_affine(affine: Any) -> Tensor:
    # affine is metadata so we want to keep it on the cpu.
    affine = torch.as_tensor(affine, dtype=torch.float64, device="cpu").clone()
    if affine.shape != (4, 4):
        raise ValueError(f"Expected a [4, 4] affine, got shape {tuple(affine.shape)}")
    if not torch.equal(affine[3], _HOMOGENEOUS_ROW):
        raise ValueError(
            f"Expected the last affine row to be [0, 0, 0, 1], got {affine[3].tolist()}"
        )
    return affine


class _Grid(TVTensor):
    """Shared base for dense grids that carry an index-to-frame affine transform."""

    affine: Tensor

    @classmethod
    def _wrap(cls, tensor: Tensor, *, affine: Tensor, check_dims: bool = True) -> Self:
        if check_dims and tensor.ndim not in (3, 4):
            raise ValueError(
                f"Expected a 3D tensor [C, H, W] or a 4D tensor [C, D, H, W], "
                f"got {tensor.ndim}D"
            )
        grid = tensor.as_subclass(cls)
        grid.affine = affine
        return grid

    @classmethod
    @override
    # ``tv_tensors.wrap`` dispatches on ``type(like)``, so ``like`` is always a
    # ``_Grid`` here.
    # pyrefly: ignore[bad-override]
    def wrap(cls, tensor: Tensor, like: "_Grid", **kwargs: Any) -> Self:
        affine = kwargs.get("affine")
        affine = like.affine if affine is None else _as_affine(affine)
        return cls._wrap(tensor, affine=affine)

    @classmethod
    @override
    def _wrap_output(
        cls,
        output: Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Tensor:
        # Metadata is lost after __torch_function__ calls. Restore it by taking
        # the affine from the first grid in the args.
        flat_params, _ = tree_flatten(
            tuple(args) + (tuple(kwargs.values()) if kwargs else ())
        )
        affine = next(x for x in flat_params if isinstance(x, _Grid)).affine

        if isinstance(output, Tensor) and not isinstance(output, _Grid):
            output = cls._wrap(output, affine=affine, check_dims=False)
        elif isinstance(output, (tuple, list)):
            # This branch exists for chunk() and unbind()
            output = type(output)(
                cls._wrap(part, affine=affine, check_dims=False) for part in output
            )
        return output

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(affine=self.affine)


class GridMask3D(_Grid):
    """:class:`~torch.Tensor` subclass for label grids with shape ``[C, H, W]`` or ``[C, D, H, W]``.

    1. A 3D tensor is a planar grid (e.g. a top-down semantic or
       traversability map).
    2. A 4D tensor is a volume (e.g. semantic occupancy).

    Any dtype is accepted (e.g. int64 for class indices, bool for validity
    masks, or float for soft labels). Values are resampled with
    nearest-neighbor interpolation.

    ``affine`` maps homogeneous cell indices to the sample's 3D frame. Its
    columns follow the spatial axis order of the data, ``(d, h, w)``, and
    index ``(0, 0, 0)`` is the center of the first cell. A planar grid
    ``[C, H, W]`` is treated as ``[C, 1, H, W]``. The ``d`` column sets the
    direction and thickness of its single layer of cells.

    Instead of resampling the data, 3D transforms compose their rotation,
    translation, scaling, or reflection into ``affine`` and leave cells
    untouched. Use :class:`~vision3d.transforms.ResampleGrid`, typically as
    the last transform, to resample the cells into the grid layout the model
    expects.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        affine: ``[4, 4]`` index-to-frame transform. Stored as ``float64``
            on the CPU.
        dtype (torch.dtype, optional): Desired data type. If omitted, will be
            inferred from ``data``.
        device (torch.device, optional): Desired device. If omitted and ``data``
            is a :class:`~torch.Tensor`, the device is taken from it. Otherwise,
            the mask is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations.
            If omitted and ``data`` is a :class:`~torch.Tensor`, the value is
            taken from it. Otherwise, defaults to ``False``.
    """

    # This stub exists so pyrefly sees the correct constructor signature instead of torch.Tensor.__init__.
    def __init__(
        self,
        data: Any,
        *,
        affine: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None: ...

    def __new__(
        cls,
        data: Any,
        *,
        affine: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        return cls._wrap(tensor, affine=_as_affine(affine))
