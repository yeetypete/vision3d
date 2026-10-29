"""Functional kernels for :class:`~vision3d.tensors.GridMask3D`."""

from collections.abc import Sequence

from torch import Tensor
from torchvision.transforms.v2 import functional as _F
from torchvision.transforms.v2.functional import register_kernel as _register_kernel
from torchvision.transforms.v2.functional._utils import _FillTypeJIT

from vision3d.tensors import GridMask3D, wrap
from vision3d.tensors._grid_mask_3d import _Grid

from ._geometry import flip_3d, rotate_3d, scale_3d, translate_3d
from ._registry import register_kernel


# TODO: Add implementations for the grid kernels and resample_grid_mask_3d.
# TODO: Register a to_dtype kernel for _Grid.
def _not_implemented(operation: str) -> NotImplementedError:
    return NotImplementedError(f"{operation} is not implemented for grids yet.")


# Because 3D transforms only update the affine transform, which every grid type
# carries, these kernels are registered on the shared base.


@register_kernel(flip_3d, _Grid, tv_tensor_wrapper=False)
def _flip_3d_grid_dispatch(inpt: _Grid, *, axis: str) -> _Grid:
    raise _not_implemented("flip_3d")


@register_kernel(translate_3d, _Grid, tv_tensor_wrapper=False)
def _translate_3d_grid_dispatch(inpt: _Grid, *, offset: Tensor) -> _Grid:
    raise _not_implemented("translate_3d")


@register_kernel(rotate_3d, _Grid, tv_tensor_wrapper=False)
def _rotate_3d_grid_dispatch(inpt: _Grid, *, rotation_matrix: Tensor) -> _Grid:
    raise _not_implemented("rotate_3d")


@register_kernel(scale_3d, _Grid, tv_tensor_wrapper=False)
def _scale_3d_grid_dispatch(inpt: _Grid, *, factor: float) -> _Grid:
    raise _not_implemented("scale_3d")


# Otherwise torchvision flips would pass grids through unchanged.


@_register_kernel(_F.horizontal_flip, _Grid)
def _horizontal_flip_grid(inpt: _Grid) -> _Grid:
    raise _not_implemented("horizontal_flip")


@_register_kernel(_F.vertical_flip, _Grid)
def _vertical_flip_grid(inpt: _Grid) -> _Grid:
    raise _not_implemented("vertical_flip")


def resample_grid(
    inpt: Tensor,
    *,
    affine: Tensor,
    size: Sequence[int],
    fill: _FillTypeJIT = None,
) -> Tensor:
    """Resample a grid into a target grid layout.

    Dispatcher entry point. :class:`~vision3d.tensors.GridMask3D` uses
    nearest-neighbor interpolation.

    Args:
        inpt: Input tensor.
        affine: ``[4, 4]`` index-to-frame transform of the target grid.
        size: Spatial size of the target grid, ``(H, W)`` or
            ``(D, H, W)``.
        fill: Value for target cells that fall outside the input grid,
            either a single value or one value per channel. ``None`` means 0.

    Returns:
        Resampled tensor.
    """
    return inpt


def resample_grid_mask_3d(
    mask: Tensor,
    *,
    input_affine: Tensor,
    affine: Tensor,
    size: Sequence[int],
    fill: _FillTypeJIT = None,
) -> Tensor:
    """Resample a label grid into a target layout with nearest-neighbor interpolation.

    Args:
        mask: Grid data ``[C, H, W]`` or ``[C, D, H, W]``.
        input_affine: ``[4, 4]`` index-to-frame transform of ``mask``.
        affine: ``[4, 4]`` index-to-frame transform of the target grid.
        size: Spatial size of the target grid, ``(H, W)`` or
            ``(D, H, W)``.
        fill: Value for target cells that fall outside ``mask``, either a
            single value or one value per channel. ``None`` means 0.

    Raises:
        NotImplementedError: Always, until resampling is implemented.
    """
    msg = "resample_grid is not implemented for grids yet."
    raise NotImplementedError(msg)


@register_kernel(resample_grid, GridMask3D, tv_tensor_wrapper=False)
def _resample_grid_mask_3d_dispatch(
    inpt: GridMask3D,
    *,
    affine: Tensor,
    size: Sequence[int],
    fill: _FillTypeJIT = None,
) -> GridMask3D:
    output = resample_grid_mask_3d(
        inpt.as_subclass(Tensor),
        input_affine=inpt.affine,
        affine=affine,
        size=size,
        fill=fill,
    )
    return wrap(output, like=inpt, affine=affine)
