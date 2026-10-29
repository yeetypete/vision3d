"""Functional kernels for :class:`~vision3d.tensors.GridMask3D`."""

from collections.abc import Sequence

from torch import Tensor
from torchvision.transforms.v2.functional._utils import _FillTypeJIT

from vision3d.tensors import GridMask3D, wrap

from ._registry import register_kernel


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
