"""Functional kernels for :class:`~vision3d.tensors.GridMask3D`."""

from collections.abc import Sequence

from torch import Tensor
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as _F
from torchvision.transforms.v2.functional import register_kernel as _register_kernel
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._utils import _FillTypeJIT

from vision3d.tensors import GridMask3D, wrap

from ._geometry import flip_3d, rotate_3d, scale_3d, translate_3d
from ._registry import _get_kernel, register_kernel

_SUPPORTED_INTERPOLATIONS = (InterpolationMode.NEAREST, InterpolationMode.BILINEAR)


def _check_grid_interpolation(
    interpolation: str | InterpolationMode | int,
) -> InterpolationMode:
    interpolation = _check_interpolation(interpolation)
    if interpolation not in _SUPPORTED_INTERPOLATIONS:
        supported = [mode.value for mode in _SUPPORTED_INTERPOLATIONS]
        raise ValueError(
            f"Grids only support interpolation modes {supported}, "
            f"got '{interpolation.value}'."
        )
    return interpolation


# TODO: Add implementations for the grid kernels and resample_grid_mask_3d.
def _not_implemented(operation: str) -> NotImplementedError:
    return NotImplementedError(f"{operation} is not implemented for grids yet.")


@register_kernel(flip_3d, GridMask3D, tv_tensor_wrapper=False)
def _flip_3d_grid_dispatch(inpt: GridMask3D, *, axis: str) -> GridMask3D:
    raise _not_implemented("flip_3d")


@register_kernel(translate_3d, GridMask3D, tv_tensor_wrapper=False)
def _translate_3d_grid_dispatch(inpt: GridMask3D, *, offset: Tensor) -> GridMask3D:
    raise _not_implemented("translate_3d")


@register_kernel(rotate_3d, GridMask3D, tv_tensor_wrapper=False)
def _rotate_3d_grid_dispatch(
    inpt: GridMask3D, *, rotation_matrix: Tensor
) -> GridMask3D:
    raise _not_implemented("rotate_3d")


@register_kernel(scale_3d, GridMask3D, tv_tensor_wrapper=False)
def _scale_3d_grid_dispatch(inpt: GridMask3D, *, factor: float) -> GridMask3D:
    raise _not_implemented("scale_3d")


# Otherwise torchvision flips would pass grids through unchanged.
@_register_kernel(_F.horizontal_flip, GridMask3D)
def _horizontal_flip_grid(inpt: GridMask3D) -> GridMask3D:
    raise _not_implemented("horizontal_flip")


@_register_kernel(_F.vertical_flip, GridMask3D)
def _vertical_flip_grid(inpt: GridMask3D) -> GridMask3D:
    raise _not_implemented("vertical_flip")


def resample_grid(
    inpt: Tensor,
    *,
    affine: Tensor,
    size: Sequence[int],
    interpolation: str | InterpolationMode | int = InterpolationMode.NEAREST,
    fill: _FillTypeJIT = None,
) -> Tensor:
    """Resample a grid into a target grid layout.

    Dispatcher entry point.

    Args:
        inpt: Input tensor.
        affine: ``[4, 4]`` index-to-frame transform of the target grid.
        size: Spatial size of the target grid, ``(H, W)`` or
            ``(D, H, W)``.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.v2.InterpolationMode`.
            Accepted string values are ``"nearest"`` and ``"bilinear"``
            (requires torchvision 0.28 or later).
            Default is ``"nearest"``.
            The corresponding ``InterpolationMode`` enum values and Pillow integer
            constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill: Value for target cells that fall outside the input grid,
            either a single value or one value per channel. ``None`` means 0.

    Returns:
        Resampled tensor.
    """
    kernel = _get_kernel(resample_grid, type(inpt))
    return kernel(
        inpt, affine=affine, size=size, interpolation=interpolation, fill=fill
    )


def resample_grid_mask_3d(
    mask: Tensor,
    *,
    input_affine: Tensor,
    affine: Tensor,
    size: Sequence[int],
    interpolation: str | InterpolationMode | int = InterpolationMode.NEAREST,
    fill: _FillTypeJIT = None,
) -> Tensor:
    """Resample a label grid into a target layout.

    Args:
        mask: Grid data ``[C, H, W]`` or ``[C, D, H, W]``.
        input_affine: ``[4, 4]`` index-to-frame transform of ``mask``.
        affine: ``[4, 4]`` index-to-frame transform of the target grid.
        size: Spatial size of the target grid, ``(H, W)`` or
            ``(D, H, W)``.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.v2.InterpolationMode`.
            Accepted string values are ``"nearest"`` and ``"bilinear"``
            (requires torchvision 0.28 or later).
            Default is ``"nearest"``.
            The corresponding ``InterpolationMode`` enum values and Pillow integer
            constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
            Bilinear requires a floating point ``mask``.
        fill: Value for target cells that fall outside ``mask``, either a
            single value or one value per channel. ``None`` means 0.

    Raises:
        ValueError: If ``interpolation`` is bilinear and ``mask`` is not
            floating point.
        NotImplementedError: Always, until resampling is implemented.
    """
    interpolation = _check_grid_interpolation(interpolation)
    if interpolation == InterpolationMode.BILINEAR and not mask.is_floating_point():
        raise ValueError(
            f"Bilinear interpolation requires a floating point grid, got {mask.dtype}."
        )
    # TODO: Raise a ValueError if len(size) != mask.ndim - 1.
    msg = "resample_grid is not implemented for grids yet."
    raise NotImplementedError(msg)


@register_kernel(resample_grid, GridMask3D, tv_tensor_wrapper=False)
def _resample_grid_mask_3d_dispatch(
    inpt: GridMask3D,
    *,
    affine: Tensor,
    size: Sequence[int],
    interpolation: str | InterpolationMode | int = InterpolationMode.NEAREST,
    fill: _FillTypeJIT = None,
) -> GridMask3D:
    output = resample_grid_mask_3d(
        inpt.as_subclass(Tensor),
        input_affine=inpt.affine,
        affine=affine,
        size=size,
        interpolation=interpolation,
        fill=fill,
    )
    return wrap(output, like=inpt, affine=affine)
