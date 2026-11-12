"""Functional kernels for :class:`~vision3d.tensors.GridMask3D`."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as _F
from torchvision.transforms.v2.functional import register_kernel as _register_kernel
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._utils import _FillTypeJIT

from vision3d.tensors import GridMask3D, wrap
from vision3d.tensors._grid_mask_3d import _as_affine

from ._geometry import flip_3d, rotate_3d, scale_3d, translate_3d
from ._registry import _get_kernel, register_kernel

_SUPPORTED_INTERPOLATIONS = (InterpolationMode.NEAREST, InterpolationMode.BILINEAR)
# In cells. Far above the float64 rounding error of the computed indices, far
# below anything that matters for where a cell center is.
_BOUNDARY_TOLERANCE = 1e-6


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


# TODO: Add implementations for the grid kernels.
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

    Each target cell takes the value of ``mask`` at the target cell's center.
    Nearest-neighbor interpolation picks the input cell that contains the
    center. A center on the boundary between two cells picks the upper one,
    i.e. the one with the higher index. Bilinear interpolation blends the
    nearest input cell centers (trilinear for volumes). Near the edge of
    ``mask`` it uses the edge cells instead of blending in ``fill``.

    Target cells whose center falls outside ``mask`` get ``fill``. For a
    planar grid, this includes centers above or below its single layer of
    cells.

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

    Returns:
        Resampled grid ``[C, *size]`` with the dtype and device of ``mask``.

    Raises:
        ValueError: If ``interpolation`` is bilinear and ``mask`` is not
            floating point, if ``mask`` is neither ``[C, H, W]`` nor
            ``[C, D, H, W]``, if ``size`` does not match the number of
            spatial dimensions of ``mask``, or if ``fill`` has neither one
            value nor one value per channel.
    """
    interpolation = _check_grid_interpolation(interpolation)
    if interpolation == InterpolationMode.BILINEAR and not mask.is_floating_point():
        raise ValueError(
            f"Bilinear interpolation requires a floating point grid, got {mask.dtype}."
        )
    if mask.ndim not in (3, 4):
        raise ValueError(
            f"Expected a grid [C, H, W] or [C, D, H, W], got shape {tuple(mask.shape)}."
        )
    if len(size) != mask.ndim - 1:
        raise ValueError(
            f"Expected size with {mask.ndim - 1} elements for a grid of shape "
            f"{tuple(mask.shape)}, got {tuple(size)}."
        )
    fill_values = _fill_values(fill, mask)

    planar = mask.ndim == 3
    if planar:
        mask = mask.unsqueeze(1)
        size = (1, *size)

    index = _input_index(input_affine, affine, size, device=mask.device)
    num_cells = torch.tensor(mask.shape[1:], device=mask.device).view(3, 1, 1, 1)
    # Cell k covers indices [k - 0.5, k + 0.5). A center on a boundary often
    # comes out a tiny bit off it, since cell sizes like 0.4 m aren't exact in
    # binary. Move the boundaries down by a tolerance, so such a center always
    # reads the upper cell instead of the side rounding happened to pick.
    edge = 0.5 + _BOUNDARY_TOLERANCE
    inside = ((index >= -edge) & (index < num_cells - edge)).all(dim=0)

    if interpolation == InterpolationMode.NEAREST:
        # Round in place to save memory, since index isn't needed afterwards.
        # Outside cells get fill below, so only keep their indices in range.
        cell = index.add_(edge).floor_().clamp_(min=0).clamp_(max=num_cells - 1)
        cell = cell.long()
        output = mask[:, cell[0], cell[1], cell[2]]
    else:
        output = _sample_bilinear(mask, index)

    output = torch.where(inside, output, fill_values)
    return output.squeeze(1) if planar else output


def _fill_values(fill: _FillTypeJIT, mask: Tensor) -> Tensor:
    if fill is None:
        fill = [0.0]
    elif not isinstance(fill, (list, tuple)):
        fill = [fill]
    num_channels = mask.shape[0]
    if len(fill) not in (1, num_channels):
        raise ValueError(
            f"Expected fill with 1 or {num_channels} values for a grid with "
            f"{num_channels} channels, got {len(fill)}."
        )
    values = torch.tensor(fill, dtype=torch.float64, device=mask.device)
    if not mask.is_floating_point():
        # Round like torchvision instead of truncating. Going through int64
        # keeps the cast defined for values out of range, so -1 becomes 255
        # for uint8 whether it was given as an int or a float.
        values = values.round().long()
    return values.to(mask.dtype).view(-1, 1, 1, 1)


def _input_index(
    input_affine: Tensor,
    affine: Tensor,
    size: Sequence[int],
    *,
    device: torch.device,
) -> Tensor:
    # Map each target cell index into the frame with the target affine, then
    # back into the input grid with the inverse input affine. Use float64 so
    # the rounding error stays well below the boundary tolerance, also for
    # indices in the thousands.
    transform = torch.linalg.solve(_as_affine(input_affine), _as_affine(affine))
    transform = transform[:3].to(device)
    # transform @ [d, h, w, 1] is the last column plus each other column times
    # its index. Add the columns one axis at a time, so the full grid of
    # target indices is never built.
    index = transform[:, 3].view(3, 1, 1, 1).repeat(1, *size)
    for axis, n in enumerate(size):
        steps = torch.arange(n, dtype=torch.float64, device=device)
        shape = [1, 1, 1]
        shape[axis] = n
        index += transform[:, axis].view(3, 1, 1, 1) * steps.view(shape)
    return index


def _sample_bilinear(mask: Tensor, index: Tensor) -> Tensor:
    # grid_sample takes coordinates in [-1, 1] ordered (w, h, d). With
    # align_corners=False, -1 and 1 are the outer edges of the edge cells, so
    # index i maps to (2 * i + 1) / n - 1. This also works for a single layer,
    # where align_corners=True would divide by zero.
    num_cells = torch.tensor(mask.shape[1:], device=index.device).view(3, 1, 1, 1)
    grid = ((2 * index + 1) / num_cells - 1).flip(0).permute(1, 2, 3, 0)
    # grid_sample needs the grid in the dtype of the input. Half precision
    # coordinates can't address single cells in large grids, so sample in at
    # least float32.
    dtype = torch.promote_types(mask.dtype, torch.float32)
    output = grid_sample(
        mask.unsqueeze(0).to(dtype),
        grid.unsqueeze(0).to(dtype),
        mode="bilinear",
        # Clamp to the edge cells instead of blending in zeros. Cells outside
        # the grid get fill afterwards.
        padding_mode="border",
        align_corners=False,
    )
    return output.squeeze(0).to(mask.dtype)


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
