"""Transforms for dense 3D grids."""

from collections.abc import Sequence
from typing import Any, override

from torchvision.transforms.v2._utils import _get_fill, _setup_fill_arg
from torchvision.transforms.v2.functional._utils import _FillType

from vision3d.tensors import GridMask3D
from vision3d.tensors._grid_mask_3d import _as_affine

from ._transform import Transform
from .functional._grid import resample_grid


class ResampleGrid(Transform):
    """Resample grids into a target grid layout.

    3D transforms only update the ``affine`` of a
    :class:`~vision3d.tensors.GridMask3D`. Use this transform, typically as the
    last transform, to resample the cells onto the target ``affine`` and
    ``size``. Values are resampled with nearest-neighbor interpolation. Other
    inputs pass through unchanged.

    Args:
        affine: ``[4, 4]`` index-to-frame transform of the target grid.
        size: Spatial size of the target grid, ``(H, W)`` for a planar grid
            or ``(D, H, W)`` for a volume.
        fill: Value for target cells that fall outside the input grid. A
            tuple gives one value per channel. Fill value can also be a
            dictionary mapping data type to the fill value, e.g.
            ``fill={GridMask3D: 255}``. Default: ``0``.
    """

    _transformed_types = (GridMask3D,)

    def __init__(
        self,
        affine: Any,
        size: Sequence[int],
        fill: _FillType | dict[type | str, _FillType] = 0,
    ) -> None:
        super().__init__()
        if len(size) not in (2, 3):
            msg = f"size must have 2 or 3 elements, got {len(size)}."
            raise ValueError(msg)
        if not all(isinstance(s, int) and s > 0 for s in size):
            msg = f"size must contain positive integers, got {tuple(size)}."
            raise ValueError(msg)
        self.affine = _as_affine(affine)
        self.size = tuple(size)
        self.fill = fill
        self._fill = _setup_fill_arg(fill)

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Resample a single grid.

        Returns:
            The grid in the target layout.
        """
        return self._call_kernel(
            resample_grid,
            inpt,
            affine=self.affine,
            size=self.size,
            fill=_get_fill(self._fill, type(inpt)),
        )
