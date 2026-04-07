"""Geometric 3D transform classes."""

from typing import Any, override

from ._transform import RandomTransform
from .functional._geometry import flip_3d


class RandomFlip3D(RandomTransform):
    """Flip inputs along a 3D axis with probability ``p``.

    Dispatches to type-specific kernels for :class:`BoundingBoxes3D` and
    :class:`PointCloud3D`. Camera data (images, intrinsics, extrinsics) passes
    through unchanged.

    Args:
        axis: Axis to flip along. One of ``"x"``, ``"y"``, ``"z"``.
        p: Probability of applying the flip. Default: ``0.5``.
    """

    def __init__(self, axis: str = "x", p: float = 0.5) -> None:
        super().__init__(p=p)
        if axis not in ("x", "y", "z"):
            msg = f"axis must be 'x', 'y', or 'z', got '{axis}'"
            raise ValueError(msg)
        self.axis = axis

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the flip to a single input.

        Returns:
            Flipped input.
        """
        return self._call_kernel(flip_3d, inpt, axis=self.axis)
