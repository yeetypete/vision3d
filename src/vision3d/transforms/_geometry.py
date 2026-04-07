"""Geometric 3D transform classes."""

from typing import Any, override

import torch

from ._transform import RandomTransform
from .functional._geometry import flip_3d, translate_3d


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


class RandomTranslate3D(RandomTransform):
    """Translate inputs by a random 3D offset with probability ``p``.

    Dispatches to type-specific kernels for :class:`PointCloud3D`,
    :class:`BoundingBoxes3D`, and :class:`CameraExtrinsics`.

    Args:
        translation_range: Maximum translation per axis. Either a single
            float (symmetric range ``[-v, v]`` for all axes) or a tuple of
            three floats ``(tx, ty, tz)`` for per-axis ranges.
        p: Probability of applying the translation. Default: ``0.5``.
    """

    def __init__(
        self,
        translation_range: float | tuple[float, float, float] = 0.5,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        if isinstance(translation_range, (int, float)):
            translation_range = (
                float(translation_range),
                float(translation_range),
                float(translation_range),
            )
        self.translation_range = translation_range

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample a random offset.

        Returns:
            Dict with ``"offset"`` key containing a ``[3]`` tensor.
        """
        tx, ty, tz = self.translation_range
        offset = torch.tensor(
            [
                (torch.rand(1).item() * 2 - 1) * tx,
                (torch.rand(1).item() * 2 - 1) * ty,
                (torch.rand(1).item() * 2 - 1) * tz,
            ]
        )
        return {"offset": offset}

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the translation to a single input.

        Returns:
            Translated input.
        """
        return self._call_kernel(translate_3d, inpt, offset=params["offset"])
