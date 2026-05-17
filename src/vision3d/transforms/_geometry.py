"""Geometric 3D transform classes."""

import math
from typing import Any, override

import torch
from torchvision.transforms.v2 import has_any

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

from ._transform import _RandomApplyTransform
from .functional._geometry import (
    _rotation_matrix,
    flip_3d,
    rotate_3d,
    scale_3d,
    translate_3d,
)


class RandomFlip3D(_RandomApplyTransform):
    """Flip inputs along a 3D axis with probability ``p``.

    Operates on :class:`~vision3d.tensors.PointCloud3D` and
    :class:`~vision3d.tensors.BoundingBoxes3D`. Camera inputs (images,
    extrinsics, intrinsics) are rejected: flipping the 3D scene without
    coordinated changes to the camera side would break geometric
    consistency.

    Args:
        axis: Axis to flip along. One of ``"x"``, ``"y"``, ``"z"``.
        p: Probability of applying the flip. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D, BoundingBoxes3D)

    def __init__(self, axis: str = "x", p: float = 0.5) -> None:
        super().__init__(p=p)
        if axis not in ("x", "y", "z"):
            msg = f"axis must be 'x', 'y', or 'z', got '{axis}'"
            raise ValueError(msg)
        self.axis = axis

    @override
    def check_inputs(self, flat_inputs: list[Any]) -> None:
        """Reject camera inputs.

        Raises:
            TypeError: If any camera tensor is present.
        """
        if has_any(flat_inputs, CameraImages, CameraExtrinsics, CameraIntrinsics):
            msg = (
                f"{type(self).__name__} cannot operate on samples that contain "
                f"camera tensors: flipping the 3D scene without coordinated "
                f"changes to the cameras would break geometric consistency."
            )
            raise TypeError(msg)

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the flip to a single input.

        Returns:
            Flipped input.
        """
        return self._call_kernel(flip_3d, inpt, axis=self.axis)


class RandomTranslate3D(_RandomApplyTransform):
    """Translate inputs by a random 3D offset with probability ``p``.

    Operates on :class:`~vision3d.tensors.PointCloud3D`,
    :class:`~vision3d.tensors.BoundingBoxes3D`, and
    :class:`~vision3d.tensors.CameraExtrinsics`.

    Args:
        translation_range: Maximum translation per axis. Either a single
            float (symmetric range ``[-v, v]`` for all axes) or a tuple of
            three floats ``(tx, ty, tz)`` for per-axis ranges.
        p: Probability of applying the translation. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D, BoundingBoxes3D, CameraExtrinsics)

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


class RandomRotate3D(_RandomApplyTransform):
    """Rotate inputs around an axis by a random angle with probability ``p``.

    Operates on :class:`~vision3d.tensors.PointCloud3D`,
    :class:`~vision3d.tensors.BoundingBoxes3D`, and
    :class:`~vision3d.tensors.CameraExtrinsics`.

    Args:
        angle_range: Maximum rotation angle in radians. Sampled uniformly
            from ``[-angle_range, angle_range]``. Default: ``pi/4``.
        axis: Rotation axis as a 3-tuple. Default: ``(0, 0, 1)`` (Z-up).
        p: Probability of applying the rotation. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D, BoundingBoxes3D, CameraExtrinsics)

    def __init__(
        self,
        angle_range: float = math.pi / 4,
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.angle_range = angle_range
        self.axis = torch.tensor(axis, dtype=torch.float32)

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample a random rotation matrix.

        Returns:
            Dict with ``"rotation_matrix"`` key containing a ``[3, 3]`` tensor.
        """
        angle = (torch.rand(1).item() * 2 - 1) * self.angle_range
        R = _rotation_matrix(self.axis, angle)
        return {"rotation_matrix": R}

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the rotation to a single input.

        Returns:
            Rotated input.
        """
        return self._call_kernel(
            rotate_3d, inpt, rotation_matrix=params["rotation_matrix"]
        )


class RandomScale3D(_RandomApplyTransform):
    """Scale inputs by a random uniform factor with probability ``p``.

    Operates on :class:`~vision3d.tensors.PointCloud3D`,
    :class:`~vision3d.tensors.BoundingBoxes3D`, and
    :class:`~vision3d.tensors.CameraExtrinsics`.

    Args:
        scale_range: Scale factor range as ``(min, max)``.
            Default: ``(0.95, 1.05)``.
        p: Probability of applying the scaling. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D, BoundingBoxes3D, CameraExtrinsics)

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.95, 1.05),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        if scale_range[0] <= 0 or scale_range[1] <= 0:
            msg = "scale_range values must be positive."
            raise ValueError(msg)
        self.scale_range = scale_range

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample a random scale factor.

        Returns:
            Dict with ``"factor"`` key.
        """
        lo, hi = self.scale_range
        factor = lo + torch.rand(1).item() * (hi - lo)
        return {"factor": factor}

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the scaling to a single input.

        Returns:
            Scaled input.
        """
        return self._call_kernel(scale_3d, inpt, factor=params["factor"])
