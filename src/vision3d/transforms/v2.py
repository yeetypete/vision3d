"""Mirror of ``torchvision.transforms.v2`` with geometric safety guarantees.

Swap

.. code-block:: python

    from torchvision.transforms import v2 as T

for

.. code-block:: python

    from vision3d.transforms import v2 as T

to make every transform refuse inputs whose presence would silently
break the geometric consistency of a 3D scene.

Photometric transforms (e.g. :class:`ColorJitter`,
:class:`GaussianBlur`) and image-geometric transforms (e.g.
:class:`Resize`, :class:`CenterCrop`) are re-exported unchanged:
their default pass-through behaviour leaves vision3d's TVTensors
untouched, which is safe.

Image-geometric transforms without a matching 3D update (e.g.
:class:`RandomHorizontalFlip`, :class:`RandomVerticalFlip`) override
``check_inputs`` to reject any vision3d-aware TVTensor: flipping the
image would leave the lidar, boxes, extrinsics, and intrinsics
inconsistent.
"""

from typing import Any

import torchvision.transforms.v2 as _T

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

_3D_AWARE_TVTENSORS = (
    PointCloud3D,
    BoundingBoxes3D,
    CameraImages,
    CameraExtrinsics,
    CameraIntrinsics,
)


class _Refuse3DAwareMixin:
    """Mixin that rejects vision3d-aware TVTensor inputs in ``check_inputs``."""

    def check_inputs(self, flat_inputs: list[Any]) -> None:
        """Raise if any vision3d-aware TVTensor is present.

        Raises:
            TypeError: If any input is a vision3d TVTensor.
        """
        offenders = sorted(
            {
                type(inpt).__name__
                for inpt in flat_inputs
                if isinstance(inpt, _3D_AWARE_TVTENSORS)
            }
        )
        if offenders:
            msg = (
                f"{type(self).__name__} cannot operate on samples that "
                f"contain vision3d TVTensors {offenders}: applying it "
                f"without coordinated changes to the 3D scene would break "
                f"geometric consistency."
            )
            raise TypeError(msg)


class RandomHorizontalFlip(_Refuse3DAwareMixin, _T.RandomHorizontalFlip):  # noqa: D101
    __doc__ = _T.RandomHorizontalFlip.__doc__


class RandomVerticalFlip(_Refuse3DAwareMixin, _T.RandomVerticalFlip):  # noqa: D101
    __doc__ = _T.RandomVerticalFlip.__doc__


# Photometric and image-geometric transforms re-exported unchanged: their
# default behaviour passes any vision3d TVTensor through untouched, which
# is what we want.
CenterCrop = _T.CenterCrop
ColorJitter = _T.ColorJitter
GaussianBlur = _T.GaussianBlur
Normalize = _T.Normalize
Pad = _T.Pad
RandomGrayscale = _T.RandomGrayscale
RandomResizedCrop = _T.RandomResizedCrop
Resize = _T.Resize


__all__ = [
    "CenterCrop",
    "ColorJitter",
    "GaussianBlur",
    "Normalize",
    "Pad",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "RandomVerticalFlip",
    "Resize",
]
