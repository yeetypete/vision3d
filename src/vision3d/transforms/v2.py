"""Mirror of :mod:`torchvision.transforms.v2` with geometric safety guarantees.

Swap

.. code-block:: python

    from torchvision.transforms import v2 as T

for

.. code-block:: python

    from vision3d.transforms import v2 as T

to make every transform refuse inputs whose presence would silently break
the geometric consistency of a 3D scene.

Each mirrored class subclasses its torchvision counterpart and adds a
``_safe_for`` declaration that places it into one of two categories.

Photometric transforms (e.g. :class:`ColorJitter`,
:class:`GaussianBlur`) and image-geometric transforms whose companion
:class:`~vision3d.tensors.CameraIntrinsics` kernel is already registered
(e.g. :class:`Resize`, :class:`CenterCrop`) accept the full set of
vision3d-aware TVTensors.

Image-geometric transforms without a matching 3D update
(e.g. :class:`RandomHorizontalFlip`, :class:`RandomVerticalFlip`)
accept plain :class:`torchvision.tv_tensors.Image` and
:class:`~torchvision.tv_tensors.Mask` only and refuse any 3D-aware
input, since flipping the image would leave lidar, boxes, extrinsics,
and intrinsics inconsistent.
"""

from typing import Any

import torchvision.transforms.v2 as _T
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import Image, Mask, TVTensor

from ._transform import ALL_VISION3D_TVTENSORS, _check_safety

_ALL: frozenset[type[TVTensor]] = ALL_VISION3D_TVTENSORS | frozenset({Image, Mask})
# Transforms that change the image plane but have no matching update for
# the 3D side (no intrinsics/extrinsics kernel for flip or rotation).
_IMAGE_ONLY: frozenset[type[TVTensor]] = frozenset({Image, Mask})


def _wrap[T: _T.Transform](
    parent_cls: type[T], safe_for: frozenset[type[TVTensor]]
) -> type[T]:
    """Build a safety-checking subclass of a torchvision v2 transform.

    Returns:
        Subclass of ``parent_cls`` whose ``forward`` validates inputs
        against ``safe_for`` before delegating to the parent.
    """

    def forward(self: T, *inputs: Any) -> Any:
        flat, _ = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        _check_safety(safe_for, flat, parent_cls.__name__)
        return parent_cls.forward(self, *inputs)

    sub = type(
        parent_cls.__name__,
        (parent_cls,),
        {"forward": forward, "_safe_for": safe_for},
    )
    sub.__module__ = __name__
    sub.__qualname__ = parent_cls.__name__
    sub.__doc__ = parent_cls.__doc__
    return sub


# Photometric transforms (fully safe for any TVTensor, since they don't affect geometry)
ColorJitter = _wrap(_T.ColorJitter, _ALL)
GaussianBlur = _wrap(_T.GaussianBlur, _ALL)
Normalize = _wrap(_T.Normalize, _ALL)
RandomGrayscale = _wrap(_T.RandomGrayscale, _ALL)

# Geometric transforms with registered intrinsics kernels (accept any TVTensor)
Resize = _wrap(_T.Resize, _ALL)
CenterCrop = _wrap(_T.CenterCrop, _ALL)
Pad = _wrap(_T.Pad, _ALL)
RandomResizedCrop = _wrap(_T.RandomResizedCrop, _ALL)

# Geometric transforms without registered intrinsics kernels (refuse 3D-aware inputs)
RandomHorizontalFlip = _wrap(_T.RandomHorizontalFlip, _IMAGE_ONLY)
RandomVerticalFlip = _wrap(_T.RandomVerticalFlip, _IMAGE_ONLY)


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
