"""Mirror of ``torchvision.transforms.v2`` with geometric safety guarantees.

Swap

.. code-block:: python

    from torchvision.transforms import v2 as T

for

.. code-block:: python

    from vision3d.transforms import v2 as T

to make every transform that would silently break the geometric
consistency of a 3D scene refuse vision3d-aware TVTensor inputs with a
:class:`TypeError` instead.

The module forwards every public name from ``torchvision.transforms.v2``
unchanged, except for the transforms listed in the module-private
``_REFUSED`` set. Those are subclassed with a refusal mixin: calling one
on a sample containing any vision3d TVTensor
(:class:`~vision3d.tensors.PointCloud3D`,
:class:`~vision3d.tensors.BoundingBoxes3D`,
:class:`~vision3d.tensors.CameraImages`,
:class:`~vision3d.tensors.CameraExtrinsics`, or
:class:`~vision3d.tensors.CameraIntrinsics`) raises
:class:`TypeError`. They still work on plain
:class:`torchvision.tv_tensors.Image` / :class:`~torchvision.tv_tensors.Mask`
samples.

To remove a transform from the refused set (after registering the
necessary kernels), delete the entry from ``_REFUSED``.

Flip-axis convention
--------------------

The registered kernels map each image-space flip to a fixed world-axis
reflection:

* ``RandomHorizontalFlip`` → world **Y** reflection
* ``RandomVerticalFlip`` → world **Z** reflection

These choices match the intuition of an upright rig (image_y aligned with
-world_Z), but projection stays consistent for any camera orientation: the
extrinsics kernel reflects the matching camera-frame axis to absorb the
discrepancy.

World X-flip has no torchvision equivalent and stays in
:class:`vision3d.transforms.RandomFlip3D` (achievable via Y-flip + a
180° yaw rotation).
"""

from typing import TYPE_CHECKING, Any, override

import torchvision.transforms.v2 as _T
from torch.utils._pytree import tree_flatten

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

if TYPE_CHECKING:
    # Re-export torchvision's transform classes for static type-checking.
    from torchvision.transforms.v2 import *

_3D_AWARE_TVTENSORS = (
    PointCloud3D,
    BoundingBoxes3D,
    CameraImages,
    CameraExtrinsics,
    CameraIntrinsics,
)

#: Torchvision v2 transforms that vision3d refuses when a vision3d
#: TVTensor is present in the sample. Anything not listed is forwarded
#: from :mod:`torchvision.transforms.v2` unchanged. Edit this set as
#: kernels are added that make a transform safe for 3D-aware samples.
_REFUSED: frozenset[str] = frozenset(
    {
        # Non-rigid image-plane warp
        "ElasticTransform",
        "RandomPerspective",
        # Rigid but no extrinsics kernel yet
        "RandomAffine",
        "RandomRotation",
        # Unusual return structure (tuple of crops)
        "FiveCrop",
        "TenCrop",
        # Depends on 2D bounding boxes
        "RandomIoUCrop",
        # Bag of mixed ops (including chiral/non-rigid)
        "AugMix",
        "AutoAugment",
        "RandAugment",
        "TrivialAugmentWide",
        # Cross-sample mixing
        "CutMix",
        "MixUp",
    }
)


class _Refuse3DAwareMixin(_T.Transform):
    """Mixin that rejects vision3d-aware TVTensor inputs in ``check_inputs``."""

    @override
    def check_inputs(self, flat_inputs: list[Any]) -> None:
        """Raise if any vision3d-aware TVTensor is present.

        Refusal runs before delegating to the wrapped class's
        ``check_inputs`` so vision3d's diagnostic always wins over a
        torchvision check that might fire for an unrelated reason
        (e.g. :class:`RandomIoUCrop` requiring 2D bounding boxes).

        Raises:
            TypeError: If any input is a vision3d TVTensor.
        """
        incompatible_types = sorted(
            {
                type(inpt).__name__
                for inpt in flat_inputs
                if isinstance(inpt, _3D_AWARE_TVTENSORS)
            }
        )
        if incompatible_types:
            msg = (
                f"{type(self).__name__} cannot operate on samples that "
                f"contain vision3d TVTensors {incompatible_types}: applying "
                f"it without coordinated changes to the 3D scene would "
                f"break geometric consistency."
            )
            raise TypeError(msg)
        super().check_inputs(flat_inputs)

    @override
    def forward(self, *inputs: Any) -> Any:
        """Run ``check_inputs`` before delegating to the wrapped transform.

        Several torchvision transforms (``AutoAugment``, ``RandAugment``,
        ``AugMix``, ``TrivialAugmentWide``, ``CutMix``, ``MixUp``)
        override :meth:`forward` and never call :meth:`check_inputs`
        themselves. Running the refusal check here guarantees it fires
        regardless of how the parent class dispatches.

        Returns:
            The output of ``super().forward(*inputs)`` if the refusal
            check passes.
        """
        flat_inputs, _ = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        self.check_inputs(flat_inputs)
        return super().forward(*inputs)


_WRAPPED_CLASSES: dict[str, type] = {}


def _wrap_refused(parent: type[_T.Transform]) -> type[_T.Transform]:
    """Build a refuse-3d-aware subclass of a torchvision transform.

    Returns:
        A new class subclassing :class:`_Refuse3DAwareMixin` and ``parent``,
        with ``__module__`` / ``__qualname__`` set so it pickles as a
        ``vision3d.transforms.v2`` attribute.
    """
    cls = type(parent.__name__, (_Refuse3DAwareMixin, parent), {})
    cls.__module__ = __name__
    cls.__qualname__ = parent.__name__
    cls.__doc__ = parent.__doc__
    return cls


def __getattr__(name: str) -> Any:
    """Forward attributes from torchvision.transforms.v2.

    Names in :data:`_REFUSED` are wrapped with :class:`_Refuse3DAwareMixin`
    on first access and cached so identity is stable across calls (so
    ``isinstance``, pickling, and ``issubclass`` checks behave normally).

    Returns:
        The torchvision class for ``name``, or its refuse-3d-aware
        subclass if ``name`` is in :data:`_REFUSED`.

    Raises:
        AttributeError: If ``name`` is private or not exposed by
            :mod:`torchvision.transforms.v2`.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    parent = getattr(_T, name, None)
    if parent is None:
        msg = f"module 'vision3d.transforms.v2' has no attribute {name!r}"
        raise AttributeError(msg)
    if name not in _REFUSED:
        return parent
    if name not in _WRAPPED_CLASSES:
        _WRAPPED_CLASSES[name] = _wrap_refused(parent)
    return _WRAPPED_CLASSES[name]


def __dir__() -> list[str]:
    """Expose every public name in :mod:`torchvision.transforms.v2`.

    Returns:
        Sorted list of public attribute names, combining torchvision's
        v2 surface with the wrapped refused entries.
    """
    return sorted({n for n in dir(_T) if not n.startswith("_")} | _REFUSED)
