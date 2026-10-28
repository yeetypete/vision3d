"""Mirror of ``torchvision.transforms.v2`` for samples with 3D data.

Import torchvision transforms through this module whenever a sample
contains vision3d TVTensors. Swap

.. code-block:: python

    from torchvision.transforms import v2 as T

for

.. code-block:: python

    from vision3d.transforms import v2 as T

to make every transform that would silently break the geometric
consistency of a 3D scene refuse vision3d-aware TVTensor inputs with a
:class:`TypeError` instead. In samples that contain a vision3d TVTensor,
plain tensors such as labels are passed through rather than transformed
as images. Wrap a tensor in :class:`~torchvision.tv_tensors.Image` to
transform it as one.

The module forwards every public name from ``torchvision.transforms.v2``.
Transform classes are subclassed to add the behavior above. The
transforms listed in the module-private ``_REFUSED`` set additionally
refuse 3D samples: calling one on a sample containing any vision3d TVTensor
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

* ``RandomHorizontalFlip``: world **Y** reflection
* ``RandomVerticalFlip``: world **Z** reflection

These choices match the intuition of an upright rig (image_y aligned with
-world_Z), but projection stays consistent for any camera orientation: the
extrinsics kernel reflects the matching camera-frame axis to absorb the
discrepancy.

World X-flip has no torchvision equivalent and stays in
:class:`vision3d.transforms.RandomFlip3D` (achievable via Y-flip + a
180 degree yaw rotation).
"""

from typing import TYPE_CHECKING, Any, override

import torchvision.transforms.v2 as _T
from torch.utils._pytree import tree_flatten
from torchvision.transforms.v2 import has_any
from torchvision.transforms.v2.functional import is_pure_tensor

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
#: TVTensor is present in the sample. Edit this set as kernels are added
#: that make a transform safe for 3D-aware samples.
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


class _PassPureTensorsIn3DSamplesMixin(_T.Transform):
    """Mixin that passes plain tensors through unchanged in samples with vision3d TVTensors.

    torchvision has a heuristic that transforms the first plain tensor in a
    sample as an image when the sample has no
    :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, or PIL image. In a
    lidar-only sample that tensor usually represents box labels, which
    would then be reordered, corrupted, or cause a crash. We avoid this with
    this mixin. See
    https://github.com/pytorch/vision/blob/v0.29.1/torchvision/transforms/v2/_transform.py#L75-L105
    """

    @override
    def _needs_transform_list(self, flat_inputs: list[Any]) -> list[bool]:
        """Skip plain tensors when the sample holds a vision3d TVTensor.

        Returns:
            List of bools, one per element of ``flat_inputs``.
        """
        needs = super()._needs_transform_list(flat_inputs)
        if not has_any(flat_inputs, *_3D_AWARE_TVTENSORS):
            return needs
        return [
            need and not is_pure_tensor(inpt)
            for need, inpt in zip(needs, flat_inputs, strict=True)
        ]


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


#: Cache of wrapped transform classes by name.
_WRAPPED_CLASSES: dict[str, type] = {}


def _wrap(parent: type[_T.Transform], *, refuse: bool) -> type[_T.Transform]:
    """Build a vision3d-aware subclass of a torchvision transform.

    Returns:
        A new class subclassing :class:`_PassPureTensorsIn3DSamplesMixin`,
        :class:`_Refuse3DAwareMixin` if ``refuse`` is set, and ``parent``,
        with ``__module__`` / ``__qualname__`` set so it pickles as a
        ``vision3d.transforms.v2`` attribute.
    """
    mixins: tuple[type, ...] = (_PassPureTensorsIn3DSamplesMixin,)
    if refuse:
        mixins = (_Refuse3DAwareMixin, *mixins)
    return type(
        parent.__name__,
        (*mixins, parent),
        {
            "__module__": __name__,
            "__qualname__": parent.__name__,
            "__doc__": parent.__doc__,
        },
    )


def __getattr__(name: str) -> Any:
    """Forward attributes from torchvision.transforms.v2.

    Transform classes are wrapped with
    :class:`_PassPureTensorsIn3DSamplesMixin`, and names in
    :data:`_REFUSED` additionally with :class:`_Refuse3DAwareMixin`.
    Wrapped classes are built on first access and cached so identity is
    stable across calls (so ``isinstance``, pickling, and ``issubclass``
    checks behave normally). Other names are forwarded unchanged.

    Returns:
        The vision3d-aware subclass for a transform class, otherwise the
        torchvision attribute for ``name``.

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
    if not (isinstance(parent, type) and issubclass(parent, _T.Transform)):
        return parent
    if name not in _WRAPPED_CLASSES:
        _WRAPPED_CLASSES[name] = _wrap(parent, refuse=name in _REFUSED)
    return _WRAPPED_CLASSES[name]


def __dir__() -> list[str]:
    """Expose every public name in :mod:`torchvision.transforms.v2`.

    Returns:
        Sorted list of public attribute names, combining torchvision's
        v2 surface with the wrapped refused entries.
    """
    return sorted({n for n in dir(_T) if not n.startswith("_")} | _REFUSED)
