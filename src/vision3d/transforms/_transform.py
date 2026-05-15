"""Base class for vision3d transforms."""

import enum
from collections.abc import Callable
from typing import Any, override

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.tv_tensors import TVTensor

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

from .functional._registry import _get_kernel

#: The full set of TVTensor types defined in vision3d. Transforms that
#: operate safely across every modality declare this as their
#: ``_safe_for``. Extend this set when a new vision3d TVTensor
#: is added.
ALL_VISION3D_TVTENSORS: frozenset[type[TVTensor]] = frozenset(
    {
        PointCloud3D,
        BoundingBoxes3D,
        CameraImages,
        CameraExtrinsics,
        CameraIntrinsics,
    }
)


class GeometricConsistencyError(TypeError):
    """Raised when a transform would break scene geometric consistency.

    Signals that a transform received a TVTensor it does not know how to
    update jointly with the rest of the scene.
    """


def _needs_transform(inpt: Any) -> bool:
    """Only TVTensor subclasses are transformed. Plain tensors pass through.

    Returns:
        True if ``inpt`` is a TVTensor subclass.
    """
    return isinstance(inpt, TVTensor)


def _check_safety(
    safe_for: frozenset[type[TVTensor]],
    flat_inputs: list[Any],
    transform_name: str,
) -> None:
    """Raise if any TVTensor input is outside the declared safe set.

    Membership is checked by exact type. A TVTensor subclass of a
    listed type is still treated as unsafe, so a transform must opt
    in to every concrete type it sees.

    Raises:
        GeometricConsistencyError: If any input is a TVTensor whose
            exact type is not in ``safe_for``.
    """
    input_types = {type(inpt) for inpt in flat_inputs if isinstance(inpt, TVTensor)}
    unsafe = input_types - safe_for
    if not unsafe:
        return
    safe_names = sorted(t.__name__ for t in safe_for) or ["(none)"]
    unsafe_names = sorted(t.__name__ for t in unsafe)
    msg = (
        f"{transform_name} received input(s) of type {unsafe_names}, "
        f"but only declares handling for {safe_names}. Running it would "
        f"break scene geometric consistency for the given inputs. Drop "
        f"the input, use a different transform, or extend `_safe_for` "
        f"if you have verified the behaviour is correct."
    )
    raise GeometricConsistencyError(msg)


class Transform(nn.Module):
    """Base class for vision3d transforms.

    Only :class:`~torchvision.tv_tensors.TVTensor` subclasses (e.g.
    :class:`~vision3d.tensors.BoundingBoxes3D`,
    :class:`~vision3d.tensors.PointCloud3D`) are transformed.
    Plain tensors (labels, scores, etc.) pass through unchanged.

    Subclasses should override :meth:`transform` and use ``_call_kernel``
    to dispatch to the correct kernel for each input type.

    Transforms are unsafe by default: a TVTensor input is accepted only
    if its type is listed in the class-level ``_safe_for`` attribute.
    Subclasses must include every type the transform handles, whether by
    updating it or by intentionally leaving it untouched. This prevents
    silently producing geometrically inconsistent scenes (e.g. flipping
    lidar but not the camera image alongside it).
    """

    _safe_for: frozenset[type[TVTensor]] = frozenset()

    def __init__(self) -> None:
        super().__init__()

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample random parameters. Override for randomised transforms.

        Returns:
            Parameter dict passed to :meth:`transform`.
        """
        return {}

    def _call_kernel(
        self, functional: Callable[..., Any], inpt: Any, *args: Any, **kwargs: Any
    ) -> Any:
        kernel = _get_kernel(functional, type(inpt))
        return kernel(inpt, *args, **kwargs)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the transform to a single input. Must be overridden."""
        raise NotImplementedError

    @override
    def forward(self, *inputs: Any) -> Any:
        """Apply the transform to one or more inputs (dicts, tuples, etc.).

        Returns:
            Transformed inputs in the same structure as the input.
        """
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        _check_safety(self._safe_for, flat_inputs, type(self).__name__)

        needs = [_needs_transform(inpt) for inpt in flat_inputs]
        params = self.make_params([inpt for inpt, nt in zip(flat_inputs, needs) if nt])

        flat_outputs = [
            self.transform(inpt, params) if nt else inpt
            for inpt, nt in zip(flat_inputs, needs)
        ]

        return tree_unflatten(flat_outputs, spec)

    @override
    def extra_repr(self) -> str:
        """Auto-generate repr from public attributes.

        Returns:
            Comma-separated key=value string.
        """
        extra = []
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue
            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                continue
            extra.append(f"{name}={value}")
        return ", ".join(extra)


class RandomTransform(Transform):
    """Base class for transforms applied with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            msg = "`p` should be a float in [0.0, 1.0]."
            raise ValueError(msg)
        super().__init__()
        self.p = p

    @override
    def forward(self, *inputs: Any) -> Any:
        """Apply the transform with probability ``p``.

        Returns:
            Transformed inputs, or the original inputs if skipped.
        """
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        _check_safety(self._safe_for, flat_inputs, type(self).__name__)
        if torch.rand(1) >= self.p:
            return inputs

        needs = [_needs_transform(inpt) for inpt in flat_inputs]
        params = self.make_params([inpt for inpt, nt in zip(flat_inputs, needs) if nt])

        flat_outputs = [
            self.transform(inpt, params) if nt else inpt
            for inpt, nt in zip(flat_inputs, needs)
        ]

        return tree_unflatten(flat_outputs, spec)
