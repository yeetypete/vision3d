"""Base class for vision3d transforms."""

import enum
from collections.abc import Callable
from typing import Any, override

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.tv_tensors import TVTensor

from .functional._registry import _get_kernel


def _needs_transform(inpt: Any) -> bool:
    """Only TVTensor subclasses are transformed. Plain tensors pass through.

    Returns:
        True if ``inpt`` is a TVTensor subclass.
    """
    return isinstance(inpt, TVTensor)


class Transform(nn.Module):
    """Base class for vision3d transforms.

    Only :class:`~torchvision.tv_tensors.TVTensor` subclasses (e.g.
    :class:`~vision3d.tensors.BoundingBoxes3D`,
    :class:`~vision3d.tensors.PointCloud3D`) are transformed.
    Plain tensors (labels, scores, etc.) pass through unchanged.

    Subclasses should override :meth:`transform` and use ``_call_kernel``
    to dispatch to the correct kernel for each input type.
    """

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
        if torch.rand(1) >= self.p:
            return inputs
        flat_inputs, spec = tree_flatten(inputs)

        needs = [_needs_transform(inpt) for inpt in flat_inputs]
        params = self.make_params([inpt for inpt, nt in zip(flat_inputs, needs) if nt])

        flat_outputs = [
            self.transform(inpt, params) if nt else inpt
            for inpt, nt in zip(flat_inputs, needs)
        ]

        return tree_unflatten(flat_outputs, spec)
