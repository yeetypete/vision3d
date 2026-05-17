"""Base class for vision3d transforms.

Mirrors :class:`torchvision.transforms.v2.Transform`: subclasses declare
which input types they operate on via the class-level
:attr:`_transformed_types` attribute, and may override
:meth:`check_inputs` to reject unsupported input combinations. Inputs
whose type is not in :attr:`_transformed_types` flow through the
transform unchanged.
"""

import enum
from collections.abc import Callable
from typing import Any, override

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.transforms.v2 import check_type
from torchvision.tv_tensors import TVTensor

from .functional._registry import _get_kernel


class Transform(nn.Module):
    """Base class for vision3d transforms.

    Mirrors :class:`torchvision.transforms.v2.Transform`. Subclasses
    override :meth:`transform` and use ``_call_kernel`` to dispatch to
    the correct kernel for each input type.

    The class-level ``_transformed_types`` tuple lists the input types
    the transform operates on; inputs whose type is not listed flow
    through unchanged. Subclasses can additionally override
    :meth:`check_inputs` to reject unsupported input combinations with
    a :class:`TypeError`.
    """

    #: Input types this transform operates on. Each entry is either a
    #: concrete type or a callable predicate. Inputs whose type does not
    #: match are passed through unchanged. Defaults to
    #: :class:`~torchvision.tv_tensors.TVTensor`, so any TVTensor
    #: subclass is dispatched to :meth:`transform` and plain tensors
    #: pass through.
    _transformed_types: tuple[type | Callable[[Any], bool], ...] = (TVTensor,)

    def __init__(self) -> None:
        super().__init__()

    def check_inputs(self, flat_inputs: list[Any]) -> None:
        """Validate inputs before transforming. Override to reject inputs.

        The base implementation is a no-op. Subclasses raise
        :class:`TypeError` for input combinations the transform cannot
        handle (e.g. a 3D-only transform that has no matching update
        for camera tensors).
        """

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample random parameters. Override for randomised transforms.

        Returns:
            Parameter dict passed to :meth:`transform`.
        """
        return {}

    def _call_kernel(
        self, functional: Callable[..., Any], inpt: Any, *args: Any, **kwargs: Any
    ) -> Any:
        kernel = _get_kernel(functional, type(inpt), allow_passthrough=True)
        return kernel(inpt, *args, **kwargs)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the transform to a single input. Must be overridden."""
        raise NotImplementedError

    def _needs_transform_list(self, flat_inputs: list[Any]) -> list[bool]:
        """Return a per-input mask of which inputs to call ``transform`` on.

        Returns:
            List of bools, one per element of ``flat_inputs``.
        """
        return [check_type(inpt, self._transformed_types) for inpt in flat_inputs]

    @override
    def forward(self, *inputs: Any) -> Any:
        """Apply the transform to one or more inputs (dicts, tuples, etc.).

        Do not override this; override :meth:`transform` instead.

        Returns:
            Transformed inputs in the same structure as the input.
        """
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        self.check_inputs(flat_inputs)

        needs = self._needs_transform_list(flat_inputs)
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


class _RandomApplyTransform(Transform):
    """Base class for transforms applied with probability ``p``.

    Mirrors :class:`torchvision.transforms.v2._RandomApplyTransform`:
    :meth:`check_inputs` always runs, but the rest of the forward pass
    is skipped with probability ``1 - p``.
    """

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

        self.check_inputs(flat_inputs)

        if torch.rand(1) >= self.p:
            return inputs

        needs = self._needs_transform_list(flat_inputs)
        params = self.make_params([inpt for inpt, nt in zip(flat_inputs, needs) if nt])

        flat_outputs = [
            self.transform(inpt, params) if nt else inpt
            for inpt, nt in zip(flat_inputs, needs)
        ]

        return tree_unflatten(flat_outputs, spec)
