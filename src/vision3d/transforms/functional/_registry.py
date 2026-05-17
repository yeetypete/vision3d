"""Kernel dispatch for vision3d transforms.

Minimal reimplementation of torchvision's kernel registry, since the public
``register_kernel`` only allows registering kernels for torchvision's own
functionals.
"""

import functools
from collections.abc import Callable
from typing import Any

from torch import Tensor
from torchvision.tv_tensors import TVTensor

# {functional: {input_type: kernel}}
_KERNEL_REGISTRY: dict[Callable[..., Any], dict[type, Callable[..., Any]]] = {}


def register_kernel(
    functional: Callable[..., Any],
    tv_tensor_cls: type[TVTensor],
    *,
    tv_tensor_wrapper: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a kernel for a functional and TVTensor type.

    Args:
        functional: The functional to register a kernel for.
        tv_tensor_cls: The TVTensor subclass this kernel handles.
        tv_tensor_wrapper: If True (default), the kernel receives an unwrapped
            pure tensor and the output is automatically re-wrapped. If False,
            the kernel receives the full TVTensor and must handle wrap itself.

    Returns:
        Decorator that registers the kernel.
    """
    registry = _KERNEL_REGISTRY.setdefault(functional, {})

    def decorator(kernel: Callable[..., Any]) -> Callable[..., Any]:
        if tv_tensor_cls in registry:
            msg = (
                f"{functional.__name__} already has a kernel "
                f"registered for {tv_tensor_cls.__name__}."
            )
            raise ValueError(msg)

        if tv_tensor_wrapper:

            @functools.wraps(kernel)
            def wrapper(inpt: TVTensor, *args: Any, **kwargs: Any) -> TVTensor:
                from vision3d.tensors import wrap

                output = kernel(inpt.as_subclass(Tensor), *args, **kwargs)
                return wrap(output, like=inpt)

            registry[tv_tensor_cls] = wrapper
        else:
            registry[tv_tensor_cls] = kernel

        return kernel

    return decorator


def _get_kernel(
    functional: Callable[..., Any],
    input_type: type,
    *,
    allow_passthrough: bool = False,
) -> Callable[..., Any]:
    """Look up the registered kernel for a functional and input type.

    Args:
        functional: The functional to look up.
        input_type: The type of the input.
        allow_passthrough: If True, return an identity kernel when the
            functional has no registered kernel for ``input_type``. If
            False (default), raise :class:`TypeError`.

    Returns:
        The kernel function, or an identity lambda when
        ``allow_passthrough`` is True and no kernel is registered for
        ``input_type``.

    Raises:
        ValueError: If the functional has no kernels registered at all.
        TypeError: If the functional has kernels but none for
            ``input_type`` and ``allow_passthrough`` is False.
    """
    registry = _KERNEL_REGISTRY.get(functional)
    if not registry:
        msg = f"No kernel registered for functional `{functional.__name__}`."
        raise ValueError(msg)

    for cls in input_type.__mro__:
        if cls in registry:
            return registry[cls]
        if cls is TVTensor:
            break

    if allow_passthrough:
        return lambda inpt, *args, **kwargs: inpt

    msg = (
        f"Functional `{functional.__name__}` supports inputs of type "
        f"{sorted(c.__name__ for c in registry)}, but got "
        f"`{input_type.__name__}` instead."
    )
    raise TypeError(msg)
