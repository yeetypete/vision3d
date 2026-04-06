"""Kernel dispatch for vision3d transforms.

Minimal reimplementation of torchvision's kernel registry, since the public
``register_kernel`` only allows registering kernels for torchvision's own
functionals.
"""

import functools
from typing import TYPE_CHECKING

import torch
from torchvision.tv_tensors import TVTensor

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# {functional: {input_type: kernel}}
KERNEL_REGISTRY: dict[Callable[..., Any], dict[type, Callable[..., Any]]] = {}


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
    registry = KERNEL_REGISTRY.setdefault(functional, {})

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

                output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)
                return wrap(output, like=inpt)

            registry[tv_tensor_cls] = wrapper
        else:
            registry[tv_tensor_cls] = kernel

        return kernel

    return decorator


def _get_kernel(functional: Callable[..., Any], input_type: type) -> Callable[..., Any]:
    """Look up the registered kernel for a functional and input type.

    Falls back to passthrough for unregistered types (labels, etc.).

    Args:
        functional: The functional to look up.
        input_type: The type of the input.

    Returns:
        The kernel function, or a passthrough lambda.
    """
    registry = KERNEL_REGISTRY.get(functional, {})
    for cls in input_type.__mro__:
        if cls in registry:
            return registry[cls]
        if cls is TVTensor:
            break
    # Passthrough for plain tensors, labels, etc.
    return lambda inpt, *args, **kwargs: inpt
