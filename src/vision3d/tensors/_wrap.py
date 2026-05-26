from typing import Any, cast

from torch import Tensor
from torchvision.tv_tensors import TVTensor

# Mirrors torchvision 0.28's dispatch
# (https://github.com/pytorch/vision/pull/9490). Required while vision3d is
# compatible with torchvision 0.25-0.27, whose wrap() hardcodes its own
# tv_tensor types. Replace with ``from torchvision.tv_tensors import wrap``
# once vision3d requires ``torchvision>=0.28``.


def wrap[T: TVTensor](
    wrappee: Tensor,
    *,
    like: T,
    **kwargs: Any,
) -> T:
    """Convert a :class:`~torch.Tensor` into the same :class:`~torchvision.tv_tensors.TVTensor` subclass as ``like``.

    If ``like`` carries metadata (e.g. ``format``, ``image_size``), that
    metadata is copied to the output. Individual fields can be overridden
    via ``kwargs``.

    Subclass authors can define a ``wrap`` classmethod on their subclass
    of :class:`~torchvision.tv_tensors.TVTensor` to control how metadata
    propagates.

    Args:
        wrappee (:class:`~torch.Tensor`): The tensor to convert.
        like (:class:`~torchvision.tv_tensors.TVTensor`): The reference.
            ``wrappee`` will be converted into the same subclass as ``like``.
        kwargs: Metadata overrides forwarded to the subclass's ``wrap``
            classmethod.

    Returns:
        :class:`~torchvision.tv_tensors.TVTensor`: A TVTensor of the same
        subclass as ``like``.
    """
    if (wrap_method := getattr(type(like), "wrap", None)) is not None:
        return cast("T", wrap_method(wrappee, like, **kwargs))
    return wrappee.as_subclass(type(like))
