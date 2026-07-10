"""Shared helpers for transforms."""

from collections.abc import Callable
from typing import Any

from torch import Tensor

from vision3d.tensors import BoundingBoxes3D


def _find_boxes(flat_inputs: list[Any]) -> BoundingBoxes3D | None:
    """Return the sole ``BoundingBoxes3D`` leaf, or ``None`` if absent.

    Args:
        flat_inputs: Leaves from :func:`torch.utils._pytree.tree_flatten`.

    Returns:
        The single :class:`~vision3d.tensors.BoundingBoxes3D` in
        ``flat_inputs``, or ``None`` when the sample carries no boxes.

    Raises:
        ValueError: If the sample holds more than one ``BoundingBoxes3D``
            leaf, since a single keep-mask cannot be applied unambiguously
            across box sets of differing length.
    """
    boxes = [inpt for inpt in flat_inputs if isinstance(inpt, BoundingBoxes3D)]
    if len(boxes) > 1:
        msg = (
            "found multiple BoundingBoxes3D leaves in the sample; "
            "RangeFilter3D supports exactly one box set"
        )
        raise ValueError(msg)
    return boxes[0] if boxes else None


def _default_labels_getter(inputs: Any) -> Tensor | None:
    """Locate a per-box ``labels`` tensor by a case-insensitive ``"labels"`` key.

    Mirrors torchvision's default ``labels_getter``: the sample is a dict, or a
    two-tuple whose second element is the targets dict (or a bare labels
    tensor). Returns ``None`` when nothing matches, so labels stay optional.

    Args:
        inputs: The sample passed to ``forward`` (a dict, or a two-tuple).

    Returns:
        The labels tensor, or ``None`` if the sample has no labels.
    """
    candidate: Any = inputs
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
        second = inputs[1]
        if isinstance(second, Tensor):
            return second
        candidate = second
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            if isinstance(key, str) and key.lower() == "labels":
                return value if isinstance(value, Tensor) else None
    return None


def _parse_labels_getter(
    labels_getter: str | Callable[[Any], Any] | None,
) -> Callable[[Any], Any]:
    """Resolve the ``labels_getter`` argument to a callable.

    Args:
        labels_getter: ``"default"`` for the built-in heuristic, a callable
            taking the sample and returning the labels tensor (or ``None``),
            or ``None`` to disable label syncing.

    Returns:
        A callable mapping a sample to its labels tensor or ``None``.

    Raises:
        ValueError: If ``labels_getter`` is not ``"default"``, a callable, or
            ``None``.
    """
    if labels_getter == "default":
        return _default_labels_getter
    if callable(labels_getter):
        return labels_getter
    if labels_getter is None:
        return lambda _inputs: None
    msg = "`labels_getter` must be 'default', a callable, or None."
    raise ValueError(msg)
