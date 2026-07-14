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
            leaf, since callers that operate on a single box set cannot tell
            which one to use.
    """
    boxes = [inpt for inpt in flat_inputs if isinstance(inpt, BoundingBoxes3D)]
    if len(boxes) > 1:
        msg = (
            "found multiple BoundingBoxes3D leaves in the sample; "
            "RangeFilter3D supports exactly one box set"
        )
        raise ValueError(msg)
    return boxes[0] if boxes else None


def _resolve_label_ids(
    labels: Any, flat_inputs: list[Any], n_boxes: int | None
) -> set[int]:
    """Validate a ``labels_getter`` result and return the label leaves' ids.

    Labels are matched to their sample leaf by identity, so a caller can locate
    each label tensor among the flattened leaves regardless of where it lives.
    This normalises the getter's return value, checks that each label tensor is
    an actual leaf of the sample (not a copy, view, or nested tensor), and, when
    the sample carries boxes, that each label tensor is per-box.

    Args:
        labels: The raw return value of a ``labels_getter``: a tensor, a
            tuple/list of tensors, or ``None``.
        flat_inputs: Leaves from :func:`torch.utils._pytree.tree_flatten`.
        n_boxes: Number of boxes in the sample, or ``None`` when the sample
            carries no boxes (in which case per-box length is not checked).

    Returns:
        The set of ``id()`` values of the label tensors, or an empty set when
        ``labels`` is ``None``.

    Raises:
        ValueError: If ``labels`` is not a tensor, tuple/list of tensors, or
            ``None``. If a returned tensor is not a leaf of the sample. If a
            returned tensor's length does not match ``n_boxes``.
    """
    if labels is None:
        return set()
    if isinstance(labels, Tensor):
        labels = (labels,)
    elif isinstance(labels, (tuple, list)) and all(
        isinstance(label, Tensor) for label in labels
    ):
        labels = tuple(labels)
    else:
        msg = (
            "`labels_getter` must return a tensor, a tuple/list of "
            f"tensors, or None, but got {type(labels).__name__}"
        )
        raise ValueError(msg)
    leaf_ids = {id(leaf) for leaf in flat_inputs}
    for label in labels:
        if id(label) not in leaf_ids:
            msg = (
                "`labels_getter` must return label tensor(s) that are "
                "leaves of the sample, not a copy, view, or nested tensor"
            )
            raise ValueError(msg)
        n_label = label.shape[0] if label.ndim else 0
        if n_boxes is not None and n_label != n_boxes:
            got = "0-dim" if not label.ndim else f"length {n_label}"
            msg = (
                f"`labels_getter` returned a {got} label tensor, but the "
                f"sample has {n_boxes} boxes; labels must be per-box"
            )
            raise ValueError(msg)
    return {id(label) for label in labels}


def _default_labels_getter(inputs: Any) -> Tensor:
    """Locate a per-box ``labels`` tensor by a case-insensitive ``"labels"`` key.

    Mirrors torchvision's default ``labels_getter``. The sample is a dict, or a
    two-tuple whose second element is the targets dict (or a bare labels
    tensor). Raises if no labels tensor can be found, so a silent no-op never
    hides a mislabelled sample. Callers that have no labels to sync should pass
    ``labels_getter=None`` instead of relying on this heuristic.

    Args:
        inputs: The sample passed to ``forward`` (a dict, or a two-tuple).

    Returns:
        The labels tensor found in the sample.

    Raises:
        ValueError: If no case-insensitive ``"labels"`` key is found, or the
            key exists but its value is not a tensor.
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
                if not isinstance(value, Tensor):
                    msg = (
                        "the default `labels_getter` found a 'labels' key whose "
                        f"value is a {type(value).__name__}, not a tensor. Pass "
                        "a callable as `labels_getter` to locate the labels, or "
                        "`labels_getter=None` if the sample has no labels."
                    )
                    raise ValueError(msg)
                return value
    msg = (
        "the default `labels_getter` could not find a labels tensor in the "
        "sample, expected a case-insensitive 'labels' key holding a tensor. "
        "Pass a callable as `labels_getter` to locate the labels, or "
        "`labels_getter=None` if the sample has no labels."
    )
    raise ValueError(msg)


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
