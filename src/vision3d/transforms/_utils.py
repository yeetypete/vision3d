"""Shared helpers for transforms."""

from collections.abc import Callable, Mapping
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


def _find_label_key(mapping: Mapping[Any, Any]) -> Any | None:
    """Find the key holding ``mapping``'s labels.

    A case-insensitive ``"labels"`` takes priority. Otherwise the first key
    containing ``"label"`` is used, which covers names like ``gt_labels``.

    Args:
        mapping: The targets mapping to search.

    Returns:
        The matching key, or ``None`` if the mapping has no label-like key.
    """
    for key in mapping:
        if isinstance(key, str) and key.lower() == "labels":
            return key
    for key in mapping:
        if isinstance(key, str) and "label" in key.lower():
            return key
    return None


def _default_labels_getter(inputs: Any) -> Any:
    """Locate a sample's per-box labels under a label-like key.

    Accepts a targets mapping, or a sequence whose second element is that
    mapping or a bare labels tensor. Pass ``labels_getter=None`` for samples
    that carry no labels.

    Args:
        inputs: The sample passed to ``forward``.

    Returns:
        The value stored under the sample's label-like key.

    Raises:
        ValueError: If the sample is too short to hold targets, or holds no
            label-like key.
    """
    if isinstance(inputs, (tuple, list)):
        if len(inputs) < 2:
            msg = (
                "the default `labels_getter` expects a mapping or a sequence "
                f"whose second element holds the targets, but got a "
                f"{len(inputs)}-element sequence."
            )
            raise ValueError(msg)
        inputs = inputs[1]
    if isinstance(inputs, Tensor):
        return inputs
    key = _find_label_key(inputs) if isinstance(inputs, Mapping) else None
    if key is None:
        msg = (
            "the default `labels_getter` could not find the labels in the "
            "sample, expected a mapping (or a sequence whose second element is a "
            "mapping or a tensor) holding a key that matches 'labels' or "
            "contains 'label'. Pass a callable as `labels_getter` to locate the "
            "labels, or `labels_getter=None` if the sample has no labels."
        )
        raise ValueError(msg)
    return inputs[key]


def _no_labels_getter(_inputs: Any) -> None:
    """Report that a sample carries no labels."""
    return


def _collect_labels(node: Any, found: list[Any]) -> None:
    """Gather the label-like value of every mapping nested in ``node``.

    Walks dicts, lists, and tuples. Each mapping contributes at most one entry,
    so a collated batch yields one per sample. A sequence stored under the key is
    flattened into separate entries.

    Args:
        node: Current node of the sample structure.
        found: Accumulator, appended to in traversal order.
    """
    if isinstance(node, Mapping):
        key = _find_label_key(node)
        for candidate, value in node.items():
            if key is not None and candidate == key:
                if isinstance(value, (tuple, list)):
                    found.extend(value)
                else:
                    found.append(value)
            else:
                _collect_labels(value, found)
    elif isinstance(node, (tuple, list)) and not isinstance(node, Tensor):
        for item in node:
            _collect_labels(item, found)


def _default_batch_labels_getter(inputs: Any) -> tuple[Any, ...]:
    """Locate the per-box labels of every sample in a collated batch.

    Searches the batch at any depth and returns one value per sample.

    Args:
        inputs: The batch passed to ``forward``.

    Returns:
        The labels values found, in traversal order.

    Raises:
        ValueError: If the batch holds no label-like key.
    """
    found: list[Any] = []
    _collect_labels(inputs, found)
    if not found:
        msg = (
            "the default `labels_getter` could not find any labels tensor in the "
            "batch, expected each sample's labels under a key matching 'labels' "
            "(or containing 'label'). Pass a callable as `labels_getter`, or "
            "`labels_getter=None` if the batch has no labels."
        )
        raise ValueError(msg)
    return tuple(found)


def _parse_labels_getter(
    labels_getter: str | Callable[[Any], Any] | None,
    *,
    default: Callable[[Any], Any] | None = None,
) -> Callable[[Any], Any]:
    """Resolve the ``labels_getter`` argument to a callable.

    Args:
        labels_getter: ``"default"`` for the built-in heuristic, a callable
            taking the sample and returning the labels tensor (or ``None``),
            or ``None`` to disable label syncing.
        default: Heuristic that ``"default"`` resolves to. Defaults to
            :func:`_default_labels_getter`. Batch transforms pass
            :func:`_default_batch_labels_getter`.

    Returns:
        A callable mapping a sample to its labels tensor or ``None``.

    Raises:
        ValueError: If ``labels_getter`` is not ``"default"``, a callable, or
            ``None``.
    """
    if labels_getter == "default":
        return default or _default_labels_getter
    if callable(labels_getter):
        return labels_getter
    if labels_getter is None:
        return _no_labels_getter
    msg = "`labels_getter` must be 'default', a callable, or None."
    raise ValueError(msg)
