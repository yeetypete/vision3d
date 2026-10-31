"""Shared helpers for transforms."""

from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor
from torch.utils._pytree import tree_leaves
from torchvision.transforms.v2 import has_any
from torchvision.transforms.v2.functional import is_pure_tensor

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


def _find_label_key(mapping: Mapping[Any, Any]) -> str | None:
    """Find the key holding a mapping's labels.

    A case-insensitive ``"labels"`` key takes priority, otherwise the first key
    containing ``"label"`` is used.

    Returns:
        The matching key, or ``None`` if the mapping has no label-like key.
    """
    keys = [key for key in mapping if isinstance(key, str)]
    key = next((key for key in keys if key.lower() == "labels"), None)
    if key is None:
        key = next((key for key in keys if "label" in key.lower()), None)
    return key


def _default_labels_getter(inputs: Any) -> Any:
    """Locate a sample's per-box labels under a label-like key.

    Mirrors torchvision's default ``labels_getter``. The sample is a mapping, or
    a tuple/list whose second item is that mapping or a pure labels tensor. In
    the mapping, a case-insensitive ``"labels"`` key takes priority, otherwise
    the first key containing ``"label"`` is used (e.g. ``gt_labels``). Callers
    that have no labels to sync should pass ``labels_getter=None``.

    Args:
        inputs: The sample passed to ``forward``.

    Returns:
        The value stored under the label-like key, or the pure tensor.

    Raises:
        ValueError: If the sample is not structured as described above, or
            holds no label-like key.
    """
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[1]
    if is_pure_tensor(inputs):
        return inputs
    if not isinstance(inputs, Mapping):
        msg = (
            "the default `labels_getter` expects a mapping, or a tuple/list whose "
            "second item is a mapping or a labels tensor, but got "
            f"{type(inputs).__name__}. Pass a callable as `labels_getter` to "
            "locate the labels, or `labels_getter=None` if the sample has no "
            "labels."
        )
        raise ValueError(msg)  # noqa: TRY004
    key = _find_label_key(inputs)
    if key is None:
        msg = (
            "the default `labels_getter` could not find a label-like key in the "
            "sample, expected a key matching 'labels' or containing 'label'. Pass "
            "a callable as `labels_getter` to locate the labels, or "
            "`labels_getter=None` if the sample has no labels."
        )
        raise ValueError(msg)
    return inputs[key]


def _has_boxes(node: Any) -> bool:
    return isinstance(node, Mapping) and has_any(list(node.values()), BoundingBoxes3D)


def _default_batch_labels_getter(inputs: Any) -> tuple[Any, ...]:
    """Locate the per-box labels of every sample in a batch.

    Each mapping that directly holds a ``BoundingBoxes3D`` must also hold that
    sample's labels, under a key chosen as in :func:`_default_labels_getter`.

    Args:
        inputs: The batch passed to ``forward``.

    Returns:
        The labels of each box-holding mapping, in traversal order.

    Raises:
        ValueError: If a mapping holds boxes but no label-like key, or no
            mapping holds boxes.
    """
    found: list[Any] = []
    for mapping in filter(_has_boxes, tree_leaves(inputs, is_leaf=_has_boxes)):
        key = _find_label_key(mapping)
        if key is None:
            msg = (
                "the default `labels_getter` could not find a label-like key next "
                "to the boxes, expected a key matching 'labels' or containing "
                "'label' in each mapping holding a BoundingBoxes3D tensor. Pass a "
                "callable as `labels_getter` to locate the labels, or "
                "`labels_getter=None` if the batch has no labels."
            )
            raise ValueError(msg)
        found.append(mapping[key])
    if not found:
        msg = (
            "the default `labels_getter` expects each sample's boxes and labels in "
            "a shared mapping, but found no mapping holding a BoundingBoxes3D tensor. "
            "Pass a callable as `labels_getter` to locate the labels, or "
            "`labels_getter=None` if the batch has no labels."
        )
        raise ValueError(msg)
    return tuple(found)


def _no_labels_getter(_inputs: Any) -> None:
    """Report that a sample contains no labels."""
    return


def _parse_labels_getter(
    labels_getter: str | Callable[[Any], Any] | None,
    default: Callable[[Any], Any] = _default_labels_getter,
) -> Callable[[Any], Any]:
    """Resolve the ``labels_getter`` argument to a callable.

    Args:
        labels_getter: ``"default"`` for the built-in heuristic, a callable
            taking the sample and returning the labels tensor (or ``None``),
            or ``None`` to disable label syncing.
        default: The getter that ``"default"`` resolves to.

    Returns:
        A callable mapping a sample to its labels tensor or ``None``.

    Raises:
        ValueError: If ``labels_getter`` is not ``"default"``, a callable, or
            ``None``.
    """
    if labels_getter == "default":
        return default
    if callable(labels_getter):
        return labels_getter
    if labels_getter is None:
        return _no_labels_getter
    msg = "`labels_getter` must be 'default', a callable, or None."
    raise ValueError(msg)
