"""Collation utilities for DataLoader."""

from typing import Any


def collate_fn(
    batch: list[tuple[Any, Any]],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Collate a batch of ``(inputs, targets)`` without stacking.

    Variable-size tensors (point clouds, bounding boxes) cannot be stacked
    into a single tensor. This collate function groups them as tuples,
    matching torchvision's detection collate pattern.

    Args:
        batch: List of ``(inputs, targets)`` from the dataset.

    Returns:
        Tuple of ``(inputs_tuple, targets_tuple)``.
    """
    inputs, targets = zip(*batch)
    return tuple(inputs), tuple(targets)
