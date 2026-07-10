"""Shared helpers for transforms."""

from typing import Any

from torch import Tensor

from vision3d.tensors import BoundingBoxes3D


def _filter_boxes_and_labels(d: dict[str, Any], keep: Tensor) -> None:
    """Filter ``boxes`` and synced ``labels`` in-place by a keep-mask.

    Rebuilds ``d["boxes"]`` from the kept rows and slices ``d["labels"]``
    with the same mask so the two stay in step. Assumes ``d`` has a
    ``boxes`` entry; callers guard on that themselves.

    Args:
        d: Sample or targets dict, mutated in place.
        keep: 1D boolean tensor ``[M]``; ``True`` where a box is kept.
    """
    # TODO: When we add more per-box annotations (e.g. velocities), mask
    # them here too, else they desync from ``boxes`` after filtering.
    boxes = d["boxes"]
    d["boxes"] = BoundingBoxes3D(boxes.as_subclass(Tensor)[keep], format=boxes.format)
    if "labels" in d:
        d["labels"] = d["labels"][keep]
