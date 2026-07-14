"""Minimum-points filtering for ground-truth boxes."""

from collections.abc import Callable
from typing import Any, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from vision3d.ops import points_in_boxes_3d
from vision3d.tensors import BoundingBoxes3D, PointCloud3D

from ._transform import Transform
from ._utils import (
    _filter_boxes_and_labels,
    _find_boxes,
    _find_points,
    _parse_labels_getter,
    _resolve_label_ids,
)


class ObjectMinPointsFilter(Transform):
    """Drop ground-truth boxes that enclose fewer than ``min_points`` points.

    Counts the points inside each box and removes boxes whose interior
    point count is strictly below ``min_points``; any per-box ``labels`` are
    filtered in sync. The point cloud and every other sample entry pass
    through unchanged. The sample must hold at most one ``BoundingBoxes3D``
    set and at most one ``PointCloud3D`` (a single keep-mask cannot span box
    sets of differing length, and counting needs one unambiguous point cloud).
    Counting is format-agnostic (incl. 9-DoF); a point in the overlap of
    several boxes is counted once for **each** box it lies in.

    Boxes with too few points carry little geometric evidence, so training
    against them mostly injects noise. This is the analog of mmdetection3d's
    ``filter_by_min_points`` ground-truth-database option.

    Pairs naturally with ``ObjectPointsSample``: when that transform
    simulates sparse returns by thinning each object down to a target
    ``keep`` count, applying this filter afterward with
    ``min_points`` set to the same value drops the objects sparse simulation
    pushed below the detectability floor. The two transforms stay
    orthogonal -- match the thresholds yourself; nothing couples them
    automatically.

    When the sample has no ``points`` entry, no box has a defined count and
    every box is kept. A present-but-empty point cloud instead counts as
    zero points per box (so boxes are dropped unless ``min_points`` is 0).

    Only the boxes and the located ``labels`` are filtered; any additional
    per-box annotations (e.g. velocities) must be passed via ``labels_getter``
    to stay in sync, or they will be left at their pre-filter length.

    Args:
        min_points: Minimum number of interior points a box must contain to
            survive. A box is dropped when its count is strictly less than
            this. ``0`` keeps every box; ``1`` keeps every box with at least
            one point.
        labels_getter: How to locate the per-box label tensor(s) that are
            filtered in sync with the boxes. Pass a callable that takes the
            sample and returns the label tensor stored in it, a tuple or list
            of such tensors (when several label tensors track the same boxes),
            or ``None``. The returned tensors must be the exact objects stored
            in the sample, not copies or views, since labels are matched to
            their leaf by identity. Alternatively, pass the string
            ``"default"`` (the default) to use the built-in heuristic, which
            finds the labels under a case-insensitive ``"labels"`` key and
            raises if the sample has boxes but no such tensor, or pass ``None``
            to filter the boxes without touching any labels.
    """

    def __init__(
        self,
        min_points: int,
        labels_getter: str | Callable[[Any], Any] | None = "default",
    ) -> None:
        super().__init__()
        if isinstance(min_points, bool) or not isinstance(min_points, int):
            msg = "`min_points` should be a non-negative int."
            raise TypeError(msg)
        if min_points < 0:
            msg = "`min_points` should be a non-negative int."
            raise ValueError(msg)
        self.min_points = min_points
        self.labels_getter = labels_getter
        self._labels_getter = _parse_labels_getter(labels_getter)

    @override
    def forward(self, *inputs: Any) -> Any:
        """Filter boxes with too few interior points.

        Accepts both a single sample dict and an ``(inputs, targets)`` pair.
        The point cloud is read from wherever it appears (the same dict, or
        the ``inputs`` dict of a pair) and is never modified.

        Returns:
            Filtered sample in the same structure as the input.

        Raises:
            ValueError: If called with no inputs. If the sample holds more
                than one box set or more than one point cloud. If the sample
                has boxes but the default ``labels_getter`` cannot find a
                labels tensor. If ``labels_getter`` returns a tensor that is
                not a leaf of the sample (e.g. a copy, view, or nested
                tensor). If a returned label tensor's length does not match
                the number of boxes.
        """
        # Hand-rolled rather than routed through the base per-leaf
        # ``transform()`` loop, for the same reason as RangeFilter3D: labels are
        # plain tensors with no distinguishing type, so they can only be located
        # via ``labels_getter`` against the nested structure, which it flattens away.
        if not inputs:
            msg = "ObjectMinPointsFilter.forward requires at least one input sample"
            raise ValueError(msg)
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        boxes = _find_boxes(flat_inputs)
        # No boxes means nothing to filter; return the sample untouched (still
        # rebuilt from the flattened leaves so the structure round-trips).
        if boxes is None:
            return tree_unflatten(list(flat_inputs), spec)

        box_keep = self._box_keep_mask(boxes, _find_points(flat_inputs))
        label_ids = _resolve_label_ids(
            self._labels_getter(inputs), flat_inputs, boxes.shape[0]
        )

        flat_outputs = [
            _filter_boxes_and_labels(inpt, box_keep, label_ids) for inpt in flat_inputs
        ]
        return tree_unflatten(flat_outputs, spec)

    def _box_keep_mask(
        self, boxes: BoundingBoxes3D, points: PointCloud3D | None
    ) -> Tensor:
        """Compute the boolean keep-mask over boxes.

        Returns:
            1D boolean tensor ``[M]``; ``True`` where a box has at least
            ``min_points`` interior points (or where counting is not
            possible, in which case every box is kept).
        """
        raw = boxes.as_subclass(Tensor)
        m = raw.shape[0]
        # ``min_points == 0`` keeps every box, so skip the point-in-box
        # computation entirely; likewise when there are no boxes or no
        # point cloud to count against.
        if m == 0 or points is None or self.min_points == 0:
            return torch.ones(m, dtype=torch.bool, device=raw.device)
        pts = points.as_subclass(Tensor)
        # [N, M] membership; summing over points gives per-box counts. An
        # empty point cloud yields an all-zero count, dropping every box
        # unless min_points is 0.
        counts = points_in_boxes_3d(pts, raw, boxes.format).sum(dim=0)
        return counts >= self.min_points
