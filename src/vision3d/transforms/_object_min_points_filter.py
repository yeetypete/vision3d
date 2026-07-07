"""Minimum-points filtering for ground-truth boxes."""

from typing import Any, override

import torch
from torch import Tensor

from vision3d.ops import points_in_boxes_3d
from vision3d.tensors import BoundingBoxes3D, PointCloud3D

from ._transform import Transform


class ObjectMinPointsFilter(Transform):
    """Drop ground-truth boxes that enclose fewer than ``min_points`` points.

    Counts the points inside each box and removes boxes whose interior
    point count is strictly below ``min_points``; labels in ``targets`` are
    filtered in sync. The point cloud and every other sample entry pass
    through unchanged. Counting is format-agnostic (incl. 9-DoF); a point in
    the overlap of several boxes is counted once for **each** box it lies in.

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

    Args:
        min_points: Minimum number of interior points a box must contain to
            survive. A box is dropped when its count is strictly less than
            this. ``0`` keeps every box; ``1`` keeps every box with at least
            one point.
    """

    def __init__(self, min_points: int) -> None:
        super().__init__()
        if isinstance(min_points, bool) or not isinstance(min_points, int):
            msg = "`min_points` should be a non-negative int."
            raise TypeError(msg)
        if min_points < 0:
            msg = "`min_points` should be a non-negative int."
            raise ValueError(msg)
        self.min_points = min_points

    @override
    def forward(self, *inputs: Any) -> Any:
        """Filter boxes with too few interior points.

        Accepts both a single sample dict and an ``(inputs, targets)`` pair.
        Boxes (and synced labels) live in the sample/targets dict; the point
        cloud is read from wherever it appears (the same dict, or the
        ``inputs`` dict of a pair) and is never modified.

        Returns:
            Filtered sample in the same structure as the input.
        """
        if len(inputs) == 1:
            return self._filter_sample(inputs[0])
        inputs_dict, targets_dict = inputs
        points = inputs_dict.get("points")
        return inputs_dict, self._filter_targets(targets_dict, points)

    def _filter_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        out = dict(sample)
        self._apply_box_mask(out, out.get("points"))
        return out

    def _filter_targets(
        self, targets: dict[str, Any], points: PointCloud3D | None
    ) -> dict[str, Any]:
        out = dict(targets)
        self._apply_box_mask(out, points)
        return out

    def _apply_box_mask(self, d: dict[str, Any], points: PointCloud3D | None) -> None:
        """Filter boxes and labels in-place by interior point count."""
        if "boxes" not in d:
            return
        boxes = d["boxes"]
        keep = self._box_keep_mask(boxes, points)
        d["boxes"] = BoundingBoxes3D(
            boxes.as_subclass(Tensor)[keep], format=boxes.format
        )
        if "labels" in d:
            d["labels"] = d["labels"][keep]

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
        if m == 0 or points is None:
            return torch.ones(m, dtype=torch.bool, device=raw.device)
        pts = points.as_subclass(Tensor)
        # [N, M] membership; summing over points gives per-box counts. An
        # empty point cloud yields an all-zero count, dropping every box
        # unless min_points is 0.
        counts = points_in_boxes_3d(pts, raw, boxes.format).sum(dim=0)
        return counts >= self.min_points
