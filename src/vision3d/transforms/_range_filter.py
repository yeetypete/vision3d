"""Range-based filtering for points and boxes."""

from collections.abc import Callable
from typing import Any, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from vision3d.ops import extract_box3d_params
from vision3d.tensors import BoundingBoxes3D, PointCloud3D

from ._transform import Transform
from ._utils import _find_boxes, _parse_labels_getter


class RangeFilter3D(Transform):
    """Drop points and boxes outside an axis-aligned 3D region.

    Points are filtered by their xyz coordinates; boxes are filtered by
    their **center** (format-agnostic), and any per-box ``labels`` are
    filtered in sync.

    **Must** be applied after spatial augmentations (rotate / scale /
    translate can push data out of the sensor range) and before the
    model sees the data.

    Args:
        point_cloud_range: Axis-aligned bounds
            ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
        labels_getter: How to locate the per-box label tensor filtered in
            sync with boxes. ``"default"`` finds it under a case-insensitive
            ``"labels"`` key; pass a callable for a custom location, or
            ``None`` to filter boxes without touching any labels. Default:
            ``"default"``. A callable must return the label tensor(s) stored
            in the sample, not a copy or view — labels are matched to their
            leaf by identity.
    """

    def __init__(
        self,
        point_cloud_range: tuple[float, ...],
        labels_getter: str | Callable[[Any], Any] | None = "default",
    ) -> None:
        super().__init__()
        if len(point_cloud_range) != 6:
            msg = "point_cloud_range must have 6 elements (x_min, y_min, z_min, x_max, y_max, z_max)"
            raise ValueError(msg)
        self.point_cloud_range = tuple(point_cloud_range)
        self.labels_getter = labels_getter
        self._labels_getter = _parse_labels_getter(labels_getter)

    @override
    def forward(self, *inputs: Any) -> Any:
        """Filter points and boxes outside the configured range.

        Accepts both a single sample dict and an ``(inputs, targets)`` pair.

        Returns:
            Filtered sample in the same structure as the input.

        Raises:
            ValueError: If the sample holds more than one box set, or if
                ``labels_getter`` returns a tensor that is not stored in the
                sample (e.g. a copy or view).
        """
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        boxes = _find_boxes(flat_inputs)
        box_keep = None if boxes is None else self._box_keep_mask(boxes)

        # Labels are matched to their leaf by identity, so the keep-mask
        # applies wherever they live; needs boxes to define the mask.
        label_ids: set[int] = set()
        if boxes is not None:
            labels = self._labels_getter(inputs)
            if labels is not None:
                labels = (labels,) if isinstance(labels, Tensor) else labels
                leaf_ids = {id(leaf) for leaf in flat_inputs}
                for label in labels:
                    if id(label) not in leaf_ids:
                        msg = (
                            "`labels_getter` must return the label tensor(s) "
                            "stored in the sample, not a copy or view"
                        )
                        raise ValueError(msg)
                label_ids = {id(label) for label in labels}

        flat_outputs = [
            self._filter_leaf(inpt, box_keep, label_ids) for inpt in flat_inputs
        ]
        return tree_unflatten(flat_outputs, spec)

    def _filter_leaf(
        self, inpt: Any, box_keep: Tensor | None, label_ids: set[int]
    ) -> Any:
        """Filter a single flattened leaf according to its type.

        Returns:
            The point cloud filtered by coordinate, boxes/labels filtered
            by the box keep-mask, or ``inpt`` unchanged.
        """
        if isinstance(inpt, PointCloud3D):
            return self._filter_points(inpt)
        if box_keep is None:
            return inpt
        if isinstance(inpt, BoundingBoxes3D):
            return BoundingBoxes3D(
                inpt.as_subclass(Tensor)[box_keep], format=inpt.format
            )
        if id(inpt) in label_ids:
            return inpt[box_keep]
        return inpt

    def _filter_points(self, points: PointCloud3D) -> PointCloud3D:
        pts = points.as_subclass(Tensor)
        min_bound = torch.tensor(
            self.point_cloud_range[:3], dtype=pts.dtype, device=pts.device
        )
        max_bound = torch.tensor(
            self.point_cloud_range[3:], dtype=pts.dtype, device=pts.device
        )
        keep = ((pts[:, :3] >= min_bound) & (pts[:, :3] < max_bound)).all(dim=-1)
        return PointCloud3D(pts[keep])

    def _box_keep_mask(self, boxes: BoundingBoxes3D) -> Tensor:
        raw = boxes.as_subclass(Tensor)
        centers, _, _ = extract_box3d_params(raw, boxes.format)
        min_bound = torch.tensor(
            self.point_cloud_range[:3], dtype=raw.dtype, device=raw.device
        )
        max_bound = torch.tensor(
            self.point_cloud_range[3:], dtype=raw.dtype, device=raw.device
        )
        return ((centers >= min_bound) & (centers < max_bound)).all(dim=-1)
