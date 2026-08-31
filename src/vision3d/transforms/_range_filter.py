"""Range-based filtering for points and boxes."""

from collections.abc import Callable
from typing import Any, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from vision3d.ops import extract_box3d_params
from vision3d.tensors import BoundingBoxes3D, PointCloud3D

from ._transform import Transform
from ._utils import (
    _filter_boxes_and_labels,
    _find_boxes,
    _parse_labels_getter,
    _resolve_label_ids,
)


class RangeFilter3D(Transform):
    """Drop points and boxes outside an axis-aligned 3D region.

    Points are filtered by their xyz coordinates; boxes are filtered by
    their **center** (format-agnostic), and any per-box ``labels`` are
    filtered in sync. The sample must hold at most one ``BoundingBoxes3D``
    set (a single keep-mask cannot span box sets of differing length).

    **Must** be applied after spatial augmentations (rotate / scale /
    translate can push data out of the sensor range) and before the
    model sees the data.

    Args:
        point_cloud_range: Axis-aligned bounds
            ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
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
            ValueError: If called with no inputs. If the sample holds more
                than one box set. If the sample has boxes but the default
                ``labels_getter`` cannot find a labels tensor. If
                ``labels_getter`` returns a tensor that is not a leaf of the
                sample (e.g. a copy, view, or nested tensor). If a returned
                label tensor's length does not match the number of boxes.
        """
        # Unlike most transforms, forward is hand-rolled rather than routed
        # through the base per-leaf ``transform()`` loop. Points and boxes have
        # distinguishing tensor types and would dispatch fine, but labels are
        # plain tensors with no distinguishing type, so they can only be located
        # via ``labels_getter`` against the nested structure -- which is gone
        # once the tree is flattened for the base loop. The base hooks
        # (check_inputs, make_params) do not fire here.
        if not inputs:
            msg = "RangeFilter3D.forward requires at least one input sample"
            raise ValueError(msg)
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        boxes = _find_boxes(flat_inputs)

        # Labels are filtered in sync with the boxes, so we only consult the
        # ``labels_getter`` when the sample carries boxes. This keeps a
        # points-only sample working under the default getter without forcing
        # ``labels_getter=None``. The label tensors are matched to their leaf by
        # identity, so the keep-mask applies wherever they live.
        if boxes is None:
            box_keep = None
            label_ids: set[int] = set()
        else:
            box_keep = self._box_keep_mask(boxes)
            label_ids = _resolve_label_ids(
                self._labels_getter(inputs), flat_inputs, boxes.shape[0]
            )

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
        return _filter_boxes_and_labels(inpt, box_keep, label_ids)

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
