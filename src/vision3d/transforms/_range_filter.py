"""Range-based filtering for points and boxes."""

from typing import Any, override

import torch
from torch import Tensor

from vision3d.ops._points_in_boxes_3d import _extract_box_params
from vision3d.tensors import BoundingBoxes3D, PointCloud3D

from ._transform import Transform


class RangeFilter3D(Transform):
    """Drop points and boxes outside an axis-aligned 3D region.

    Replaces MMDetection3D's separate ``ObjectRangeFilter`` and
    ``PointsRangeFilter`` in one transform. Points are filtered by
    their xyz coordinates; boxes are filtered by their **center**
    (format-agnostic via ``_extract_box_params``). Labels in
    ``targets`` are filtered in sync with boxes.

    Applied after spatial augmentations (rotate / scale / translate
    can push data out of the sensor range) and before the model sees
    the data.

    Unlike the scene transforms, this class inherits from
    :class:`nn.Module` rather than :class:`Transform` because it
    needs to couple boxes and labels — the standard per-leaf dispatch
    cannot handle filtering two tensors with the same mask.

    Args:
        point_cloud_range: Axis-aligned bounds
            ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
    """

    def __init__(self, point_cloud_range: tuple[float, ...]) -> None:
        super().__init__()
        if len(point_cloud_range) != 6:
            msg = "point_cloud_range must have 6 elements (x_min, y_min, z_min, x_max, y_max, z_max)"
            raise ValueError(msg)
        self.point_cloud_range = tuple(point_cloud_range)

    @override
    def forward(self, *inputs: Any) -> Any:
        """Filter points and boxes outside the configured range.

        Accepts both a single sample dict and an
        ``(inputs, targets)`` pair.

        Returns:
            Filtered sample in the same structure as the input.
        """
        if len(inputs) == 1:
            return self._filter_sample(inputs[0])
        inputs_dict, targets_dict = inputs
        return self._filter_inputs(inputs_dict), self._filter_targets(targets_dict)

    def _filter_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        out = dict(inputs)
        if "points" in out:
            out["points"] = self._filter_points(out["points"])
        return out

    def _filter_targets(self, targets: dict[str, Any]) -> dict[str, Any]:
        out = dict(targets)
        if "boxes" in out:
            boxes = out["boxes"]
            keep = self._box_keep_mask(boxes)
            out["boxes"] = BoundingBoxes3D(
                boxes.as_subclass(Tensor)[keep], format=boxes.format
            )
            if "labels" in out:
                out["labels"] = out["labels"][keep]
        return out

    def _filter_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        out = dict(sample)
        if "points" in out:
            out["points"] = self._filter_points(out["points"])
        if "boxes" in out:
            boxes = out["boxes"]
            keep = self._box_keep_mask(boxes)
            out["boxes"] = BoundingBoxes3D(
                boxes.as_subclass(Tensor)[keep], format=boxes.format
            )
            if "labels" in out:
                out["labels"] = out["labels"][keep]
        return out

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
        centers, _, _ = _extract_box_params(raw, boxes.format)
        min_bound = torch.tensor(
            self.point_cloud_range[:3], dtype=raw.dtype, device=raw.device
        )
        max_bound = torch.tensor(
            self.point_cloud_range[3:], dtype=raw.dtype, device=raw.device
        )
        return ((centers >= min_bound) & (centers < max_bound)).all(dim=-1)
