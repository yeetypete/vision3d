"""Shared scaffolding for transforms that drop boxes by a keep-mask."""

from typing import Any

from torch import Tensor

from vision3d.tensors import BoundingBoxes3D, PointCloud3D

from ._transform import Transform


class _BoxFilterTransform(Transform):
    """Base for transforms that drop boxes (and synced labels) by a mask.

    Subclasses implement :meth:`_box_keep_mask` to compute a boolean
    keep-mask over the boxes; this base owns the shared "rebuild
    ``BoundingBoxes3D`` and sync ``labels`` in step" logic so the two
    stay coupled in one place. The optional ``points`` argument is passed
    through for subclasses whose mask depends on the point cloud;
    subclasses that filter boxes by geometry alone ignore it.
    """

    def _box_keep_mask(
        self, boxes: BoundingBoxes3D, points: PointCloud3D | None = None
    ) -> Tensor:
        """Compute the boolean keep-mask over boxes. Must be overridden.

        Returns:
            1D boolean tensor ``[M]``; ``True`` where a box is kept.
        """
        raise NotImplementedError

    def _apply_box_mask(
        self, d: dict[str, Any], points: PointCloud3D | None = None
    ) -> None:
        """Filter ``boxes`` and synced ``labels`` in-place by the keep-mask."""
        if "boxes" not in d:
            return
        boxes = d["boxes"]
        keep = self._box_keep_mask(boxes, points)
        d["boxes"] = BoundingBoxes3D(
            boxes.as_subclass(Tensor)[keep], format=boxes.format
        )
        if "labels" in d:
            d["labels"] = d["labels"][keep]
