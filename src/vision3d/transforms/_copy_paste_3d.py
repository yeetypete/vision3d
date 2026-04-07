"""3D copy-paste data augmentation with lazy object database."""

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor, nn

from vision3d.ops import (
    box3d_overlap_bev,
    points_in_boxes_3d,
    points_in_boxes_3d_indices,
)
from vision3d.tensors import BoundingBoxes3D, PointCloud3D


@dataclass
class ObjectEntry:
    """A single object extracted from a scene.

    Attributes:
        points: Points in scene frame ``[M, 3+C]``.
        box: Full box tensor ``[K]`` in its original format.
        class_name: The class name string.
    """

    points: Tensor
    box: Tensor
    class_name: str


class CopyPaste3D(nn.Module):
    """Batch-level 3D copy-paste data augmentation.

    Maintains a lazy object database that grows as batches pass through.
    For each sample, pastes additional objects from the database to reach
    a target count per class. Objects are pasted at their original scene
    position from the source frame.

    Operates on collated batches ``(tuple_of_inputs, tuple_of_targets)``,
    not individual samples. Each instance should be used with only one
    dataset to avoid cross-contamination.

    Args:
        target_counts: Dict mapping class name to desired object count
            per sample. E.g. ``{"Car": 15, "Pedestrian": 10}``.
        min_points: Minimum number of points an extracted object must
            have to be stored in the database. Default: ``5``.
        max_database_size: Maximum entries per class. None means
            unlimited. Default: ``None``.
        p: Probability of applying the augmentation. Default: ``1.0``.
    """

    def __init__(
        self,
        target_counts: dict[str, int],
        min_points: int = 5,
        max_database_size: int | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_counts = target_counts
        self.min_points = min_points
        self.max_database_size = max_database_size
        self.p = p

        self._database: dict[str, deque[ObjectEntry]] = defaultdict(
            lambda: deque(maxlen=self.max_database_size)
        )

    @override
    def forward(
        self,
        inputs: tuple[dict[str, Any], ...],
        targets: tuple[dict[str, Any], ...],
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        """Apply copy-paste augmentation to a collated batch.

        Args:
            inputs: Tuple of input dicts from collation.
            targets: Tuple of target dicts from collation.

        Returns:
            Modified ``(inputs, targets)`` tuples.
        """
        # Extract objects from current batch into database
        for inp, tgt in zip(inputs, targets):
            self._extract_objects(inp, tgt)

        # Skip pasting with probability 1-p
        if torch.rand(1).item() >= self.p:
            return inputs, targets

        # Paste objects into each sample
        new_inputs = []
        new_targets = []
        for inp, tgt in zip(inputs, targets):
            new_inp, new_tgt = self._paste_objects(inp, tgt)
            new_inputs.append(new_inp)
            new_targets.append(new_tgt)

        return tuple(new_inputs), tuple(new_targets)

    def _extract_objects(self, inputs: dict[str, Any], targets: dict[str, Any]) -> None:
        """Extract per-object point clouds and store in database."""
        points = inputs["points"]
        boxes = targets["boxes"]
        class_names = targets.get("class_names", [])

        if boxes.shape[0] == 0 or len(class_names) == 0:
            return

        raw_points = points.as_subclass(Tensor)
        raw_boxes = boxes.as_subclass(Tensor)
        fmt = boxes.format

        indices = points_in_boxes_3d_indices(raw_points, raw_boxes, fmt)

        for j in range(raw_boxes.shape[0]):
            if j >= len(class_names):
                break
            mask = indices == j
            obj_points = raw_points[mask]

            if obj_points.shape[0] < self.min_points:
                continue

            entry = ObjectEntry(
                points=obj_points.detach().cpu(),
                box=raw_boxes[j].detach().cpu(),
                class_name=class_names[j],
            )
            self._database[class_names[j]].append(entry)

    def _paste_objects(
        self,
        inputs: dict[str, Any],
        targets: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Paste objects from database into a single sample.

        Returns:
            Modified ``(inputs, targets)`` dicts.
        """
        points = inputs["points"]
        boxes = targets["boxes"]
        class_names = list(targets.get("class_names", []))
        labels = targets.get("labels", torch.zeros(0, dtype=torch.long))
        fmt = boxes.format

        raw_points = points.as_subclass(Tensor)
        raw_boxes = boxes.as_subclass(Tensor)

        # Count existing objects per class
        existing_counts: dict[str, int] = {}
        for name in class_names:
            existing_counts[name] = existing_counts.get(name, 0) + 1

        pasted_boxes: list[Tensor] = []
        pasted_points: list[Tensor] = []
        pasted_names: list[str] = []

        all_boxes = raw_boxes

        for cls_name, target_count in self.target_counts.items():
            n_existing = existing_counts.get(cls_name, 0)
            n_paste = max(0, target_count - n_existing)
            db = self._database.get(cls_name)
            if not db or n_paste == 0:
                continue

            # Sample candidates (shuffle to avoid always picking the same ones)
            candidates = list(db)
            random.shuffle(candidates)

            for entry in candidates[:n_paste]:
                # Check collision at the object's original position
                if all_boxes.shape[0] > 0:
                    overlap = box3d_overlap_bev(entry.box.unsqueeze(0), all_boxes, fmt)
                    if overlap.any():
                        continue

                pasted_boxes.append(entry.box)
                pasted_points.append(entry.points)
                pasted_names.append(cls_name)
                all_boxes = torch.cat([all_boxes, entry.box.unsqueeze(0)])

        if not pasted_boxes:
            return inputs, targets

        # Remove scene points inside pasted box regions
        pasted_boxes_tensor = torch.stack(pasted_boxes)
        remove_mask = points_in_boxes_3d(raw_points, pasted_boxes_tensor, fmt).any(
            dim=1
        )
        kept_points = raw_points[~remove_mask]

        # Concatenate pasted points
        all_pasted_points = torch.cat(pasted_points)
        new_points = torch.cat([kept_points, all_pasted_points])

        # Concatenate boxes and labels
        new_boxes = torch.cat([raw_boxes, pasted_boxes_tensor])
        new_labels = torch.cat(
            [
                labels,
                torch.arange(
                    labels.shape[0],
                    labels.shape[0] + len(pasted_names),
                    dtype=labels.dtype,
                ),
            ]
        )
        new_class_names = class_names + pasted_names

        new_inputs = {**inputs, "points": PointCloud3D(new_points)}
        new_targets = {
            **targets,
            "boxes": BoundingBoxes3D(new_boxes, format=fmt),
            "labels": new_labels,
            "class_names": new_class_names,
        }

        return new_inputs, new_targets
