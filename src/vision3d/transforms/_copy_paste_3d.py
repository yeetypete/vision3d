"""3D copy-paste data augmentation with lazy object database."""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, override

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor, nn

from vision3d.ops import (
    box3d_corners,
    box3d_overlap,
    points_in_boxes_3d,
    points_in_boxes_3d_indices,
    project_to_image,
)
from vision3d.ops._points_in_boxes_3d import _extract_box_params
from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraImages,
    PointCloud3D,
)


@dataclass
class CameraCrop:
    """Image crop and convex-hull mask for one camera view of an object.

    Attributes:
        crop: Cropped image region ``[C, crop_h, crop_w]``.
        mask: Boolean hull mask ``[crop_h, crop_w]``.
        bbox: Bounding box in image coords ``(x_min, y_min, x_max, y_max)``.
    """

    crop: Tensor
    mask: Tensor
    bbox: tuple[int, int, int, int]


@dataclass
class ObjectEntry:
    """A single object extracted from a scene.

    Attributes:
        points: Points in scene frame ``[M, 3+C]``.
        box: Full box tensor ``[K]`` in its original format.
        label: Integer class label.
        camera_crops: Per-camera crops, or None when no camera data is
            available.  ``camera_crops[i]`` is None if the object is not
            visible in camera ``i``.
    """

    points: Tensor
    box: Tensor
    label: int
    camera_crops: list[CameraCrop | None] | None = field(default=None, repr=False)


def _convex_hull_2d(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Compute 2-D convex hull (Andrew's monotone chain).

    Pure-Python implementation for small point sets (≤ 8 points).

    Args:
        points: List of ``(x, y)`` pairs.

    Returns:
        Hull vertices in counter-clockwise order.
    """
    pts = sorted(points)

    # Lower hull
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2:
            o, a = lower[-2], lower[-1]
            if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) > 0:
                break
            lower.pop()
        lower.append(p)

    # Upper hull
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2:
            o, a = upper[-2], upper[-1]
            if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) > 0:
                break
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _fill_convex_polygon(
    vertices: list[tuple[float, float]],
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    """Rasterise a convex polygon into a boolean mask using Pillow.

    Args:
        vertices: CCW-ordered hull vertices ``(x, y)`` in crop-local coords.
        height: Mask height (pixels).
        width: Mask width (pixels).
        device: Device for the output mask.

    Returns:
        Boolean mask ``[height, width]``.
    """
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(vertices, fill=1)
    mask_np = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(height, width)
    return torch.from_numpy(mask_np.copy()).bool().to(device)


_HullMaskResult = tuple[Tensor, tuple[int, int, int, int], float]


def _project_boxes_to_camera(
    boxes: Tensor,
    fmt: BoundingBox3DFormat,
    extrinsic: Tensor,
    intrinsic: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project all box corners into a single camera at once.

    Args:
        boxes: ``[M, K]`` 3-D bounding boxes.
        fmt: Box format.
        extrinsic: ``[4, 4]`` lidar-to-camera.
        intrinsic: ``[3, 3]`` camera K.

    Returns:
        Tuple of ``(uv, depth)`` where ``uv`` is ``[M, 8, 2]`` pixel
        coordinates and ``depth`` is ``[M, 8]``.
    """
    corners = box3d_corners(boxes, fmt)  # [M, 8, 3]
    m = corners.shape[0]
    flat = corners.reshape(m * 8, 3)
    uv_flat, depth_flat = project_to_image(flat, extrinsic, intrinsic)
    return uv_flat.reshape(m, 8, 2), depth_flat.reshape(m, 8)


def _hull_mask_from_projected(
    uv: Tensor,
    depth: Tensor,
    img_h: int,
    img_w: int,
) -> _HullMaskResult | None:
    """Compute a convex-hull mask from pre-projected corners.

    Converts to plain Python immediately and does all geometry in floats
    to avoid torch dispatch overhead on tiny (8-element) tensors.

    Args:
        uv: ``[8, 2]`` pixel coordinates for one box.
        depth: ``[8]`` depth values for one box.
        img_h: Image height.
        img_w: Image width.

    Returns:
        ``(mask, bbox, depth)`` or ``None`` if the object is not visible.
    """
    uv_list: list[list[float]] = uv.tolist()
    depth_list: list[float] = depth.tolist()

    visible: list[tuple[float, float]] = []
    depth_sum = 0.0
    for i in range(len(depth_list)):
        if depth_list[i] > 0:
            visible.append((uv_list[i][0], uv_list[i][1]))
            depth_sum += depth_list[i]

    if len(visible) < 3:
        return None

    hull = _convex_hull_2d(visible)
    if len(hull) < 3:
        return None

    # Bounding rect of hull, clipped to image
    hull_xs = [p[0] for p in hull]
    hull_ys = [p[1] for p in hull]
    x_min = max(math.floor(min(hull_xs)), 0)
    y_min = max(math.floor(min(hull_ys)), 0)
    x_max = min(math.ceil(max(hull_xs)), img_w - 1)
    y_max = min(math.ceil(max(hull_ys)), img_h - 1)

    crop_h = y_max - y_min + 1
    crop_w = x_max - x_min + 1
    if crop_h < 1 or crop_w < 1:
        return None

    # Shift hull to crop-local coordinates and fill
    local_hull = [(x - x_min, y - y_min) for x, y in hull]
    mask = _fill_convex_polygon(local_hull, crop_h, crop_w, uv.device)

    mean_depth = depth_sum / len(visible)
    return mask, (x_min, y_min, x_max, y_max), mean_depth


def _batch_hull_masks(
    boxes: Tensor,
    fmt: BoundingBox3DFormat,
    extrinsic: Tensor,
    intrinsic: Tensor,
    img_h: int,
    img_w: int,
) -> list[_HullMaskResult | None]:
    """Compute hull masks for multiple boxes in one camera (batched projection).

    Args:
        boxes: ``[M, K]`` boxes.
        fmt: Box format.
        extrinsic: ``[4, 4]``.
        intrinsic: ``[3, 3]``.
        img_h: Image height.
        img_w: Image width.

    Returns:
        List of length ``M``, each element a ``(mask, bbox, depth)`` tuple
        or ``None``.
    """
    if boxes.shape[0] == 0:
        return []
    uv_all, depth_all = _project_boxes_to_camera(boxes, fmt, extrinsic, intrinsic)
    return [
        _hull_mask_from_projected(uv_all[i], depth_all[i], img_h, img_w)
        for i in range(boxes.shape[0])
    ]


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
        target_counts: Dict mapping integer class label to desired object
            count per sample. E.g. ``{0: 15, 1: 10}``.
        min_points: Minimum number of points an extracted object must
            have to be stored in the database. Default: ``5``.
        max_database_size: Maximum entries per class. None means
            unlimited. Default: ``None``.
        p: Probability of applying the augmentation. Default: ``1.0``.
    """

    def __init__(
        self,
        target_counts: dict[int, int],
        min_points: int = 5,
        max_database_size: int | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_counts = target_counts
        self.min_points = min_points
        self.max_database_size = max_database_size
        self.p = p

        self._database: dict[int, deque[ObjectEntry]] = defaultdict(
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
        for inp, tgt in zip(inputs, targets):
            self._extract_objects(inp, tgt)

        if torch.rand(1).item() >= self.p:
            return inputs, targets

        new_inputs = []
        new_targets = []
        for inp, tgt in zip(inputs, targets):
            new_inp, new_tgt = self._paste_objects(inp, tgt)
            new_inputs.append(new_inp)
            new_targets.append(new_tgt)

        return tuple(new_inputs), tuple(new_targets)

    def _has_camera_data(self, inputs: dict[str, Any]) -> bool:
        return "images" in inputs and "extrinsics" in inputs and "intrinsics" in inputs

    def _extract_objects(self, inputs: dict[str, Any], targets: dict[str, Any]) -> None:
        """Extract per-object point clouds and store in database."""
        points = inputs["points"]
        boxes = targets["boxes"]
        labels = targets.get("labels", torch.zeros(0, dtype=torch.long))

        if boxes.shape[0] == 0 or labels.shape[0] == 0:
            return

        raw_points = points.as_subclass(Tensor)
        raw_boxes = boxes.as_subclass(Tensor)
        fmt = boxes.format

        indices = points_in_boxes_3d_indices(raw_points, raw_boxes, fmt)

        # First pass: find objects with enough points
        valid: list[tuple[int, Tensor]] = []
        for j in range(raw_boxes.shape[0]):
            if j >= labels.shape[0]:
                break
            mask = indices == j
            obj_points = raw_points[mask]
            if obj_points.shape[0] >= self.min_points:
                valid.append((j, obj_points))

        # Batch camera crop extraction for all valid objects at once
        has_cameras = self._has_camera_data(inputs)
        camera_crops_map: dict[int, list[CameraCrop | None]] = {}
        if has_cameras and valid:
            camera_crops_map = self._extract_all_camera_crops(
                raw_boxes, fmt, inputs, [j for j, _ in valid]
            )

        for j, obj_points in valid:
            label = int(labels[j].item())
            entry = ObjectEntry(
                points=obj_points.detach().cpu(),
                box=raw_boxes[j].detach().cpu(),
                label=label,
                camera_crops=camera_crops_map.get(j),
            )
            self._database[label].append(entry)

    def _extract_all_camera_crops(
        self,
        boxes: Tensor,
        fmt: BoundingBox3DFormat,
        inputs: dict[str, Any],
        valid_indices: list[int],
    ) -> dict[int, list[CameraCrop | None]]:
        """Extract image crops for multiple objects from all camera views.

        Uses batched projection per camera to avoid per-object overhead.

        Args:
            boxes: All boxes ``[M, K]``.
            fmt: Box format.
            inputs: Input dict with camera data.
            valid_indices: Indices into ``boxes`` for objects that passed
                the min_points filter.

        Returns:
            Dict mapping box index to per-camera crop list.
        """
        images = inputs["images"]  # [N, C, H, W]
        extrinsics = inputs["extrinsics"]  # [N, 4, 4]
        intrinsics = inputs["intrinsics"]  # [N, 3, 3]
        n_cams = images.shape[0]
        img_h, img_w = images.shape[2], images.shape[3]

        if not valid_indices:
            return {}

        valid_boxes = boxes[valid_indices]  # [V, K]

        result: dict[int, list[CameraCrop | None]] = {
            j: [None] * n_cams for j in valid_indices
        }
        for cam_idx in range(n_cams):
            uv_all, depth_all = _project_boxes_to_camera(
                valid_boxes, fmt, extrinsics[cam_idx], intrinsics[cam_idx]
            )
            for vi, j in enumerate(valid_indices):
                hull_result = _hull_mask_from_projected(
                    uv_all[vi], depth_all[vi], img_h, img_w
                )
                if hull_result is None:
                    continue
                mask, (x_min, y_min, x_max, y_max), _depth = hull_result
                crop = images[cam_idx, :, y_min : y_max + 1, x_min : x_max + 1]
                result[j][cam_idx] = CameraCrop(
                    crop=crop.detach().cpu(),
                    mask=mask.detach().cpu(),
                    bbox=(x_min, y_min, x_max, y_max),
                )
        return result

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
        labels = targets.get("labels", torch.zeros(0, dtype=torch.long))
        fmt = boxes.format

        raw_points = points.as_subclass(Tensor)
        raw_boxes = boxes.as_subclass(Tensor)

        # Count existing objects per label
        existing_counts: dict[int, int] = {}
        for lbl in labels.tolist():
            existing_counts[lbl] = existing_counts.get(lbl, 0) + 1

        pasted_entries: list[ObjectEntry] = []
        pasted_boxes: list[Tensor] = []
        pasted_points: list[Tensor] = []
        pasted_labels: list[int] = []

        all_boxes = raw_boxes

        for label_id, target_count in self.target_counts.items():
            n_existing = existing_counts.get(label_id, 0)
            n_paste = max(0, target_count - n_existing)
            db = self._database.get(label_id)
            if not db or n_paste == 0:
                continue

            # Sample candidates (randperm for torch-seedable randomness)
            candidates = list(db)
            perm = torch.randperm(len(candidates))
            candidates = [candidates[i] for i in perm]

            for entry in candidates[:n_paste]:
                box = entry.box.to(raw_boxes.device)
                if all_boxes.shape[0] > 0:
                    overlap = box3d_overlap(box.unsqueeze(0), all_boxes, fmt)
                    if overlap.any():
                        continue

                pasted_entries.append(entry)
                pasted_boxes.append(box)
                pasted_points.append(entry.points.to(raw_points.device))
                pasted_labels.append(entry.label)
                all_boxes = torch.cat([all_boxes, box.unsqueeze(0)])

        if not pasted_boxes:
            return inputs, targets

        # Remove scene points inside pasted box regions
        pasted_boxes_tensor = torch.stack(pasted_boxes)
        remove_mask = points_in_boxes_3d(raw_points, pasted_boxes_tensor, fmt).any(
            dim=1
        )
        kept_points = raw_points[~remove_mask]

        # Concatenate
        new_points = torch.cat([kept_points, torch.cat(pasted_points)])
        new_boxes = torch.cat([raw_boxes, pasted_boxes_tensor])
        new_labels = torch.cat(
            [
                labels,
                torch.tensor(pasted_labels, dtype=labels.dtype, device=labels.device),
            ]
        )

        new_inputs: dict[str, Any] = {
            **inputs,
            "points": PointCloud3D(new_points),
        }
        new_targets: dict[str, Any] = {
            **targets,
            "boxes": BoundingBoxes3D(new_boxes, format=fmt),
            "labels": new_labels,
        }

        # Pass through class_names if present
        if "class_names" in targets and pasted_entries:
            # Look up names for pasted labels from existing mapping
            label_to_name: dict[int, str] = {}
            class_names = targets["class_names"]
            for i, name in enumerate(class_names):
                if i < labels.shape[0]:
                    label_to_name[int(labels[i].item())] = name
            new_targets["class_names"] = list(class_names) + [
                label_to_name.get(lbl, str(lbl)) for lbl in pasted_labels
            ]

        # Camera image paste
        if self._has_camera_data(inputs):
            new_images = self._paste_camera_images(
                inputs, raw_boxes, fmt, pasted_entries
            )
            if new_images is not None:
                new_inputs["images"] = new_images

        return new_inputs, new_targets

    def _paste_camera_images(
        self,
        inputs: dict[str, Any],
        existing_boxes: Tensor,
        fmt: BoundingBox3DFormat,
        pasted_entries: list[ObjectEntry],
    ) -> CameraImages | None:
        """Paste object image crops into camera views with depth-aware occlusion.

        Returns:
            Updated ``CameraImages`` or ``None`` if nothing to paste.
        """
        images = inputs["images"]  # [N, C, H, W] — cloned per-camera on write
        extrinsics = inputs["extrinsics"]  # [N, 4, 4]
        intrinsics = inputs["intrinsics"]  # [N, 3, 3]
        n_cams = images.shape[0]
        img_h, img_w = images.shape[2], images.shape[3]

        # Pre-compute centers (format-aware) once outside the camera loop
        pasted_box_stack = torch.stack([e.box for e in pasted_entries])
        p_centers, _, _ = _extract_box_params(pasted_box_stack, fmt)
        p_ones = torch.ones(
            p_centers.shape[0], 1, dtype=p_centers.dtype, device=p_centers.device
        )
        pasted_centers_hom = torch.cat([p_centers, p_ones], dim=-1)  # [P, 4]

        has_existing = existing_boxes.shape[0] > 0
        e_centers_hom = torch.zeros(
            0, 4, dtype=existing_boxes.dtype, device=existing_boxes.device
        )
        if has_existing:
            e_centers, _, _ = _extract_box_params(existing_boxes, fmt)
            e_ones = torch.ones(
                e_centers.shape[0], 1, dtype=e_centers.dtype, device=e_centers.device
            )
            e_centers_hom = torch.cat([e_centers, e_ones], dim=-1)

        any_pasted = False
        cloned = False

        for cam_idx in range(n_cams):
            ext = extrinsics[cam_idx]  # [4, 4]
            K = intrinsics[cam_idx]  # [3, 3]

            # Depths of pasted objects in this camera
            paste_depths = (ext @ pasted_centers_hom.T).T[:, 2]  # [P]
            order = paste_depths.argsort(descending=True)

            # Batched hull masks for existing scene boxes (for occlusion)
            existing_masks: list[_HullMaskResult | None] = []
            existing_depths: list[float] = []
            if has_existing:
                existing_depths_t = (ext @ e_centers_hom.T).T[:, 2]
                existing_depths = existing_depths_t.tolist()
                existing_masks = _batch_hull_masks(
                    existing_boxes, fmt, ext, K, img_h, img_w
                )

            # Batched hull masks for pasted objects (target projection)
            pasted_masks = _batch_hull_masks(
                pasted_box_stack, fmt, ext, K, img_h, img_w
            )

            for idx in order:
                idx_int = int(idx.item())
                entry = pasted_entries[idx_int]
                if entry.camera_crops is None:
                    continue
                if cam_idx >= len(entry.camera_crops):
                    continue
                cam_crop = entry.camera_crops[cam_idx]
                if cam_crop is None:
                    continue

                result = pasted_masks[idx_int]
                if result is None:
                    continue

                target_mask, (tx_min, ty_min, tx_max, ty_max), _target_depth = result
                target_h = ty_max - ty_min + 1
                target_w = tx_max - tx_min + 1

                # Resize stored crop and mask to target size
                src_crop = cam_crop.crop.unsqueeze(0).float()  # [1, C, sh, sw]
                resized_crop = F.interpolate(
                    src_crop,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # [C, th, tw]

                src_mask = cam_crop.mask.unsqueeze(0).unsqueeze(0).float()
                resized_mask = (
                    F.interpolate(src_mask, size=(target_h, target_w), mode="nearest")
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )

                # Intersect target hull with resized source mask
                paste_mask = target_mask & resized_mask

                # Depth-aware occlusion: subtract closer existing boxes
                paste_depth = float(paste_depths[idx_int].item())
                for e_idx in range(len(existing_masks)):
                    if (
                        existing_depths[e_idx] <= 0
                        or existing_depths[e_idx] >= paste_depth
                    ):
                        continue
                    e_result = existing_masks[e_idx]
                    if e_result is None:
                        continue
                    _e_mask, (ex_min, ey_min, ex_max, ey_max), _e_depth = e_result

                    # Compute overlap region
                    ox_min = max(tx_min, ex_min)
                    oy_min = max(ty_min, ey_min)
                    ox_max = min(tx_max, ex_max)
                    oy_max = min(ty_max, ey_max)
                    if ox_min > ox_max or oy_min > oy_max:
                        continue

                    # Subtract overlapping part of existing mask
                    overlap_h = oy_max - oy_min + 1
                    overlap_w = ox_max - ox_min + 1
                    p_oy = oy_min - ty_min
                    p_ox = ox_min - tx_min
                    e_oy = oy_min - ey_min
                    e_ox = ox_min - ex_min
                    paste_mask[
                        p_oy : p_oy + overlap_h, p_ox : p_ox + overlap_w
                    ] &= ~_e_mask[e_oy : e_oy + overlap_h, e_ox : e_ox + overlap_w]

                if not paste_mask.any():
                    continue

                # Clone on first write to avoid mutating the input
                if not cloned:
                    images = images.clone()
                    cloned = True

                # Paste into image
                region = images[cam_idx, :, ty_min : ty_max + 1, tx_min : tx_max + 1]
                region[:, paste_mask] = resized_crop[:, paste_mask].to(region.dtype)
                any_pasted = True

        if not any_pasted:
            return None
        return CameraImages(images)
