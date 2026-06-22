"""3D copy-paste data augmentation with lazy object database."""

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, override

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.tv_tensors import TVTensor

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
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)
from vision3d.transforms._transform import Transform


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
        points: Points in scene frame ``[M, 3+C]``, or ``None`` for
            camera-only entries.
        box: Full box tensor ``[K]`` in its original format.
        label: Integer class label.
        camera_crops: Per-camera crops, or None when no camera data is
            available.  ``camera_crops[i]`` is None if the object is not
            visible in camera ``i``.
    """

    points: Tensor | None
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


def _apply_offset(
    boxes: Tensor,
    fmt: BoundingBox3DFormat,
    offsets: Tensor,
) -> Tensor:
    """Return a copy of *boxes* translated by per-box *offsets*.

    Args:
        boxes: ``[M, K]`` 3-D bounding boxes.
        fmt: Box format.
        offsets: ``[M, 3]`` per-box ``(x, y, z)`` offsets in scene units.

    Returns:
        A new ``[M, K]`` tensor with the box positions shifted.
    """
    out = boxes.clone()
    out[:, :3] += offsets
    if fmt is BoundingBox3DFormat.XYZXYZ:
        out[:, 3:6] += offsets  # also shift the max corner
    return out


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


_AxisRange = tuple[float, float]
_OffsetRange = _AxisRange | Sequence[_AxisRange]
_OffsetStd = float | Sequence[float | None] | None


def _normalize_offset_range(offset_range: _OffsetRange) -> list[_AxisRange]:
    """Expand *offset_range* into one ``(min, max)`` pair per x/y/z axis.

    Accepts a single ``(min, max)`` pair (broadcast to all three axes) or a
    sequence of three ``(min, max)`` pairs.

    Args:
        offset_range: Offset range specification.

    Returns:
        List of three ``(min, max)`` float pairs.

    Raises:
        TypeError: If a per-axis entry is a scalar rather than a pair.
        ValueError: If the shape is neither a pair nor three pairs, or any
            pair has ``min > max``.
    """
    seq = list(offset_range)
    msg_per_axis = "Each per-axis `offset_range` entry must be a (min, max) pair."
    if (
        len(seq) == 2
        and isinstance(seq[0], (int, float))
        and isinstance(seq[1], (int, float))
    ):
        pair = (seq[0], seq[1])
        ranges = [pair, pair, pair]
    elif len(seq) == 3:
        ranges = []
        for axis in seq:
            if isinstance(axis, (int, float)):
                raise TypeError(msg_per_axis)
            entry = tuple(axis)
            if len(entry) != 2:
                raise ValueError(msg_per_axis)
            ranges.append((entry[0], entry[1]))
    else:
        msg = (
            "`offset_range` should be a (min, max) pair applied to all axes "
            "or a 3-tuple of (min, max) pairs."
        )
        raise ValueError(msg)
    for lo, hi in ranges:
        if lo > hi:
            msg = "`offset_range` min must not exceed max."
            raise ValueError(msg)
    return ranges


def _normalize_offset_std(offset_std: _OffsetStd) -> list[float | None]:
    """Expand *offset_std* into one standard deviation (or ``None``) per axis.

    Args:
        offset_std: A single value shared across axes, ``None``, or a 3-tuple
            of per-axis values (each a float or ``None``).

    Returns:
        List of three ``float | None`` values.

    Raises:
        ValueError: If a sequence is given with a length other than three.
    """
    if offset_std is None:
        return [None, None, None]
    if isinstance(offset_std, (int, float)):
        return [offset_std, offset_std, offset_std]
    seq = list(offset_std)
    if len(seq) != 3:
        msg = "`offset_std` should be a float, None, or a 3-tuple."
        raise ValueError(msg)
    return list(seq)


class CopyPaste3D(Transform):
    """Batch-level 3D copy-paste data augmentation.

    Maintains a lazy object database that grows as batches pass through.
    For each sample, pastes additional objects from the database to reach
    a target count per class. Objects are pasted at their original scene
    position from the source frame.

    Operates on collated batches ``(tuple_of_inputs, tuple_of_targets)``,
    not individual samples. Each instance should be used with only one
    dataset to avoid cross-contamination.

    :class:`CopyPaste3D` **must** be the first
    transform in any pipeline, before any 3D spatial transform
    (:class:`RandomFlip3D`, :class:`RandomRotate3D`,
    :class:`RandomScale3D`, :class:`RandomTranslate3D`). Pasted objects
    are extracted and re-inserted in the source-frame geometry of the
    scene they came from. If a scene transform has already mutated the
    frame, the pasted objects will disagree with the rest of the scene
    and the resulting boxes/points will be inconsistent.

    Args:
        target_counts: Dict mapping integer class label to desired object
            count per sample. E.g. ``{0: 15, 1: 10}``.
        min_points: Minimum number of points an extracted object must
            have to be stored in the database. Default: ``5``.
        max_database_size: Maximum entries per class. None means
            unlimited. Default: ``None``.
        offset_range: Random position offset applied per pasted object (to its
            box and points) before the overlap check, so placements stay
            collision-free at the jittered pose. Either a single ``(min, max)``
            interval, in scene units, applied independently to the x, y, and z
            axes, or a 3-tuple of per-axis intervals
            ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``. ``(0.0, 0.0)``
            disables jittering. For each object up to ``max_jitter_attempts``
            offsets are drawn and the first collision-free one is used; if none
            is collision-free the object falls back to its original (un-jittered)
            pose. Default: ``(0.0, 0.0)``.
        offset_std: Standard deviation, in scene units, of the offset
            distribution. ``None`` (the default) draws offsets uniformly across
            ``offset_range``; a positive value draws from a normal distribution
            centred on each range's midpoint and truncated to it. May be a
            single value shared across axes or a 3-tuple of per-axis values
            (each a float, or ``None`` to keep that axis uniform).
            Default: ``None``.
        max_jitter_attempts: Number of jittered positions to try per object
            before falling back to its original (un-jittered) pose. Larger
            values raise the chance of placing a genuinely jittered object in a
            crowded scene. Ignored when jittering is disabled. Default: ``5``.
        p: Probability of applying the augmentation. Default: ``1.0``.

    Note:
        Where ``offset_std`` is set, the 68-95-99.7 rule applies per axis: ~68%
        of offsets fall within ±1 std of the midpoint, ~95% within ±2 std, and
        ~99.7% within ±3 std. This holds only when ``offset_range`` is wide
        enough not to clip the tails, so choose each range as roughly ±2 to
        ±3 std (e.g. ``offset_std=0.5`` with ``offset_range=(-1.5, 1.5)``). A
        narrower range flattens the bell back toward uniform.
    """

    def __init__(
        self,
        target_counts: dict[int, int],
        min_points: int = 5,
        max_database_size: int | None = None,
        offset_range: _OffsetRange = (0.0, 0.0),
        offset_std: _OffsetStd = None,
        max_jitter_attempts: int = 5,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= p <= 1.0):
            msg = "`p` should be a float in [0.0, 1.0]."
            raise ValueError(msg)
        if min_points < 1:
            msg = "`min_points` should be a positive integer."
            raise ValueError(msg)
        if max_database_size is not None and max_database_size < 1:
            msg = "`max_database_size` should be a positive integer or None."
            raise ValueError(msg)
        if max_jitter_attempts < 1:
            msg = "`max_jitter_attempts` should be a positive integer."
            raise ValueError(msg)
        self.offset_range = _normalize_offset_range(offset_range)
        self.offset_std = _normalize_offset_std(offset_std)
        for (lo, hi), std in zip(self.offset_range, self.offset_std):
            if std is not None and std <= 0:
                msg = "`offset_std` values should be positive or None."
                raise ValueError(msg)
            if std is not None and lo == hi:
                msg = (
                    "`offset_std` has no effect on an axis whose `offset_range` "
                    "is degenerate (min == max); widen the range or use None "
                    "for that axis."
                )
                raise ValueError(msg)
        self.target_counts = target_counts
        self.min_points = min_points
        self.max_database_size = max_database_size
        self.max_jitter_attempts = max_jitter_attempts
        self._jitter = any(r != (0.0, 0.0) for r in self.offset_range)
        self.p = p

        self._database: dict[int, deque[ObjectEntry]] = defaultdict(
            lambda: deque(maxlen=self.max_database_size)
        )

    @override
    def forward(self, *inputs: Any) -> Any:
        """Apply copy-paste augmentation to a collated batch.

        Accepts any pytree structure containing
        :class:`~vision3d.tensors.PointCloud3D`,
        :class:`~vision3d.tensors.BoundingBoxes3D`, and optionally camera
        tensors and plain-tensor labels.

        Returns:
            The same pytree structure with modified leaves.
        """
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        self.check_inputs(flat_inputs)

        batch_inputs, batch_targets = self._extract_samples(flat_inputs)

        for inp, tgt in zip(batch_inputs, batch_targets):
            self._extract_objects(inp, tgt)

        if torch.rand(1).item() >= self.p:
            return tree_unflatten(flat_inputs, spec)

        output_inputs = []
        output_targets = []
        for inp, tgt in zip(batch_inputs, batch_targets):
            new_inp, new_tgt = self._paste_objects(inp, tgt)
            output_inputs.append(new_inp)
            output_targets.append(new_tgt)

        self._insert_outputs(flat_inputs, output_inputs, output_targets)
        return tree_unflatten(flat_inputs, spec)

    def _extract_samples(
        self, flat_inputs: list[Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Group flat pytree leaves into per-sample input and target dicts.

        Returns:
            ``(batch_inputs, batch_targets)``: Lists of per-sample dicts.

        Raises:
            TypeError: If required types are missing or counts don't match.
        """
        points: list[PointCloud3D] = []
        boxes: list[BoundingBoxes3D] = []
        images: list[CameraImages] = []
        extrinsics: list[CameraExtrinsics] = []
        intrinsics: list[CameraIntrinsics] = []
        labels: list[Tensor] = []

        for obj in flat_inputs:
            if isinstance(obj, PointCloud3D):
                points.append(obj)
            elif isinstance(obj, BoundingBoxes3D):
                boxes.append(obj)
            elif isinstance(obj, CameraImages):
                images.append(obj)
            elif isinstance(obj, CameraExtrinsics):
                extrinsics.append(obj)
            elif isinstance(obj, CameraIntrinsics):
                intrinsics.append(obj)
            elif isinstance(obj, Tensor) and not isinstance(obj, TVTensor):
                labels.append(obj)

        n = len(boxes)
        has_points = len(points) > 0
        has_cameras = len(images) > 0
        has_labels = len(labels) > 0

        mismatched: list[str] = []
        if has_points and len(points) != n:
            mismatched.append(f"PointCloud3D ({len(points)})")
        if has_cameras and len(images) != n:
            mismatched.append(f"CameraImages ({len(images)})")
        if has_cameras and len(extrinsics) != n:
            mismatched.append(f"CameraExtrinsics ({len(extrinsics)})")
        if has_cameras and len(intrinsics) != n:
            mismatched.append(f"CameraIntrinsics ({len(intrinsics)})")
        if has_labels and len(labels) != n:
            mismatched.append(f"plain tensors ({len(labels)})")
        if mismatched:
            raise TypeError(
                f"{type(self).__name__}() requires equal sized lists of "
                f"inputs per sample. Got {n} BoundingBoxes3D but "
                f"{', '.join(mismatched)}."
            )

        batch_inputs: list[dict[str, Any]] = []
        batch_targets: list[dict[str, Any]] = []
        for i in range(n):
            inp: dict[str, Any] = {}
            if has_points:
                inp["points"] = points[i]
            if has_cameras:
                inp["images"] = images[i]
                inp["extrinsics"] = extrinsics[i]
                inp["intrinsics"] = intrinsics[i]

            tgt: dict[str, Any] = {"boxes": boxes[i]}
            if has_labels:
                tgt["labels"] = labels[i]

            batch_inputs.append(inp)
            batch_targets.append(tgt)

        return batch_inputs, batch_targets

    def _insert_outputs(
        self,
        flat_inputs: list[Any],
        output_inputs: list[dict[str, Any]],
        output_targets: list[dict[str, Any]],
    ) -> None:
        """Replace modified leaves in *flat_inputs* in-place.

        Uses per-type counters to walk through the flat list and replace
        each leaf with the corresponding value from the output dicts.
        """
        c_pts = 0
        c_img = 0
        c_box = 0
        c_lbl = 0

        for i, obj in enumerate(flat_inputs):
            if isinstance(obj, PointCloud3D):
                flat_inputs[i] = output_inputs[c_pts]["points"]
                c_pts += 1
            elif isinstance(obj, CameraImages):
                new_img = output_inputs[c_img].get("images")
                if new_img is not None:
                    flat_inputs[i] = new_img
                c_img += 1
            elif isinstance(obj, BoundingBoxes3D):
                flat_inputs[i] = output_targets[c_box]["boxes"]
                c_box += 1
            elif isinstance(obj, (CameraExtrinsics, CameraIntrinsics)):
                pass  # never modified by copy-paste
            elif isinstance(obj, Tensor) and not isinstance(obj, TVTensor):
                new_lbl = output_targets[c_lbl].get("labels")
                if new_lbl is not None:
                    flat_inputs[i] = new_lbl
                c_lbl += 1

    def _has_camera_data(self, inputs: dict[str, Any]) -> bool:
        return "images" in inputs and "extrinsics" in inputs and "intrinsics" in inputs

    def _extract_objects(self, inputs: dict[str, Any], targets: dict[str, Any]) -> None:
        """Extract per-object point clouds and store in database."""
        points = inputs.get("points")
        boxes = targets["boxes"]
        labels = targets.get("labels", torch.zeros(0, dtype=torch.long))

        if boxes.shape[0] == 0 or labels.shape[0] == 0:
            return

        fmt = boxes.format

        # Find valid objects: With point clouds this means meeting min_points,
        # for camera-only inputs all labeled boxes are valid.
        valid: list[tuple[int, Tensor | None]] = []
        if points is not None:
            indices = points_in_boxes_3d_indices(points, boxes, fmt)
            for j in range(boxes.shape[0]):
                if j >= labels.shape[0]:
                    break
                mask = indices == j
                obj_points = points[mask]
                if obj_points.shape[0] >= self.min_points:
                    valid.append((j, obj_points))
        else:
            for j in range(min(boxes.shape[0], labels.shape[0])):
                valid.append((j, None))

        # Batch camera crop extraction for all valid objects at once
        has_cameras = self._has_camera_data(inputs)
        camera_crops_map: dict[int, list[CameraCrop | None]] = {}
        if has_cameras and valid:
            camera_crops_map = self._extract_all_camera_crops(
                boxes, fmt, inputs, [j for j, _ in valid]
            )

        for j, obj_points in valid:
            label = int(labels[j].item())
            entry = ObjectEntry(
                points=obj_points.detach().cpu() if obj_points is not None else None,
                box=boxes[j].detach().cpu(),
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

    def _sample_axis_offsets(
        self,
        n: int,
        lo: float,
        hi: float,
        std: float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Draw ``n`` offsets for one axis from ``[lo, hi]``.

        Uniform when ``std`` is ``None``, otherwise a normal distribution
        centred on the midpoint and truncated to the interval, sampled exactly
        via the inverse CDF (no rejection loop).

        Args:
            n: Number of offsets to sample.
            lo: Interval lower bound.
            hi: Interval upper bound.
            std: Standard deviation, or ``None`` for uniform sampling.
            device: Device for the output tensor.
            dtype: Dtype for the output tensor.

        Returns:
            ``[n]`` tensor of offsets.
        """
        if lo == hi:
            return torch.full((n,), lo, device=device, dtype=dtype)
        if std is None:
            return torch.empty(n, device=device, dtype=dtype).uniform_(lo, hi)

        # Truncated normal via inverse-CDF: map uniform draws through the
        # standard-normal CDF restricted to [alpha, beta], then back.
        mean = (lo + hi) / 2.0
        alpha = torch.tensor((lo - mean) / std, device=device, dtype=dtype)
        beta = torch.tensor((hi - mean) / std, device=device, dtype=dtype)
        lo_cdf = torch.special.ndtr(alpha)
        hi_cdf = torch.special.ndtr(beta)
        u = torch.rand(n, device=device, dtype=dtype)
        p = lo_cdf + u * (hi_cdf - lo_cdf)
        # Clamp guards against tiny inverse-CDF rounding outside the interval.
        return (mean + std * torch.special.ndtri(p)).clamp(lo, hi)

    def _sample_offsets(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Draw ``n`` per-object ``(x, y, z)`` offsets.

        Each axis is sampled independently from its configured range and
        standard deviation.

        Args:
            n: Number of offsets to sample.
            device: Device for the output tensor.
            dtype: Dtype for the output tensor.

        Returns:
            ``[n, 3]`` tensor of offsets.
        """
        cols = [
            self._sample_axis_offsets(n, lo, hi, std, device, dtype)
            for (lo, hi), std in zip(self.offset_range, self.offset_std)
        ]
        return torch.stack(cols, dim=1)

    def _place_candidate(
        self,
        box: Tensor,
        fmt: BoundingBox3DFormat,
        occupied: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        """Find a collision-free pose for one candidate box.

        When jittering is enabled, up to ``max_jitter_attempts`` jittered
        offsets are drawn and the first that does not overlap *occupied* is
        used. The original (zero) offset is always appended as a final
        fallback, so an object whose jittered poses all collide is still
        placed at its source position when that position is itself free.

        Args:
            box: Candidate box ``[K]`` at its source position.
            fmt: Box format.
            occupied: Boxes ``[M, K]`` the candidate must not overlap (scene
                boxes plus objects already pasted this sample).

        Returns:
            ``(placed_box, offset)`` for the chosen pose, or ``None`` if every
            attempt (including the original) collides.
        """
        device = box.device
        dtype = box.dtype
        # Jittered attempts first, the original (zero) offset last as fallback.
        if self._jitter:
            jittered = self._sample_offsets(self.max_jitter_attempts, device, dtype)
            offsets = torch.cat(
                [jittered, torch.zeros(1, 3, device=device, dtype=dtype)]
            )
        else:
            offsets = torch.zeros(1, 3, device=device, dtype=dtype)

        attempts = _apply_offset(
            box.unsqueeze(0).expand(offsets.shape[0], -1), fmt, offsets
        )

        if occupied.shape[0] > 0:
            collide = box3d_overlap(attempts, occupied, fmt).any(dim=1)
        else:
            collide = torch.zeros(offsets.shape[0], dtype=torch.bool, device=device)

        free = (~collide).nonzero(as_tuple=False)
        if free.numel() == 0:
            return None
        k = int(free[0].item())
        return attempts[k], offsets[k]

    def _paste_objects(
        self,
        inputs: dict[str, Any],
        targets: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Paste objects from database into a single sample.

        Returns:
            Modified ``(inputs, targets)`` dicts.
        """
        points = inputs.get("points")
        boxes = targets["boxes"]
        labels = targets.get("labels", torch.zeros(0, dtype=torch.long))
        fmt = boxes.format

        # Count existing objects per label
        existing_counts: dict[int, int] = {}
        for lbl in labels.tolist():
            existing_counts[lbl] = existing_counts.get(lbl, 0) + 1

        pasted_entries: list[ObjectEntry] = []
        pasted_boxes: list[Tensor] = []
        pasted_points: list[Tensor] = []
        pasted_labels: list[int] = []

        all_boxes = boxes

        device = boxes.device

        for label_id, target_count in self.target_counts.items():
            n_paste = max(0, target_count - existing_counts.get(label_id, 0))
            db = self._database.get(label_id)
            if not db or n_paste == 0:
                continue

            perm = torch.randperm(len(db)).tolist()
            candidates = [db[i] for i in perm[:n_paste]]

            if self._jitter:
                # Jitter path: place one at a time so each candidate's jittered
                # pose is checked against both the scene boxes and the objects
                # already pasted this sample. This costs one overlap kernel per
                # candidate, which is why it is gated behind ``_jitter``.
                for entry in candidates:
                    placed = self._place_candidate(entry.box.to(device), fmt, all_boxes)
                    if placed is None:
                        continue
                    box_k, offset_k = placed
                    # Re-bind the entry to the placed box so camera projection
                    # and points stay consistent with the placed 3-D box.
                    entry = replace(entry, box=box_k.detach().cpu())
                    pasted_entries.append(entry)
                    pasted_boxes.append(box_k)
                    if entry.points is not None and points is not None:
                        obj_points = entry.points.to(points.device).clone()
                        obj_points[:, :3] += offset_k.to(obj_points.device)
                        pasted_points.append(obj_points)
                    pasted_labels.append(entry.label)
                    all_boxes = torch.cat([all_boxes, box_k.unsqueeze(0)])
            else:
                # Default fast path: two batched overlap kernels per class
                # (candidates vs scene, candidates vs each other) followed by
                # a cheap greedy accept on CPU. No per-candidate GPU sync.
                cand_boxes = torch.stack([c.box for c in candidates]).to(device)

                # Candidates vs existing scene boxes.
                if all_boxes.shape[0] > 0:
                    safe = ~box3d_overlap(cand_boxes, all_boxes, fmt).any(dim=1)
                else:
                    safe = torch.ones(
                        cand_boxes.shape[0], dtype=torch.bool, device=device
                    )

                # Candidates vs each other.
                cc = box3d_overlap(cand_boxes, cand_boxes, fmt)
                cc.fill_diagonal_(False)

                safe_cpu = safe.cpu()
                cc_cpu = cc.cpu()
                accepted_k: list[int] = []
                for k in range(len(candidates)):
                    if not safe_cpu[k].item():
                        continue
                    if cc_cpu[k, accepted_k].any().item():
                        continue
                    accepted_k.append(k)

                if not accepted_k:
                    continue
                for k in accepted_k:
                    entry = candidates[k]
                    pasted_entries.append(entry)
                    pasted_boxes.append(cand_boxes[k])
                    if entry.points is not None and points is not None:
                        pasted_points.append(entry.points.to(points.device))
                    pasted_labels.append(entry.label)
                all_boxes = torch.cat([all_boxes, cand_boxes[accepted_k]])

        if not pasted_boxes:
            return inputs, targets

        pasted_boxes_tensor = torch.stack(pasted_boxes)

        new_inputs: dict[str, Any] = {**inputs}

        # Point cloud update: remove scene points in pasted regions, add pasted points
        if points is not None:
            remove_mask = points_in_boxes_3d(points, pasted_boxes_tensor, fmt).any(
                dim=1
            )
            kept_points = points[~remove_mask]
            if pasted_points:
                new_points = torch.cat([kept_points, torch.cat(pasted_points)])
            else:
                new_points = kept_points
            new_inputs["points"] = PointCloud3D(new_points)

        # Box and label update
        new_boxes = torch.cat([boxes, pasted_boxes_tensor])
        new_labels = torch.cat(
            [
                labels,
                torch.tensor(pasted_labels, dtype=labels.dtype, device=labels.device),
            ]
        )

        new_targets: dict[str, Any] = {
            **targets,
            "boxes": BoundingBoxes3D(new_boxes, format=fmt),
            "labels": new_labels,
        }

        # Camera image paste
        if self._has_camera_data(inputs):
            new_images = self._paste_camera_images(inputs, boxes, fmt, pasted_entries)
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

        # Pre-compute centers (format-aware) once outside the camera loop.
        # Database entries live on CPU so we must bring them onto the working device.
        pasted_box_stack = torch.stack([e.box for e in pasted_entries]).to(
            images.device
        )
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

                # Resize stored crop and mask to target size.
                # Stored on CPU so we must bring onto the working device before resize.
                src_crop = (
                    cam_crop.crop.unsqueeze(0).float().to(images.device)
                )  # [1, C, sh, sw]
                resized_crop = F.interpolate(
                    src_crop,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # [C, th, tw]

                src_mask = (
                    cam_crop.mask.unsqueeze(0).unsqueeze(0).float().to(images.device)
                )
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
