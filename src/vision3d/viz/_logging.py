"""Log vision3d data to a Rerun viewer."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from vision3d.datasets import SampleInputs, SampleTargets
from vision3d.metrics import Prediction3D
from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    GridMask3D,
    PointCloud3D,
)

try:
    import rerun as rr
except ImportError as e:
    msg = "rerun-sdk is required for visualization. Install with: pip install vision3d[viz]"
    raise ImportError(msg) from e

# Thickness of planar grids relative to their smallest cell edge. Rerun
# rejects a voxel size of zero.
_PLANAR_THICKNESS = 1e-3

# Multipliers of ``torchvision.utils._generate_color_palette``, which maps an
# index ``i`` to the color ``(i * _PALETTE_FACTORS) % 255``.
_PALETTE_FACTORS = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], device="cpu")


def log_point_cloud(
    entity: str,
    points: PointCloud3D | Tensor,
    *,
    color_by_distance: bool = True,
) -> None:
    """Log a point cloud to Rerun.

    Args:
        entity: Rerun entity path (e.g. ``"world/lidar"``).
        points: Point cloud ``[N, 3+C]``. First 3 columns are xyz.
        color_by_distance: Color points by distance from origin.
    """
    xyz = points[:, :3].detach().cpu()

    colors = None
    if color_by_distance:
        distances = torch.linalg.norm(xyz, dim=1)
        max_dist = max(float(torch.quantile(distances, 0.98)), 1e-6)
        normalized = (distances / max_dist).clamp(0, 1)
        colors = torch.zeros(len(xyz), 4, dtype=torch.uint8)
        colors[:, 0] = (normalized * 255).to(torch.uint8)
        colors[:, 2] = ((1 - normalized) * 255).to(torch.uint8)
        colors[:, 3] = 255

    rr.log(entity, rr.Points3D(xyz, colors=colors))


def log_boxes_3d(
    entity: str,
    boxes: BoundingBoxes3D,
    *,
    labels: list[str] | None = None,
    class_ids: list[int] | None = None,
    label_to_id: dict[str, int] | None = None,
    scores: list[float] | Tensor | None = None,
    score_threshold: float | None = None,
    fill_mode: rr.components.FillModeLike | None = None,
    show_labels: bool | None = None,
    log_heading: bool = True,
) -> None:
    """Log 3D bounding boxes to Rerun.

    Logs boxes as ``rr.Boxes3D`` and optionally heading arrows as
    ``rr.Arrows3D`` on a ``/heading`` sub-entity. Designed to serve both
    ground-truth and prediction boxes: route each to its own entity (e.g.
    ``"world/gt/boxes"`` vs ``"world/pred/boxes"``) and distinguish them
    visually with ``fill_mode`` while keeping per-class colors.

    Args:
        entity: Rerun entity path (e.g. ``"world/gt/boxes"``).
        boxes: Bounding boxes in any supported format.
        labels: Per-box label strings for display. When ``scores`` is
            given, the score is appended to each label.
        class_ids: Per-box class IDs for coloring via AnnotationContext.
        label_to_id: Mapping from class name to class ID. When provided,
            an ``rr.AnnotationContext`` is logged statically on the
            entity so ``class_ids`` resolve to consistent colors and
            display names across frames.
        scores: Per-box confidence scores. When given, each box label
            shows its score (e.g. ``"car 0.87"``).
        score_threshold: If set, boxes with ``scores`` below this value are
            dropped before logging. Requires ``scores``.
        fill_mode: Box fill mode (e.g. ``"majorwireframe"``,
            ``"densewireframe"``, ``"solid"``).
        show_labels: Force per-box labels on (``True``) or off (``False``)
            in the viewer. ``None`` leaves Rerun's default heuristic, which
            hides labels when there are many boxes.
        log_heading: If True and boxes have rotation, log heading arrows.

    Raises:
        ValueError: If ``score_threshold`` is set without ``scores``, or if
            ``scores``, ``labels``, or ``class_ids`` length does not match
            the number of boxes.
    """
    if label_to_id is not None:
        rr.log(
            entity,
            rr.AnnotationContext([(i, name) for name, i in label_to_id.items()]),
            static=True,
        )

    raw = boxes.as_subclass(Tensor).detach().cpu()
    fmt = boxes.format

    score_list = (
        scores.detach().cpu().tolist() if isinstance(scores, Tensor) else scores
    )
    if score_list is not None and len(score_list) != raw.shape[0]:
        msg = f"scores has length {len(score_list)} but there are {raw.shape[0]} boxes"
        raise ValueError(msg)
    if labels is not None and len(labels) != raw.shape[0]:
        msg = f"labels has length {len(labels)} but there are {raw.shape[0]} boxes"
        raise ValueError(msg)
    if class_ids is not None and len(class_ids) != raw.shape[0]:
        msg = (
            f"class_ids has length {len(class_ids)} but there are {raw.shape[0]} boxes"
        )
        raise ValueError(msg)
    if score_threshold is not None:
        if score_list is None:
            msg = "score_threshold requires scores"
            raise ValueError(msg)
        keep = [i for i, s in enumerate(score_list) if s >= score_threshold]
        raw = raw[keep]
        score_list = [score_list[i] for i in keep]
        if class_ids is not None:
            class_ids = [class_ids[i] for i in keep]
        if labels is not None:
            labels = [labels[i] for i in keep]

    n = raw.shape[0]

    if n == 0:
        rr.log(entity, rr.Clear(recursive=True))
        return

    display_labels = _build_labels(labels, class_ids, label_to_id, score_list)

    centers, sizes, yaws = _extract_centers_sizes_yaws(raw, fmt)

    quaternions = [
        rr.Quaternion(xyzw=[0.0, 0.0, math.sin(y / 2), math.cos(y / 2)]) for y in yaws
    ]

    rr.log(
        entity,
        rr.Boxes3D(
            centers=centers,
            sizes=sizes,
            quaternions=quaternions,
            class_ids=class_ids,
            labels=display_labels,
            fill_mode=fill_mode,
            show_labels=show_labels,
        ),
    )

    if log_heading and BoundingBox3DFormat.is_rotated(fmt):
        half_len = sizes[:, 0] / 2
        face_area = sizes[:, 1] * sizes[:, 2]
        face_scale = torch.sqrt(face_area)
        arrow_len = face_scale * 0.6
        yaws_t = torch.tensor(yaws)
        cos_y = torch.cos(yaws_t)
        sin_y = torch.sin(yaws_t)

        origins = centers.clone()
        origins[:, 0] += half_len * cos_y
        origins[:, 1] += half_len * sin_y

        vectors = torch.zeros(n, 3)
        vectors[:, 0] = arrow_len * cos_y
        vectors[:, 1] = arrow_len * sin_y

        radii = face_scale * 0.06

        rr.log(
            f"{entity}/heading",
            rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                radii=radii,
                colors=[(255, 255, 255)] * n,
            ),
        )


def log_cameras(
    entity_prefix: str,
    images: CameraImages | Tensor,
    intrinsics: CameraIntrinsics | Tensor | None = None,
    extrinsics: CameraExtrinsics | Tensor | None = None,
    *,
    jpeg_quality: int | None = None,
) -> None:
    """Log all camera images with optional pinhole projection to Rerun.

    Each camera is logged to ``{entity_prefix}_{i}``.

    Args:
        entity_prefix: Rerun entity path prefix (e.g. ``"world/cam"``).
        images: Camera images ``[N_cams, C, H, W]``.
        intrinsics: Intrinsic matrices ``[N_cams, 3, 3]``.
        extrinsics: Extrinsic matrices ``[N_cams, 4, 4]`` (lidar-to-camera).
        jpeg_quality: If set, JPEG-encode each image at this quality (0-100)
            before logging. ``None`` (default) logs uncompressed.
    """
    for i in range(images.shape[0]):
        _log_single_camera(
            f"{entity_prefix}_{i}",
            images,
            intrinsics,
            extrinsics,
            camera_index=i,
            jpeg_quality=jpeg_quality,
        )


def _log_single_camera(
    entity: str,
    images: CameraImages | Tensor,
    intrinsics: CameraIntrinsics | Tensor | None,
    extrinsics: CameraExtrinsics | Tensor | None,
    *,
    camera_index: int,
    jpeg_quality: int | None = None,
) -> None:
    img = images[camera_index].detach().cpu()
    # [C, H, W] -> [H, W, C], uint8
    if img.is_floating_point() and img.max() <= 1.0:
        img = (img * 255).to(torch.uint8)
    elif img.is_floating_point():
        img = img.to(torch.uint8)
    img = img.permute(1, 2, 0)

    if extrinsics is not None:
        ext = extrinsics[camera_index].detach().cpu()
        rr.log(
            entity,
            rr.Transform3D(
                translation=ext[:3, 3],
                mat3x3=ext[:3, :3],
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )

    if intrinsics is not None:
        K = intrinsics[camera_index].detach().cpu()
        h, w = img.shape[:2]
        rr.log(
            entity,
            rr.Pinhole(
                image_from_camera=K,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )

    archetype = rr.Image(img)
    if jpeg_quality is not None:
        archetype = archetype.compress(jpeg_quality=jpeg_quality)
    rr.log(entity, archetype)


def log_grid_mask(
    entity: str,
    grid: GridMask3D,
    *,
    channel_names: Sequence[str] | None = None,
    colors: Sequence[tuple[int, int, int]] | None = None,
    background: float | None = 0,
    colormap: rr.components.ColormapLike | None = None,
    value_range: tuple[float, float] | None = None,
    opacity: float | None = None,
) -> None:
    """Log a label grid to Rerun as sparse voxels.

    Each channel is logged as an ``rr.VoxelGridMap`` placed by the grid's
    ``affine``, so planar grids ``[C, H, W]`` and volumes ``[C, D, H, W]``
    line up with the rest of the scene in the same frame. A planar grid is
    drawn flat in the plane of its cells, ignoring the thickness of its ``d``
    axis. Rerun needs a positive voxel size, so the layer is a thousandth of
    a cell thick. A grid with one channel is logged at ``entity``, otherwise
    each channel is logged at ``{entity}/{name}``.

    How cells are colored depends on the dtype:

    - ``bool``: Multi-label masks. Set cells take one color per channel.
    - Integer: Class indices. Each cell takes the color of its value.
    - Floating point: Soft labels or regression targets, colored through
      ``colormap`` over ``value_range``.

    Args:
        entity: Rerun entity path (e.g. ``"world/gt/map"``).
        grid: Label grid to log.
        channel_names: One name per channel for the entity paths. Defaults to
            the channel indices.
        colors: RGB colors, indexed by channel for ``bool`` grids and by value
            for integer grids. Defaults to the palette torchvision generates
            for segmentation masks. Ignored for floating point grids.
        background: Cells equal to this value are not logged, e.g. ``False``
            for masks, a free-space class, or zero soft labels. ``None`` logs
            every cell.
        colormap: Colormap for floating point grids. Defaults to Rerun's
            default colormap.
        value_range: ``(min, max)`` mapped onto ``colormap``. Defaults to the
            range of the logged values.
        opacity: Voxel opacity in ``[0, 1]``.

    Raises:
        ValueError: If ``channel_names`` does not have one name per channel,
            or if the affine shears the grid, which voxels cannot represent.
    """
    data = grid.as_subclass(Tensor).detach().cpu()
    planar = data.ndim == 3
    if planar:
        data = data.unsqueeze(1)
    num_channels = data.shape[0]
    if channel_names is None:
        channel_names = [str(c) for c in range(num_channels)]
    if len(channel_names) != num_channels:
        msg = (
            f"channel_names has length {len(channel_names)} but the grid has "
            f"{num_channels} channels"
        )
        raise ValueError(msg)
    palette = None
    if colors is not None:
        palette = torch.tensor(colors, dtype=torch.uint8, device="cpu").reshape(-1, 3)

    voxel_size, translation, quaternion, flip_d = _grid_pose(grid.affine, planar=planar)
    for c, name in enumerate(channel_names):
        path = entity if num_channels == 1 else f"{entity}/{name}"
        channel = data[c]
        keep = (
            channel != background
            if background is not None
            else torch.ones_like(channel, dtype=torch.bool)
        )
        indices = keep.nonzero()
        if len(indices) == 0:
            rr.log(path, rr.Clear(recursive=False))
            continue
        values = channel[keep]
        d, h, w = indices.unbind(1)
        if flip_d:
            d = -d - 1
        # Local voxel axes are (x, y, z) = (h, w, d). See ``_grid_pose``.
        voxel_indices = torch.stack([h, w, d], dim=1).to(torch.int32)

        voxel_values = None
        voxel_colors = None
        channel_range = None
        if channel.dtype.is_floating_point:
            voxel_values = values.float()
            channel_range = value_range or (float(values.min()), float(values.max()))
        else:
            # A mask takes the color of its channel, a class grid that of each value.
            ids = (
                values.long()
                if channel.dtype != torch.bool
                else torch.tensor([c], device="cpu")
            )
            voxel_colors = _palette_colors(ids, palette)

        rr.log(
            path,
            rr.VoxelGridMap(
                voxel_indices,
                voxel_size=voxel_size,
                values=voxel_values,
                colors=voxel_colors,
                translation=translation,
                quaternion=rr.Quaternion(xyzw=quaternion),
                opacity=opacity,
                value_range=channel_range,
                colormap=colormap,
            ),
        )


def log_sample(
    inputs: SampleInputs,
    targets: SampleTargets | None = None,
    *,
    predictions: Prediction3D | None = None,
    entity_prefix: str = "world",
    label_to_id: dict[str, int] | None = None,
    score_threshold: float | None = None,
    jpeg_quality: int | None = None,
    map_channel_names: Sequence[str] | None = None,
    map_colors: Sequence[tuple[int, int, int]] | None = None,
) -> None:
    """Log a full sample dict to Rerun.

    Convenience function that dispatches to type-specific loggers. Ground
    truth and predictions are logged to separate entities
    (``{entity_prefix}/gt/boxes`` and ``{entity_prefix}/pred/boxes``) so they
    can be toggled independently; both keep per-class colors and are
    distinguished by fill style (ground truth as translucent colored
    faces, predictions as a wireframe).

    Args:
        inputs: :class:`~vision3d.datasets.SampleInputs` with ``"points"``,
            ``"images"``, ``"extrinsics"``, ``"intrinsics"`` keys.
        targets: Optional :class:`~vision3d.datasets.SampleTargets` with
            ``"boxes"``, ``"labels"`` keys (ground truth). A ``"map"`` grid
            is logged to ``{entity_prefix}/gt/map`` with
            :func:`log_grid_mask` at half opacity.
        predictions: Optional :class:`~vision3d.metrics.Prediction3D` with
            ``"boxes"``, ``"scores"``, ``"labels"`` keys. Prediction labels
            show their score.
        entity_prefix: Rerun entity path prefix.
        label_to_id: Mapping from class name to class ID for consistent
            coloring. Build this across all frames before logging.
        score_threshold: If set, predictions below this score are dropped.
        jpeg_quality: If set, JPEG-encode camera images at this quality
            (0-100) before logging. See :func:`log_cameras`.
        map_channel_names: Channel names for ``targets["map"]``, e.g.
            ``NuScenes3D.map_layers``. See :func:`log_grid_mask`.
        map_colors: One RGB color per channel of ``targets["map"]``, e.g.
            ``NuScenes3D.map_colors``. See :func:`log_grid_mask`.
    """
    if "points" in inputs:
        log_point_cloud(f"{entity_prefix}/lidar", inputs["points"])

    if "images" in inputs:
        log_cameras(
            f"{entity_prefix}/cam",
            inputs["images"],
            inputs.get("intrinsics"),
            inputs.get("extrinsics"),
            jpeg_quality=jpeg_quality,
        )

    if targets and "boxes" in targets:
        class_ids = targets["labels"].tolist() if "labels" in targets else None
        log_boxes_3d(
            f"{entity_prefix}/gt/boxes",
            targets["boxes"],
            class_ids=class_ids,
            label_to_id=label_to_id,
            fill_mode="transparentfillmajorwireframe",
        )

    if targets and "map" in targets:
        log_grid_mask(
            f"{entity_prefix}/gt/map",
            targets["map"],
            channel_names=map_channel_names,
            colors=map_colors,
            opacity=0.5,
        )

    if predictions and "boxes" in predictions:
        class_ids = predictions["labels"].tolist() if "labels" in predictions else None
        log_boxes_3d(
            f"{entity_prefix}/pred/boxes",
            predictions["boxes"],
            class_ids=class_ids,
            label_to_id=label_to_id,
            scores=predictions["scores"],
            score_threshold=score_threshold,
            fill_mode="majorwireframe",
            show_labels=True,
        )


def _build_labels(
    labels: list[str] | None,
    class_ids: list[int] | None,
    label_to_id: dict[str, int] | None,
    scores: list[float] | None,
) -> list[str] | None:
    """Build per-box display labels, appending scores when available.

    Returns ``None`` when there is nothing to display (no explicit labels,
    no resolvable class names, and no scores), letting Rerun fall back to
    its ``AnnotationContext`` names.

    Returns:
        Per-box label strings, or ``None``.
    """
    if scores is None:
        return labels

    base = labels
    if base is None and class_ids is not None and label_to_id is not None:
        id_to_label = {i: name for name, i in label_to_id.items()}
        base = [id_to_label.get(c, str(c)) for c in class_ids]

    if base is None:
        return [f"{s:.2f}" for s in scores]
    return [f"{name} {s:.2f}" for name, s in zip(base, scores)]


def _extract_centers_sizes_yaws(
    raw: Tensor, fmt: BoundingBox3DFormat
) -> tuple[Tensor, Tensor, list[float]]:
    """Extract centers, sizes (l, w, h), and yaw angles from raw box tensor.

    Returns:
        Tuple of (centers, sizes, yaws). Centers and sizes are tensors,
        yaws is a list of floats.

    Raises:
        ValueError: If ``fmt`` is not a supported format.
    """
    if fmt is BoundingBox3DFormat.XYZXYZ:
        mins = raw[:, :3]
        maxs = raw[:, 3:]
        centers = (mins + maxs) / 2
        sizes = maxs - mins
        yaws = [0.0] * raw.shape[0]
    elif fmt is BoundingBox3DFormat.XYZLWH:
        centers = raw[:, :3]
        sizes = raw[:, 3:6]
        yaws = [0.0] * raw.shape[0]
    elif fmt is BoundingBox3DFormat.XYZLWHY or fmt is BoundingBox3DFormat.XYZLWHYPR:
        centers = raw[:, :3]
        sizes = raw[:, 3:6]
        yaws = raw[:, 6].tolist()
    else:
        msg = f"Unsupported format: {fmt}"
        raise ValueError(msg)

    return centers, sizes, yaws


def _grid_pose(
    affine: Tensor, *, planar: bool = False
) -> tuple[list[float], list[float], list[float], bool]:
    """Split a grid affine into a Rerun voxel grid pose.

    The local voxel axes ``(x, y, z)`` are the grid's ``(h, w, d)`` axes, so a
    planar grid lies in the local ``xy``-plane. A reflection cannot be a
    rotation, so it is undone by reversing the ``d`` axis, and the caller
    maps each ``d`` index to ``-d - 1``. A planar grid is flattened to a
    thin layer centered on its cells instead, which a reversed ``d`` axis
    leaves in place, so its indices need no remapping.

    Returns:
        ``(voxel_size, translation, quaternion_xyzw, flip_d)``, where the
        translation places the minimum corner of local voxel ``(0, 0, 0)``.

    Raises:
        ValueError: If the grid axes are not orthogonal.
    """
    affine = affine.double()
    columns = affine[:3, :3]
    axes = torch.stack([columns[:, 1], columns[:, 2], columns[:, 0]], dim=1)
    size = torch.linalg.norm(axes, dim=0)
    rotation = axes / size
    identity = torch.eye(3, dtype=torch.float64, device=rotation.device)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1e-6):
        msg = (
            "Rerun voxel grids need orthogonal grid axes, but the affine "
            f"shears the grid: {affine.tolist()}"
        )
        raise ValueError(msg)
    reflected = bool(torch.linalg.det(rotation) < 0)
    if reflected:
        rotation[:, 2] = -rotation[:, 2]
    first_center = affine[:3, 3]
    if planar:
        size[2] = _PLANAR_THICKNESS * min(float(size[0]), float(size[1]))
    elif reflected:
        # Local voxel 0 along z is the cell at d = -1.
        first_center = first_center - columns[:, 0]
    flip_d = reflected and not planar
    translation = first_center - rotation @ (size / 2)
    return (
        size.tolist(),
        translation.tolist(),
        _rotation_to_quaternion(rotation),
        flip_d,
    )


def _rotation_to_quaternion(rotation: Tensor) -> list[float]:
    """Convert a rotation matrix to a unit quaternion.

    Uses Shepperd's method, which branches on the largest quaternion
    component to stay numerically stable.

    Returns:
        Quaternion ``(x, y, z, w)``.
    """
    m = rotation.tolist()
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        q = (
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            s / 4,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2 * math.sqrt(1 + m[0][0] - m[1][1] - m[2][2])
        q = (
            s / 4,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        )
    elif m[1][1] > m[2][2]:
        s = 2 * math.sqrt(1 + m[1][1] - m[0][0] - m[2][2])
        q = (
            (m[0][1] + m[1][0]) / s,
            s / 4,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        )
    else:
        s = 2 * math.sqrt(1 + m[2][2] - m[0][0] - m[1][1])
        q = (
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            s / 4,
            (m[1][0] - m[0][1]) / s,
        )
    return list(q)


def _palette_colors(ids: Tensor, palette: Tensor | None) -> Tensor:
    """Look up RGB colors for channel or class indices.

    Returns:
        ``[N, 3]`` uint8 colors, from ``palette`` when given and otherwise from
        the formula torchvision uses for segmentation masks.
    """
    if palette is not None:
        return palette[ids % len(palette)]
    return (ids[:, None] * _PALETTE_FACTORS % 255).to(torch.uint8)
