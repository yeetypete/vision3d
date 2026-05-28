"""Log vision3d data to a Rerun viewer."""

import math

import torch
from torch import Tensor

from vision3d.datasets import SampleInputs, SampleTargets
from vision3d.ops._project import project_to_image
from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    Cylinder3DFormat,
    Cylinders3D,
    PointCloud3D,
)

try:
    import rerun as rr
except ImportError as e:
    msg = "rerun-sdk is required for visualization. Install with: pip install vision3d[viz]"
    raise ImportError(msg) from e


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
    log_heading: bool = True,
) -> None:
    """Log 3D bounding boxes to Rerun.

    Logs boxes as ``rr.Boxes3D`` and optionally heading arrows as
    ``rr.Arrows3D`` on a ``/heading`` sub-entity.

    Args:
        entity: Rerun entity path (e.g. ``"world/boxes"``).
        boxes: Bounding boxes in any supported format.
        labels: Per-box label strings for display.
        class_ids: Per-box class IDs for coloring via AnnotationContext.
        label_to_id: Mapping from class name to class ID. When provided,
            an ``rr.AnnotationContext`` is logged statically on the
            entity so ``class_ids`` resolve to consistent colors and
            display names across frames.
        log_heading: If True and boxes have rotation, log heading arrows.
    """
    if label_to_id is not None:
        rr.log(
            entity,
            rr.AnnotationContext([(i, name) for name, i in label_to_id.items()]),
            static=True,
        )

    raw = boxes.as_subclass(Tensor).detach().cpu()
    fmt = boxes.format
    n = raw.shape[0]

    if n == 0:
        rr.log(entity, rr.Clear(recursive=True))
        return

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
            labels=labels,
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


def log_cylinders_3d(
    entity: str,
    cylinders: Cylinders3D,
    *,
    labels: list[str] | None = None,
    class_ids: list[int] | None = None,
    label_to_id: dict[str, int] | None = None,
) -> None:
    """Log upright 3D cylinders to Rerun.

    Logs cylinders as ``rr.Cylinders3D`` aligned with the Z axis.

    Args:
        entity: Rerun entity path (e.g. ``"world/cylinders"``).
        cylinders: Cylinders in any supported format.
        labels: Per-cylinder label strings for display.
        class_ids: Per-cylinder class IDs for coloring via AnnotationContext.
        label_to_id: Mapping from class name to class ID. When provided,
            an ``rr.AnnotationContext`` is logged statically on the
            entity so ``class_ids`` resolve to consistent colors and
            display names across frames.
    """
    if label_to_id is not None:
        rr.log(
            entity,
            rr.AnnotationContext([(i, name) for name, i in label_to_id.items()]),
            static=True,
        )

    raw = cylinders.as_subclass(Tensor).detach().cpu()
    fmt = cylinders.format
    n = raw.shape[0]

    if n == 0:
        rr.log(entity, rr.Clear(recursive=True))
        return

    centers, radii, heights = _extract_centers_radii_heights(raw, fmt)

    rr.log(
        entity,
        rr.Cylinders3D(
            centers=centers,
            radii=radii,
            lengths=heights,
            class_ids=class_ids,
            labels=labels,
        ),
    )


def log_cylinders_on_cameras(
    entity_prefix: str,
    cylinders: Cylinders3D,
    extrinsics: CameraExtrinsics | Tensor,
    intrinsics: CameraIntrinsics | Tensor,
    *,
    class_ids: list[int] | None = None,
    label_to_id: dict[str, int] | None = None,
    num_circle_segments: int = 24,
    num_verticals: int = 6,
    line_radius: float = -0.5,
) -> None:
    """Project upright 3D cylinders into each camera and log 2D wireframes.

    Rerun's viewer reprojects ``rr.Boxes3D`` into 2D pinhole views but not
    ``rr.Cylinders3D``, so cylinders are projected here explicitly and logged
    as :class:`~rerun.LineStrips2D` (two circles plus vertical edges) under
    each camera entity. Cylinders whose center falls behind or far outside a
    camera's frustum are skipped for that camera.

    Args:
        entity_prefix: Camera entity prefix (e.g. ``"world/cam"``); strips are
            logged to ``{entity_prefix}_{i}/cylinders`` to match
            :func:`log_cameras`.
        cylinders: Cylinders in the lidar frame.
        extrinsics: Lidar-to-camera transforms ``[N_cams, 4, 4]``.
        intrinsics: Per-camera intrinsic matrices ``[N_cams, 3, 3]``.
        class_ids: Per-cylinder class IDs for coloring via AnnotationContext.
        label_to_id: Mapping from class name to class ID. When provided, an
            ``rr.AnnotationContext`` is logged statically on each camera's
            cylinder entity so ``class_ids`` resolve to consistent colors.
        num_circle_segments: Number of segments approximating each circle.
        num_verticals: Number of vertical edges connecting the two circles.
        line_radius: Wireframe line radius. Negative values are interpreted in
            UI points (constant on-screen thickness); positive values are in
            image pixels.
    """
    raw = cylinders.as_subclass(Tensor).detach().cpu()
    fmt = cylinders.format
    ext = extrinsics.as_subclass(Tensor).detach().cpu()
    intr = intrinsics.as_subclass(Tensor).detach().cpu()
    n_cams = ext.shape[0]
    n = raw.shape[0]

    centers, radii, heights = (
        _extract_centers_radii_heights(raw, fmt)
        if n > 0
        else (raw[:, :3], raw[:, 0], raw[:, 0])
    )

    # Circle angles (closed loop) and the subset used for vertical edges.
    ring = torch.linspace(0, 2 * math.pi, num_circle_segments + 1)
    ring_cos, ring_sin = torch.cos(ring), torch.sin(ring)
    vert = torch.linspace(0, 2 * math.pi, num_verticals + 1)[:-1]
    vert_cos, vert_sin = torch.cos(vert), torch.sin(vert)

    for cam in range(n_cams):
        entity = f"{entity_prefix}_{cam}/cylinders"

        if label_to_id is not None:
            rr.log(
                entity,
                rr.AnnotationContext([(i, name) for name, i in label_to_id.items()]),
                static=True,
            )

        E, K = ext[cam], intr[cam]
        w, h_img = float(2 * K[0, 2]), float(2 * K[1, 2])
        strips: list[Tensor] = []
        strip_class_ids: list[int] = []

        for j in range(n):
            cx, cy, cz = centers[j].tolist()
            r, h = float(radii[j]), float(heights[j])
            top_z, bot_z = cz + h / 2, cz - h / 2

            # Skip cylinders whose center is behind or far outside this camera.
            center_uv, _ = project_to_image(torch.tensor([[cx, cy, cz]]), E, K)
            if torch.isnan(center_uv).any():
                continue
            u, v = center_uv[0].tolist()
            margin = 0.5 * max(w, h_img)
            if not (-margin <= u <= w + margin and -margin <= v <= h_img + margin):
                continue

            ring_x = cx + r * ring_cos
            ring_y = cy + r * ring_sin
            polylines = [
                torch.stack([ring_x, ring_y, torch.full_like(ring_cos, top_z)], dim=1),
                torch.stack([ring_x, ring_y, torch.full_like(ring_cos, bot_z)], dim=1),
            ]
            for k in range(num_verticals):
                px = cx + r * float(vert_cos[k])
                py = cy + r * float(vert_sin[k])
                polylines.append(torch.tensor([[px, py, top_z], [px, py, bot_z]]))

            for poly in polylines:
                uv, _ = project_to_image(poly, E, K)
                if torch.isnan(uv).any():
                    continue
                strips.append(uv)
                strip_class_ids.append(class_ids[j] if class_ids is not None else 0)

        if strips:
            rr.log(
                entity,
                rr.LineStrips2D(
                    strips,
                    radii=line_radius,
                    class_ids=strip_class_ids if class_ids is not None else None,
                ),
            )
        else:
            rr.log(entity, rr.Clear(recursive=True))


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


def log_sample(
    inputs: SampleInputs,
    targets: SampleTargets | None = None,
    *,
    entity_prefix: str = "world",
    label_to_id: dict[str, int] | None = None,
    jpeg_quality: int | None = None,
) -> None:
    """Log a full sample dict to Rerun.

    Convenience function that dispatches to type-specific loggers.

    Args:
        inputs: Dict with ``"points"``, ``"images"``, ``"extrinsics"``,
            ``"intrinsics"`` keys.
        targets: Optional dict with ``"boxes"``, ``"labels"`` keys.
        entity_prefix: Rerun entity path prefix.
        label_to_id: Mapping from class name to class ID for consistent
            coloring. Build this across all frames before logging.
        jpeg_quality: If set, JPEG-encode camera images at this quality
            (0-100) before logging. See :func:`log_cameras`.
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
            f"{entity_prefix}/boxes",
            targets["boxes"],
            class_ids=class_ids,
            label_to_id=label_to_id,
        )

    if targets and "cylinders" in targets:
        class_ids = targets["labels"].tolist() if "labels" in targets else None
        log_cylinders_3d(
            f"{entity_prefix}/cylinders",
            targets["cylinders"],
            class_ids=class_ids,
            label_to_id=label_to_id,
        )
        # Cylinders don't reproject into 2D camera views, so project them
        # explicitly onto each camera as 2D wireframes.
        if "extrinsics" in inputs and "intrinsics" in inputs:
            log_cylinders_on_cameras(
                f"{entity_prefix}/cam",
                targets["cylinders"],
                inputs["extrinsics"],
                inputs["intrinsics"],
                class_ids=class_ids,
                label_to_id=label_to_id,
            )


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


def _extract_centers_radii_heights(
    raw: Tensor, fmt: Cylinder3DFormat
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract centers, radii, and heights from a raw cylinder tensor.

    Returns:
        Tuple of (centers ``[N, 3]``, radii ``[N]``, heights ``[N]``).

    Raises:
        ValueError: If ``fmt`` is not a supported format.
    """
    if fmt is Cylinder3DFormat.XYZRH:
        centers = raw[:, :3]
        radii = raw[:, 3]
        heights = raw[:, 4]
    else:
        msg = f"Unsupported format: {fmt}"
        raise ValueError(msg)

    return centers, radii, heights
