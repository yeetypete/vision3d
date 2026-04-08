"""Log vision3d data to a Rerun viewer."""

import math
from typing import Any

import torch

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

try:
    import rerun as rr
except ImportError as e:
    msg = "rerun-sdk is required for visualization. Install with: pip install vision3d[viz]"
    raise ImportError(msg) from e


def log_point_cloud(
    entity: str,
    points: PointCloud3D | torch.Tensor,
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
    log_heading: bool = True,
) -> None:
    """Log 3D bounding boxes to Rerun.

    Logs boxes as ``rr.Boxes3D`` and optionally heading arrows as
    ``rr.Arrows3D`` on a ``/heading`` sub-entity.

    For consistent class coloring across frames, pass ``class_ids`` and
    log an ``rr.AnnotationContext`` on the entity once before logging
    any boxes (see :func:`log_annotation_context`).

    Args:
        entity: Rerun entity path (e.g. ``"world/boxes"``).
        boxes: Bounding boxes in any supported format.
        labels: Per-box label strings for display.
        class_ids: Per-box class IDs for coloring via AnnotationContext.
        log_heading: If True and boxes have rotation, log heading arrows.
    """
    raw = boxes.as_subclass(torch.Tensor).detach().cpu()
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


def log_cameras(
    entity_prefix: str,
    images: CameraImages | torch.Tensor,
    intrinsics: CameraIntrinsics | torch.Tensor | None = None,
    extrinsics: CameraExtrinsics | torch.Tensor | None = None,
) -> None:
    """Log all camera images with optional pinhole projection to Rerun.

    Each camera is logged to ``{entity_prefix}`` for a single camera, or
    ``{entity_prefix}_{i}`` for multiple cameras.

    Args:
        entity_prefix: Rerun entity path prefix (e.g. ``"world/cam"``).
        images: Camera images ``[N_cams, C, H, W]``.
        intrinsics: Intrinsic matrices ``[N_cams, 3, 3]``.
        extrinsics: Extrinsic matrices ``[N_cams, 4, 4]`` (lidar-to-camera).
    """
    n_cams = images.shape[0]
    for i in range(n_cams):
        entity = f"{entity_prefix}_{i}" if n_cams > 1 else entity_prefix
        _log_single_camera(entity, images, intrinsics, extrinsics, camera_index=i)


def _log_single_camera(
    entity: str,
    images: CameraImages | torch.Tensor,
    intrinsics: CameraIntrinsics | torch.Tensor | None,
    extrinsics: CameraExtrinsics | torch.Tensor | None,
    *,
    camera_index: int,
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

    rr.log(entity, rr.Image(img))


def log_sample(
    inputs: dict[str, Any],
    targets: dict[str, Any] | None = None,
    *,
    entity_prefix: str = "world",
    label_to_id: dict[str, int] | None = None,
) -> None:
    """Log a full sample dict to Rerun.

    Convenience function that dispatches to type-specific loggers.

    Args:
        inputs: Dict with ``"points"``, ``"images"``, ``"extrinsics"``,
            ``"intrinsics"`` keys.
        targets: Optional dict with ``"boxes"``, ``"class_names"`` keys.
        entity_prefix: Rerun entity path prefix.
        label_to_id: Mapping from class name to class ID for consistent
            coloring. Build this across all frames before logging.
    """
    if "points" in inputs:
        log_point_cloud(f"{entity_prefix}/lidar", inputs["points"])

    if "images" in inputs:
        log_cameras(
            f"{entity_prefix}/cam",
            inputs["images"],
            inputs.get("intrinsics"),
            inputs.get("extrinsics"),
        )

    if targets and "boxes" in targets:
        class_ids = None
        labels_list = None
        if "labels" in targets:
            class_ids = targets["labels"].tolist()
        if "class_names" in targets:
            labels_list = targets["class_names"]
        elif label_to_id is not None and class_ids is not None:
            id_to_label = {v: k for k, v in label_to_id.items()}
            labels_list = [id_to_label.get(i, str(i)) for i in class_ids]

        log_boxes_3d(
            f"{entity_prefix}/boxes",
            targets["boxes"],
            labels=labels_list,
            class_ids=class_ids,
        )


def _extract_centers_sizes_yaws(
    raw: torch.Tensor, fmt: BoundingBox3DFormat
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
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
