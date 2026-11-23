"""Log vision3d data to a Rerun viewer."""

import torch
from torch import Tensor

from vision3d.datasets import SampleInputs, SampleTargets
from vision3d.metrics import Prediction3D
from vision3d.ops import extract_box3d_params
from vision3d.ops._rotation import _rotation_matrix_to_quaternion
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
        colors = torch.zeros(len(xyz), 4, dtype=torch.uint8, device=xyz.device)
        colors[:, 0] = (normalized * 255).to(torch.uint8)
        colors[:, 2] = ((1 - normalized) * 255).to(torch.uint8)
        colors[:, 3] = 255

    rr.log(
        entity,
        rr.Points3D.from_fields(
            # Rerun otherwise keeps stale components, such as earlier colors.
            clear_unset=True,
            positions=xyz,
            colors=colors,
        ),
    )


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
    # Rerun cannot ingest low-precision dtypes such as bfloat16. Keep float64
    # so that sizes stay exact.
    raw = raw.to(torch.promote_types(raw.dtype, torch.float32))
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

    centers, half_dims, rot = extract_box3d_params(raw, fmt)
    # Rerun expects (x, y, z, w). Ours are (w, x, y, z).
    quaternions = _rotation_matrix_to_quaternion(rot)[:, [1, 2, 3, 0]]

    rr.log(
        entity,
        rr.Boxes3D.from_fields(
            # Rerun otherwise keeps stale components, such as earlier labels.
            clear_unset=True,
            centers=centers,
            half_sizes=half_dims,
            quaternions=quaternions,
            class_ids=class_ids,
            labels=display_labels,
            fill_mode=fill_mode,
            show_labels=show_labels,
        ),
    )

    if log_heading and BoundingBox3DFormat.is_rotated(fmt):
        face_scale = 2 * torch.sqrt(half_dims[:, 1] * half_dims[:, 2])
        # The box's local +x axis in world coordinates.
        forward = rot[:, :, 0]
        origins = centers + half_dims[:, :1] * forward
        vectors = (face_scale * 0.6)[:, None] * forward

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
    else:
        rr.log(f"{entity}/heading", rr.Clear(recursive=False))


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
    predictions: Prediction3D | None = None,
    entity_prefix: str = "world",
    label_to_id: dict[str, int] | None = None,
    score_threshold: float | None = None,
    jpeg_quality: int | None = None,
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
            ``"boxes"``, ``"labels"`` keys (ground truth).
        predictions: Optional :class:`~vision3d.metrics.Prediction3D` with
            ``"boxes"``, ``"scores"``, ``"labels"`` keys. Prediction labels
            show their score.
        entity_prefix: Rerun entity path prefix.
        label_to_id: Mapping from class name to class ID for consistent
            coloring. Build this across all frames before logging.
        score_threshold: If set, predictions below this score are dropped.
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
            f"{entity_prefix}/gt/boxes",
            targets["boxes"],
            class_ids=class_ids,
            label_to_id=label_to_id,
            fill_mode="transparentfillmajorwireframe",
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
