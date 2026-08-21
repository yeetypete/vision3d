"""Logging helpers for the interactive 3D annotator.

These differ from :mod:`vision3d.viz._logging` in two ways that the annotator
needs:

* boxes are logged **one entity per box**, so a single box can be rewritten by
  the annotator without touching its neighbours (Rerun components are arrays,
  so a shared entity would require rewriting every box on every edit);
* rotations use the full 9-DoF parameterization (yaw, pitch, roll) rather than
  the yaw-only subset.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)
from vision3d.viz._blueprint import camera_grid

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError as e:
    msg = "rerun-sdk is required for visualization. Install with: pip install vision3d[viz]"
    raise ImportError(msg) from e

__all__ = [
    "annotator_layout",
    "boxes_to_map",
    "colorize_points_from_cameras",
    "lidar_to_map",
    "load_labels",
    "log_boxes_3d_editable",
    "log_ego",
    "log_labels",
    "log_point_cloud_rgb",
    "log_point_cloud_rgb_by_sweep",
    "log_source",
    "reserve_box_slots",
]

#: View class identifiers registered by the ``vision3d-annotator`` binary, one
#: per slice axis. These must match ``SliceAxis::identifier`` on the Rust side.
SLICE_VIEW_CLASSES = ("BoxSliceZ", "BoxSliceY", "BoxSliceX")

#: View class identifier for the box overview list, also registered by the
#: ``vision3d-annotator`` binary.
BOX_LIST_VIEW_CLASS = "BoxList"

#: Forked 3D view class, also registered by the annotator binary. Behaves like
#: the built-in ``Spatial3DView`` except that dragging a box moves it rather
#: than orbiting the camera.
ANNOTATE_3D_VIEW_CLASS = "Annotate3D"

# Points closer than this to a camera's plane are treated as behind it.
_MIN_DEPTH = 1e-3


def annotator_layout(
    camera_names: Sequence[str],
    grid: Sequence[Sequence[int]] | None = None,
    *,
    origin: str = "/world",
    entity_prefix: str = "world/ego/cam",
    box_entity: str = "world/annotations",
    ego: str = "world/ego",
    extra_contents: Sequence[str] = ("world/sweeps/**",),
    slice_column_share: float = 1.0,
    list_column_share: float = 0.8,
) -> rrb.Blueprint:
    """Build the annotator layout: box list left, 3D centre, slice views right.

    The left column lists every box in the frame: click a row to select it
    (the slice views follow the selection), change a box's class from the
    dropdown, or create a new box.

    To create a box, point at the target in the 3D view and press ``N``. The box
    is dropped at the hovered position, sized 5x5x5 m, and selected, ready to be
    trimmed to shape in the slice views.

    The right column holds the three box-frame slice views, which is where
    manipulation happens. Each is locked to the *selected* box's own coordinate
    frame, so the box appears as an axis-aligned rectangle:

    * ``BoxSliceZ`` looks down -Z (bird's eye): drag to translate in local XY,
      drag edges to change length/width, drag the handle to change yaw.
    * ``BoxSliceY`` looks along -Y: local XZ, and pitch.
    * ``BoxSliceX`` looks along -X: local YZ, and roll.

    Together the three cover all 9 degrees of freedom.

    Requires the ``vision3d-annotator`` binary rather than the stock Rerun
    viewer, since the slice view classes are registered by that binary. In the
    stock viewer the three left-hand views will show as unknown class.

    Args:
        camera_names: Per-camera display names indexed by tensor position.
        grid: Row-major grid of indices into ``camera_names``, as accepted by
            :func:`vision3d.viz.camera_grid`.
        origin: Entity path the 3D and slice views are rooted at.
        entity_prefix: Prefix for camera entity origins.
        box_entity: Entity prefix the editable boxes were logged under by
            :func:`log_boxes_3d_editable`.
        ego: Entity carrying the ego pose. The 3D view is rooted here, so sensor
            data renders in its raw frame and the view rides with the machine,
            while map-frame annotations are transformed into it.
        extra_contents: Further entity expressions to include in the 3D view.
            Anything outside the ego subtree, such as map-frame lidar sweeps,
            has to be named explicitly.
        slice_column_share: Width share of the right-hand slice column relative
            to the main column, which is fixed at 3.
        list_column_share: Width share of the box-list column.

    Returns:
        A :class:`~rerun.blueprint.Blueprint` with the full annotator layout.
    """
    slices = [
        rrb.View(
            class_identifier=class_id,
            origin=origin,
            contents=[f"{origin}/**"],
            name=name,
        )
        for class_id, name in zip(
            SLICE_VIEW_CLASSES,
            ("Top (yaw)", "Front (pitch)", "Side (roll)"),
            strict=True,
        )
    ]

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.View(
                class_identifier=BOX_LIST_VIEW_CLASS,
                origin=origin,
                contents=[f"{box_entity}/**"],
                name="Boxes",
            ),
            rrb.Vertical(
                rrb.View(
                    class_identifier=ANNOTATE_3D_VIEW_CLASS,
                    origin=ego,
                    contents=[f"{ego}/**", f"{box_entity}/**", *extra_contents],
                    name="3D",
                ),
                camera_grid(
                    camera_names,
                    grid,
                    entity_prefix=entity_prefix,
                    overlay_entities=(box_entity,),
                ),
                row_shares=[2, 1],
            ),
            rrb.Vertical(*slices),
            column_shares=[list_column_share, 3.0, slice_column_share],
        ),
        collapse_panels=True,
    )


def lidar_to_map(
    dataset: Any, index: int, anchor: Tensor | None = None
) -> Tensor | None:
    """Transform from a sample's lidar frame to the map frame.

    Mirrors the dataset's own ``lidar_to_global`` composition (ego pose times the
    lidar's calibrated extrinsics) by reading the same nuScenes-layout tables.
    :class:`~vision3d.datasets.SampleInputs` doesn't carry it, hence the reach
    into the dataset's devkit handle.

    Args:
        dataset: A nuScenes-layout dataset.
        index: Sample index.
        anchor: Optional translation to subtract, keeping map coordinates near
            the origin. Global coordinates run to hundreds of metres, where
            float32 starts costing centimetres of precision.

    Returns:
        A ``[4, 4]`` homogeneous transform, or ``None`` if the tables could not
        be read.
    """
    try:
        from vision3d.datasets.nuscenes import _make_transform

        nusc = dataset._nusc
        sample = nusc.get("sample", dataset._sample_tokens[index])
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar["ego_pose_token"])
        calib = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
    except (AttributeError, KeyError, ImportError):
        return None

    transform = _make_transform(ego["translation"], ego["rotation"]) @ _make_transform(
        calib["translation"], calib["rotation"]
    )
    if anchor is not None:
        transform[:3, 3] -= anchor
    return transform


def log_ego(entity: str, lidar_from_map: Tensor, axis_length: float = 2.0) -> None:
    """Log the sensor's pose in the map frame, as a trackable entity.

    With geometry in the map frame the machine itself is no longer at the origin
    and is otherwise invisible. Logging its pose gives you something to see and,
    more usefully, something for the 3D view to track, so the camera can follow
    the machine while annotations stay fixed in the world.

    Args:
        entity: Entity path for the sensor pose (e.g. ``"world/ego"``).
        lidar_from_map: The ``[4, 4]`` lidar-to-map transform for this frame.
        axis_length: Length of the drawn axes, in metres.
    """
    rr.log(
        entity,
        rr.Transform3D(
            translation=lidar_from_map[:3, 3],
            mat3x3=lidar_from_map[:3, :3],
        ),
    )
    # Drawn as a child so it inherits the pose above and renders in the
    # machine's own frame; this SDK's Transform3D has no axis_length of its own.
    rr.log(
        f"{entity}/axes",
        rr.Arrows3D(
            origins=[[0.0, 0.0, 0.0]] * 3,
            vectors=[
                [axis_length, 0.0, 0.0],
                [0.0, axis_length, 0.0],
                [0.0, 0.0, axis_length],
            ],
            colors=[[229, 86, 86], [86, 229, 122], [86, 132, 229]],
        ),
    )


def boxes_to_map(boxes: BoundingBoxes3D, lidar_from_map: Tensor) -> BoundingBoxes3D:
    """Re-express boxes in the map frame.

    Only the boxes move. Points and camera extrinsics stay in the ego frame and
    are logged under an ego node carrying this same transform, so the 3D view can
    be rooted there and render them in their raw coordinates -- the view then
    rides with the machine, while annotations stay put in the world.

    Args:
        boxes: Boxes in the lidar frame.
        lidar_from_map: The ``[4, 4]`` lidar-to-map transform.

    Returns:
        The boxes in the map frame.

    Note:
        Orientation is composed in yaw only. Ego roll and pitch are small for
        ground machines, and the 7-DoF box formats carry no roll/pitch to compose
        into; a general composition would have to go via rotation matrices.
    """
    rot = lidar_from_map[:3, :3]
    trans = lidar_from_map[:3, 3]

    raw = boxes.as_subclass(Tensor).clone()
    if raw.shape[0]:
        raw[:, :3] = raw[:, :3] @ rot.T + trans
        if raw.shape[1] >= 7:
            raw[:, 6] += float(torch.atan2(rot[1, 0], rot[0, 0]))

    return BoundingBoxes3D(raw, format=boxes.format)


def log_source(path: str, entity: str = "meta/source") -> None:
    """Record which file this recording came from.

    The annotator writes its sidecar next to this path. Without it the tool has
    no idea what it is looking at and cannot choose an export location.

    Args:
        path: Source file, typically the bag.
        entity: Entity to record it at; must match the annotator's
            ``export::SOURCE_ENTITY``.
    """
    rr.log(entity, rr.TextDocument(str(path)), static=True)


def load_labels(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Read a sidecar written by the annotator.

    Args:
        path: A ``.labels.jsonl`` file.

    Returns:
        ``(records, class name to id)``. The class map comes from the header, so
        reloading cannot silently renumber classes -- ids are baked into the
        exported records and into every colour derived from them.

    Raises:
        ValueError: If the header is missing or of an unknown schema.
    """
    records: list[dict] = []
    classes: dict[str, int] = {}

    with path.open() as handle:
        header = json.loads(handle.readline() or "{}")
        if not str(header.get("schema", "")).startswith("vision3d.annotations/"):
            msg = f"{path} has no recognisable annotation header"
            raise ValueError(msg)
        for entry in header.get("classes", []):
            classes[entry["name"]] = entry["id"]
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records, classes


def log_labels(
    entity_prefix: str,
    records: list[dict],
    classes: dict[str, int],
    *,
    fill_mode: rr.components.FillModeLike | None = None,
) -> int:
    """Re-log exported annotations as editable boxes.

    Track names are preserved exactly, which is what lets a correction session
    edit the same objects rather than create duplicates alongside them.

    Static records are logged without a timeline, matching how the annotator
    marks an object that does not move.

    Args:
        entity_prefix: Where boxes live (e.g. ``"world/annotations"``).
        records: Rows from :func:`load_labels`.
        classes: Class name to id, for the annotation context.
        fill_mode: Box fill mode. Left unset by default, which is what a box
            created in the annotator carries -- the viewer then falls back to
            ``TransparentFillMajorWireframe``. Forcing a wireframe here leaves
            the box with no face for the picker to hit, so a reloaded box could
            not be dragged in the 3D view; the fill is transparent, so it does
            not hide the points either way.

    Returns:
        The number of boxes logged.
    """
    if classes:
        rr.log(
            entity_prefix,
            rr.AnnotationContext([(i, name) for name, i in classes.items()]),
            static=True,
        )

    for record in records:
        path = f"{entity_prefix}/{record['track']}"
        box = rr.Boxes3D(
            centers=[record["center"]],
            half_sizes=[record["half_size"]],
            quaternions=[rr.Quaternion(xyzw=record["quat"])],
            class_ids=None if record.get("class_id") is None else [record["class_id"]],
            labels=None if record.get("class") is None else [record["class"]],
            fill_mode=fill_mode,
        )
        if record.get("static") or record.get("t") is None:
            rr.log(path, box, static=True)
        else:
            # Absolute, matching the feed: a duration index here would place
            # reloaded boxes 56 years before the point clouds.
            rr.set_time("time", timestamp=np.datetime64(int(record["t"]), "ns"))
            rr.log(path, box)

    return len(records)


def reserve_box_slots(
    entity_prefix: str = "world/annotations",
    count: int = 64,
    *,
    slot_prefix: str = "new",
) -> None:
    """Pre-register entity paths that the annotator can create boxes into.

    The Rerun viewer decides which entities a visualizer may draw from an index
    built out of *schema-addition* events, and that index is additive-only. An
    entity path that first appears after the viewer is running -- which is what
    happens when the annotator creates a box in-viewer -- is written to the store
    successfully but never becomes visualizable, so nothing renders it.

    Working around that means making the paths exist up front. Each slot is
    logged with a throwaway box and immediately cleared: the clear removes the
    data, but the schema registration it leaves behind is permanent, which is
    exactly what the viewer's index needs.

    Call this once, at a time strictly before the first real frame, so the clear
    cannot mask a box the annotator later writes.

    Args:
        entity_prefix: Prefix the annotator creates boxes under.
        count: How many slots to reserve. The annotator names created boxes
            ``new_0``, ``new_1``, ... so this is the ceiling on how many boxes
            can be created per session.
        slot_prefix: Leaf-name prefix, matching the annotator's own naming.
    """
    for i in range(count):
        path = f"{entity_prefix}/{slot_prefix}_{i}"
        rr.log(path, rr.Boxes3D(centers=[(0.0, 0.0, 0.0)], sizes=[(0.01, 0.01, 0.01)]))
        rr.log(path, rr.Clear(recursive=True))


def _to_uint8_images(images: CameraImages | Tensor) -> Tensor:
    """Normalize a camera image batch to ``[N, H, W, 3]`` uint8.

    Args:
        images: Camera images ``[N_cams, C, H, W]``, float in ``[0, 1]``,
            float in ``[0, 255]``, or uint8.

    Returns:
        Images as ``[N_cams, H, W, 3]`` uint8 on the CPU.
    """
    img = images.detach().cpu()
    if img.is_floating_point():
        img = (img * 255).to(torch.uint8) if img.max() <= 1.0 else img.to(torch.uint8)
    img = img.permute(0, 2, 3, 1)
    if img.shape[-1] == 1:
        img = img.expand(-1, -1, -1, 3)
    return img[..., :3].contiguous()


def colorize_points_from_cameras(
    points: PointCloud3D | Tensor,
    images: CameraImages | Tensor,
    intrinsics: CameraIntrinsics | Tensor,
    extrinsics: CameraExtrinsics | Tensor,
    *,
    fallback: tuple[int, int, int] = (110, 110, 110),
) -> Tensor:
    """Assign an RGB color to every point by projecting it into the cameras.

    Each point is projected into every camera using ``extrinsics``
    (source-frame-to-camera) followed by ``intrinsics``. Points landing inside
    more than one image are colored from the camera that sees them closest,
    which keeps overlapping fields of view deterministic. Points seen by no
    camera get ``fallback``.

    Args:
        points: Point cloud ``[N, 3+C]``; the first three columns are xyz in
            the same source frame the extrinsics map from.
        images: Camera images ``[N_cams, C, H, W]``.
        intrinsics: Intrinsic matrices ``[N_cams, 3, 3]``.
        extrinsics: Extrinsic matrices ``[N_cams, 4, 4]``, source-to-camera.
        fallback: RGB color for points no camera sees.

    Returns:
        Per-point colors as ``[N, 3]`` uint8 on the CPU.

    Raises:
        ValueError: If the camera count differs between ``images``,
            ``intrinsics``, and ``extrinsics``.
    """
    xyz = points[:, :3].detach().cpu().to(torch.float32)
    imgs = _to_uint8_images(images)
    K = intrinsics.detach().cpu().to(torch.float32)
    E = extrinsics.detach().cpu().to(torch.float32)

    n_cams = imgs.shape[0]
    if K.shape[0] != n_cams or E.shape[0] != n_cams:
        msg = (
            f"camera count mismatch: images has {n_cams}, intrinsics has "
            f"{K.shape[0]}, extrinsics has {E.shape[0]}"
        )
        raise ValueError(msg)

    n = xyz.shape[0]
    colors = torch.tensor(fallback, dtype=torch.uint8).expand(n, 3).clone()
    best_depth = torch.full((n,), float("inf"))

    for i in range(n_cams):
        # Source frame -> camera frame.
        p_cam = xyz @ E[i, :3, :3].T + E[i, :3, 3]
        depth = p_cam[:, 2]
        in_front = depth > _MIN_DEPTH

        # Camera frame -> pixels. Guard the division so points behind the
        # camera cannot produce in-bounds pixel coordinates.
        safe_depth = torch.where(in_front, depth, torch.ones_like(depth))
        uvw = p_cam @ K[i].T
        u = (uvw[:, 0] / safe_depth).round().long()
        v = (uvw[:, 1] / safe_depth).round().long()

        h, w = imgs.shape[1:3]
        visible = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        # Only take over a point if this camera sees it closer than whichever
        # camera claimed it before.
        take = visible & (depth < best_depth)
        if not bool(take.any()):
            continue

        idx = take.nonzero(as_tuple=True)[0]
        colors[idx] = imgs[i, v[idx], u[idx]]
        best_depth[idx] = depth[idx]

    return colors


def log_point_cloud_rgb(
    entity: str,
    points: PointCloud3D | Tensor,
    images: CameraImages | Tensor,
    intrinsics: CameraIntrinsics | Tensor,
    extrinsics: CameraExtrinsics | Tensor,
    *,
    radii: float | None = None,
    fallback: tuple[int, int, int] = (110, 110, 110),
) -> None:
    """Log a point cloud colored from the camera images.

    Thin wrapper around :func:`colorize_points_from_cameras` that logs the
    result as ``rr.Points3D``.

    Args:
        entity: Rerun entity path (e.g. ``"world/lidar"``).
        points: Point cloud ``[N, 3+C]``. First three columns are xyz.
        images: Camera images ``[N_cams, C, H, W]``.
        intrinsics: Intrinsic matrices ``[N_cams, 3, 3]``.
        extrinsics: Extrinsic matrices ``[N_cams, 4, 4]``, source-to-camera.
        radii: Point radius in scene units. ``None`` leaves Rerun's default.
        fallback: RGB color for points no camera sees.
    """
    xyz = points[:, :3].detach().cpu()
    colors = colorize_points_from_cameras(
        points, images, intrinsics, extrinsics, fallback=fallback
    )
    rr.log(entity, rr.Points3D(xyz, colors=colors, radii=radii))


def log_point_cloud_rgb_by_sweep(
    entity: str,
    points: PointCloud3D | Tensor,
    images: CameraImages | Tensor,
    intrinsics: CameraIntrinsics | Tensor,
    extrinsics: CameraExtrinsics | Tensor,
    *,
    radii: float | None = None,
    fallback: tuple[int, int, int] = (110, 110, 110),
) -> int:
    """Log a multi-sweep cloud as one entity per sweep.

    Datasets loaded with ``num_sweeps > 1`` carry a trailing time-offset column:
    seconds before the key frame. Grouping on it and logging each group to
    ``{entity}/sweep_{k}`` (``sweep_0`` being the key frame) means each sweep can
    be shown or hidden on its own from the viewer's blueprint tree, which is the
    interactive control over how much accumulated cloud you see.

    Falls back to a single :func:`log_point_cloud_rgb` call for single-sweep
    clouds, which have no such column.

    Args:
        entity: Rerun entity path prefix (e.g. ``"world/lidar"``).
        points: Point cloud ``[N, 3+C]``. First three columns are xyz; a 6th
            column, when present, is the per-point time offset.
        images: Camera images ``[N_cams, C, H, W]``.
        intrinsics: Intrinsic matrices ``[N_cams, 3, 3]``.
        extrinsics: Extrinsic matrices ``[N_cams, 4, 4]``, source-to-camera.
        radii: Point radius in scene units.
        fallback: RGB color for points no camera sees.

    Returns:
        The number of sweeps logged.
    """
    raw = points[:, :].detach().cpu()
    if raw.shape[1] < 6:
        log_point_cloud_rgb(
            entity,
            points,
            images,
            intrinsics,
            extrinsics,
            radii=radii,
            fallback=fallback,
        )
        return 1

    colors = colorize_points_from_cameras(
        points, images, intrinsics, extrinsics, fallback=fallback
    )

    # Ascending offset means sweep_0 is the key frame.
    offsets = torch.unique(raw[:, 5]).sort().values
    for k, offset in enumerate(offsets):
        mask = raw[:, 5] == offset
        rr.log(
            f"{entity}/sweep_{k}",
            rr.Points3D(raw[mask, :3], colors=colors[mask], radii=radii),
        )

    return len(offsets)


def _quaternion_from_ypr(yaw: float, pitch: float, roll: float) -> rr.Quaternion:
    """Build an xyzw quaternion from intrinsic Tait-Bryan ZY'X'' angles.

    Args:
        yaw: Rotation about Z in radians.
        pitch: Rotation about the once-rotated Y in radians.
        roll: Rotation about the twice-rotated X in radians.

    Returns:
        The equivalent :class:`rerun.Quaternion`.
    """
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

    return rr.Quaternion(
        xyzw=[
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * cr + sy * sp * sr,
        ]
    )


def _extract_9dof(
    raw: Tensor, fmt: BoundingBox3DFormat
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract centers, sizes, and full yaw/pitch/roll from a raw box tensor.

    Unlike the yaw-only extraction used by :func:`vision3d.viz.log_boxes_3d`,
    this preserves pitch and roll for :attr:`BoundingBox3DFormat.XYZLWHYPR`.

    Args:
        raw: Raw box tensor ``[N, D]``.
        fmt: Format of ``raw``.

    Returns:
        Tuple of ``(centers [N, 3], sizes [N, 3], ypr [N, 3])``.

    Raises:
        ValueError: If ``fmt`` is not a supported format.
    """
    n = raw.shape[0]
    if fmt is BoundingBox3DFormat.XYZXYZ:
        mins, maxs = raw[:, :3], raw[:, 3:]
        return (mins + maxs) / 2, maxs - mins, torch.zeros(n, 3)
    if fmt is BoundingBox3DFormat.XYZLWH:
        return raw[:, :3], raw[:, 3:6], torch.zeros(n, 3)
    if fmt is BoundingBox3DFormat.XYZLWHY:
        ypr = torch.zeros(n, 3)
        ypr[:, 0] = raw[:, 6]
        return raw[:, :3], raw[:, 3:6], ypr
    if fmt is BoundingBox3DFormat.XYZLWHYPR:
        return raw[:, :3], raw[:, 3:6], raw[:, 6:9]

    msg = f"Unsupported format: {fmt}"
    raise ValueError(msg)


def log_boxes_3d_editable(
    entity_prefix: str,
    boxes: BoundingBoxes3D,
    *,
    labels: list[str] | None = None,
    class_ids: list[int] | None = None,
    label_to_id: dict[str, int] | None = None,
    box_ids: list[str] | None = None,
    fill_mode: rr.components.FillModeLike | None = None,
) -> list[str]:
    """Log 3D boxes one entity per box so the annotator can edit them singly.

    Each box lands at ``{entity_prefix}/{box_id}`` carrying a single-element
    ``rr.Boxes3D``. The annotator rewrites exactly that entity when a box is
    manipulated, which keeps an edit from disturbing any other box.

    Rotations use the full 9-DoF parameterization: for
    :attr:`BoundingBox3DFormat.XYZLWHYPR` inputs, pitch and roll are preserved
    rather than dropped.

    Args:
        entity_prefix: Path prefix for the per-box entities
            (e.g. ``"world/annotations"``).
        boxes: Bounding boxes in any supported format.
        labels: Per-box label strings for display.
        class_ids: Per-box class IDs for coloring via AnnotationContext.
        label_to_id: Mapping from class name to class ID. When provided, an
            ``rr.AnnotationContext`` is logged statically on ``entity_prefix``.
        box_ids: Stable per-box identifiers used as the entity leaf name. These
            are the annotator's notion of object identity across frames, so
            they should be stable over time. Defaults to ``"box_0"``, ``"box_1"``…
        fill_mode: Box fill mode. Left unset by default, so the viewer's
            ``TransparentFillMajorWireframe`` applies -- the same thing a box
            created in the annotator gets. A wireframe-only box has no face for
            the picker, which makes it undraggable in a 3D view, and the default
            fill is transparent, so it does not obscure the points either.

    Returns:
        The entity path of every box that was logged, in input order.

    Raises:
        ValueError: If ``labels``, ``class_ids``, or ``box_ids`` length does not
            match the number of boxes.
    """
    raw = boxes.as_subclass(Tensor).detach().cpu()
    n = raw.shape[0]

    for name, seq in (
        ("labels", labels),
        ("class_ids", class_ids),
        ("box_ids", box_ids),
    ):
        if seq is not None and len(seq) != n:
            msg = f"{name} has length {len(seq)} but there are {n} boxes"
            raise ValueError(msg)

    if label_to_id is not None:
        rr.log(
            entity_prefix,
            rr.AnnotationContext([(i, name) for name, i in label_to_id.items()]),
            static=True,
        )

    if box_ids is None:
        box_ids = [f"box_{i}" for i in range(n)]

    centers, sizes, ypr = _extract_9dof(raw, boxes.format)

    paths = []
    for i in range(n):
        path = f"{entity_prefix}/{box_ids[i]}"
        rr.log(
            path,
            rr.Boxes3D(
                centers=centers[i : i + 1],
                sizes=sizes[i : i + 1],
                quaternions=[
                    _quaternion_from_ypr(
                        float(ypr[i, 0]), float(ypr[i, 1]), float(ypr[i, 2])
                    )
                ],
                class_ids=None if class_ids is None else class_ids[i : i + 1],
                labels=None if labels is None else labels[i : i + 1],
                fill_mode=fill_mode,
            ),
        )
        paths.append(path)

    return paths
