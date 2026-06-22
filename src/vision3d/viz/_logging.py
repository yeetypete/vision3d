"""Log vision3d data to a Rerun viewer."""

import math
import numbers
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Self, SupportsFloat

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
    PointCloud3D,
)

try:
    import rerun as rr
except ImportError as e:
    msg = "rerun-sdk is required for visualization. Install with: pip install vision3d[viz]"
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from pathlib import Path

    from rerun import RecordingStream
    from rerun.blueprint import BlueprintLike


class LoggingInputError(ValueError):
    """Raised when logging is called with malformed input.

    Signals a caller bug -- a non-scalar metric, mismatched per-box list
    lengths, an unsupported box format -- as opposed to a transient sink or
    transport failure. Subclasses :class:`ValueError`, so existing
    ``except ValueError`` handlers still catch it, but lets
    :class:`RerunLogger` tell a usage bug apart from a visualization hiccup
    and re-raise it even in best-effort (non-``strict``) mode.
    """


def log_point_cloud(
    entity: str,
    points: PointCloud3D | Tensor,
    *,
    color_by_distance: bool = True,
    static: bool = False,
    recording: "RecordingStream | None" = None,
) -> None:
    """Log a point cloud to Rerun.

    Args:
        entity: Rerun entity path (e.g. ``"world/lidar"``).
        points: Point cloud ``[N, 3+C]``. First 3 columns are xyz.
        color_by_distance: Color points by distance from origin.
        static: Log without a timeline so the cloud shows at every point on
            every timeline. Use for geometry that is constant across a
            recording, such as a fixed sample inspected over training steps.
        recording: Target recording. ``None`` (default) uses Rerun's active
            global recording.
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

    rr.log(entity, rr.Points3D(xyz, colors=colors), static=static, recording=recording)


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
    static: bool = False,
    recording: "RecordingStream | None" = None,
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
        static: Log without a timeline so the boxes show at every point on
            every timeline. Use for ground truth on a fixed sample inspected
            over training steps, where only the predictions change over time.
        recording: Target recording. ``None`` (default) uses Rerun's active
            global recording.

    Raises:
        LoggingInputError: If ``score_threshold`` is set without ``scores``,
            or if ``scores``, ``labels``, or ``class_ids`` length does not
            match the number of boxes.
    """
    if label_to_id is not None:
        rr.log(
            entity,
            rr.AnnotationContext([(i, name) for name, i in label_to_id.items()]),
            static=True,
            recording=recording,
        )

    raw = boxes.as_subclass(Tensor).detach().cpu()
    fmt = boxes.format

    score_list = (
        scores.detach().cpu().tolist() if isinstance(scores, Tensor) else scores
    )
    if score_list is not None and len(score_list) != raw.shape[0]:
        msg = f"scores has length {len(score_list)} but there are {raw.shape[0]} boxes"
        raise LoggingInputError(msg)
    if labels is not None and len(labels) != raw.shape[0]:
        msg = f"labels has length {len(labels)} but there are {raw.shape[0]} boxes"
        raise LoggingInputError(msg)
    if class_ids is not None and len(class_ids) != raw.shape[0]:
        msg = (
            f"class_ids has length {len(class_ids)} but there are {raw.shape[0]} boxes"
        )
        raise LoggingInputError(msg)
    if score_threshold is not None:
        if score_list is None:
            msg = "score_threshold requires scores"
            raise LoggingInputError(msg)
        keep = [i for i, s in enumerate(score_list) if s >= score_threshold]
        raw = raw[keep]
        score_list = [score_list[i] for i in keep]
        if class_ids is not None:
            class_ids = [class_ids[i] for i in keep]
        if labels is not None:
            labels = [labels[i] for i in keep]

    n = raw.shape[0]

    if n == 0:
        rr.log(entity, rr.Clear(recursive=True), static=static, recording=recording)
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
        static=static,
        recording=recording,
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
            static=static,
            recording=recording,
        )


def log_cameras(
    entity_prefix: str,
    images: CameraImages | Tensor,
    intrinsics: CameraIntrinsics | Tensor | None = None,
    extrinsics: CameraExtrinsics | Tensor | None = None,
    *,
    jpeg_quality: int | None = None,
    recording: "RecordingStream | None" = None,
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
        recording: Target recording. ``None`` (default) uses Rerun's active
            global recording.
    """
    for i in range(images.shape[0]):
        _log_single_camera(
            f"{entity_prefix}_{i}",
            images,
            intrinsics,
            extrinsics,
            camera_index=i,
            jpeg_quality=jpeg_quality,
            recording=recording,
        )


def _log_single_camera(
    entity: str,
    images: CameraImages | Tensor,
    intrinsics: CameraIntrinsics | Tensor | None,
    extrinsics: CameraExtrinsics | Tensor | None,
    *,
    camera_index: int,
    jpeg_quality: int | None = None,
    recording: "RecordingStream | None" = None,
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
            recording=recording,
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
            recording=recording,
        )

    archetype = rr.Image(img)
    if jpeg_quality is not None:
        archetype = archetype.compress(jpeg_quality=jpeg_quality)
    rr.log(entity, archetype, recording=recording)


def log_sample(
    inputs: SampleInputs,
    targets: SampleTargets | None = None,
    *,
    predictions: Prediction3D | None = None,
    entity_prefix: str = "world",
    label_to_id: dict[str, int] | None = None,
    score_threshold: float | None = None,
    jpeg_quality: int | None = None,
    recording: "RecordingStream | None" = None,
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
        recording: Target recording. ``None`` (default) uses Rerun's active
            global recording.
    """
    if "points" in inputs:
        log_point_cloud(f"{entity_prefix}/lidar", inputs["points"], recording=recording)

    if "images" in inputs:
        log_cameras(
            f"{entity_prefix}/cam",
            inputs["images"],
            inputs.get("intrinsics"),
            inputs.get("extrinsics"),
            jpeg_quality=jpeg_quality,
            recording=recording,
        )

    if targets and "boxes" in targets:
        class_ids = targets["labels"].tolist() if "labels" in targets else None
        log_boxes_3d(
            f"{entity_prefix}/gt/boxes",
            targets["boxes"],
            class_ids=class_ids,
            label_to_id=label_to_id,
            fill_mode="transparentfillmajorwireframe",
            recording=recording,
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
            recording=recording,
        )


def log_scalars(
    values: Mapping[str, SupportsFloat | Tensor],
    *,
    step: int | None = None,
    epoch: int | None = None,
    prefix: str = "train",
    recording: "RecordingStream | None" = None,
) -> None:
    """Log scalar training metrics to Rerun.

    Each entry in ``values`` is logged as an :class:`rerun.Scalars` archetype
    to its own entity (``{prefix}/{name}``), forming a time series that Rerun
    plots in a :class:`~rerun.blueprint.TimeSeriesView` (see
    :func:`vision3d.viz.time_series_view`). This is the primitive for tracking
    quantities such as loss, learning rate, or validation metrics over a
    training run. Call it once per logging point (e.g. every optimizer step)
    from a training loop, which lives outside this repo and supplies ``step``
    and ``epoch`` from its own counters.

    Metric names may contain ``/`` to build a nested entity hierarchy, e.g.
    ``{"loss/total": ..., "loss/cls": ...}`` groups under ``{prefix}/loss``.

    Args:
        values: Mapping from metric name to scalar value (Python or numpy
            number, or single-element tensor). Tensor values must hold a
            single element and are moved to the CPU before logging.
        step: Optimizer/iteration step. When given, scalars are logged on a
            ``"step"`` timeline so they align by iteration.
        epoch: Training epoch. When given, scalars are logged on an
            ``"epoch"`` timeline so they align by epoch. May be combined with
            ``step`` to log on both timelines at once.
        prefix: Entity path prefix grouping these metrics (e.g. ``"train"``,
            ``"val"``). Pass ``""`` to log each metric at the root.
        recording: Target recording. ``None`` (default) uses Rerun's active
            global recording; pass an explicit stream to avoid relying on
            global state (see :class:`RerunLogger`).

    Note:
        ``step``/``epoch`` move the recording's timeline cursor, which
        persists: a later ``rr.log`` with no explicit time lands on the last
        value set here. Pass ``step`` on every call (or use
        :class:`RerunLogger`) to keep things aligned.
    """
    if step is not None:
        rr.set_time("step", sequence=step, recording=recording)
    if epoch is not None:
        rr.set_time("epoch", sequence=epoch, recording=recording)

    for name, value in values.items():
        entity = f"{prefix}/{name}" if prefix else name
        rr.log(entity, rr.Scalars(_scalar_value(name, value)), recording=recording)


def style_series(
    entity: str,
    *,
    name: str | None = None,
    color: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
    width: float | None = None,
    recording: "RecordingStream | None" = None,
) -> None:
    """Style a scalar time series for plotting.

    Logs an :class:`rerun.SeriesLines` archetype statically on ``entity`` to
    control how the curve produced by :func:`log_scalars` appears in a
    :class:`~rerun.blueprint.TimeSeriesView`. This is most useful for
    overlaying multiple training runs in one plot: give each run's series a
    stable legend name and a distinct color so they can be told apart. Route
    each run to its own entity prefix (e.g. ``log_scalars(..., prefix=
    "runs/baseline/train")``) and style the matching entity here.

    Call once, before or after logging the series; the style is static so it
    applies across the whole recording.

    Args:
        entity: Rerun entity path of the scalar series to style (e.g.
            ``"runs/baseline/train/loss/total"``), matching the entity
            :func:`log_scalars` writes to.
        name: Legend name for the series. ``None`` keeps Rerun's
            entity-derived default.
        color: RGB or RGBA color (0-255 per channel). ``None`` lets Rerun
            auto-assign a color.
        width: Line width in points. ``None`` uses Rerun's default.
        recording: Target recording. ``None`` (default) uses Rerun's active
            global recording.
    """
    rr.log(
        entity,
        rr.SeriesLines(names=name, colors=color, widths=width),
        static=True,
        recording=recording,
    )


class RerunLogger:
    """High-level Rerun logger for training metrics.

    Wraps the Rerun recording lifecycle -- initialization, sink selection, an
    optional dashboard blueprint, rank-aware disabling, and flushing -- around
    :func:`log_scalars` and :func:`style_series`, so a training loop can log
    metrics in a single line. Training lives outside vision3d; this is the
    recommended entry point for wiring vision3d's Rerun logging into it.

    Choose at most one sink:

    * ``save_path`` -- write an ``.rrd`` file (typical for headless or cluster
      runs; open it later with ``rerun <file>.rrd``).
    * ``grpc_url`` -- stream to a running Rerun viewer/server.
    * ``spawn`` -- launch a local viewer and stream to it (interactive use).

    With none of them the recording is buffered in memory only.

    For distributed training, pass ``rank`` (e.g.
    ``torch.distributed.get_rank()``): the logger becomes a no-op on every
    non-zero rank and never touches Rerun there, so it can be called
    unconditionally from shared loop code. ``enabled=False`` disables it
    everywhere (e.g. for debug runs).

    Logging is **best-effort by default**: an *operational* failure in any
    logging call (e.g. a dropped connection or a full disk) is caught and
    suppressed with a single warning, so a visualization hiccup never crashes a
    long training run. Pass ``strict=True`` to let such errors propagate
    instead. *Input* errors are treated differently: malformed arguments (a
    non-scalar metric, mismatched per-box list lengths, ...) raise
    :class:`LoggingInputError` and always propagate, even in best-effort mode,
    since they signal a caller bug that suppression would only hide.

    Record the run's hyperparameters once with :meth:`log_config` so they
    travel with the recording -- essential context when comparing runs later.

    Example:
        >>> logger = RerunLogger(  # doctest: +SKIP
        ...     "bevfusion", save_path="run.rrd", rank=rank
        ... )
        >>> logger.log_config({"lr": 1e-3, "batch_size": 4})  # doctest: +SKIP
        >>> for step, batch in enumerate(loader):  # doctest: +SKIP
        ...     loss = train_step(batch)
        ...     logger.log({"loss/total": loss, "lr": lr}, step=step, every=50)
        >>> logger.close()  # doctest: +SKIP

    Note:
        Logging a tensor value calls ``.item()``, which synchronizes the GPU.
        In a hot loop, log every N steps with ``every=`` to avoid stalling on
        every iteration.

    Args:
        name: Rerun application id for the recording.
        save_path: Path to write an ``.rrd`` file to.
        grpc_url: URL of a Rerun gRPC sink to stream to.
        spawn: Launch a local Rerun viewer and stream to it.
        blueprint: Optional dashboard layout, built from
            :func:`time_series_view` and :func:`lidar_view`. When ``None``,
            Rerun auto-arranges views.
        prefix: Entity-path namespace prepended to every metric (e.g.
            ``"runs/baseline"`` to keep one run's curves together for
            cross-run comparison). Empty by default.
        rank: Process rank; the logger only records on rank 0.
        enabled: Master switch; ``False`` disables logging entirely.
        strict: If ``True``, let logging errors propagate instead of
            suppressing them best-effort.
        recording_id: Optional stable recording id (e.g. to group runs).

    Raises:
        ValueError: If more than one of ``save_path``, ``grpc_url``, or
            ``spawn`` is given.
    """

    def __init__(
        self,
        name: str = "vision3d",
        *,
        save_path: "str | Path | None" = None,
        grpc_url: str | None = None,
        spawn: bool = False,
        blueprint: "BlueprintLike | None" = None,
        prefix: str = "",
        rank: int = 0,
        enabled: bool = True,
        strict: bool = False,
        recording_id: str | None = None,
    ) -> None:
        self.enabled = enabled and rank == 0
        self._prefix = prefix.strip("/")
        self._strict = strict
        self._warned = False
        self._closed = False
        self._rec: RecordingStream | None = None
        if not self.enabled:
            return
        if sum((save_path is not None, grpc_url is not None, spawn)) > 1:
            msg = "pass at most one of save_path, grpc_url, spawn"
            raise ValueError(msg)
        # rr.init also registers this as the global recording (so the docs
        # scraper and bare rr.* helpers find it), but we capture the stream and
        # target it explicitly below -- so two loggers never cross-talk and
        # close() only finalizes this recording's sink.
        rr.init(name, recording_id=recording_id, spawn=spawn)
        self._rec = rr.get_global_data_recording()
        if save_path is not None:
            rr.save(str(save_path), recording=self._rec)
        elif grpc_url is not None:
            rr.connect_grpc(grpc_url, recording=self._rec)
        if blueprint is not None:
            rr.send_blueprint(blueprint, recording=self._rec)

    @property
    def recording(self) -> "RecordingStream | None":
        """This logger's recording stream, or ``None`` when disabled.

        Prefer the rank-aware scene methods (:meth:`log_sample`,
        :meth:`log_boxes_3d`, :meth:`log_point_cloud`, :meth:`log_cameras`)
        for 3D data: they route into this recording and become no-ops off
        rank 0, so they can be called unconditionally from shared loop code.

        This property is the escape hatch for raw ``rr.*`` calls that have no
        method here (e.g. ``rr.log`` of a custom archetype). It is ``None``
        when the logger is disabled, so guard such calls with ``if
        logger.recording is not None:`` to keep them off non-zero ranks --
        passing ``None`` to a bare ``rr.log`` would otherwise fall back to
        Rerun's global recording.
        """
        return self._rec

    def _entity_prefix(self, group: str) -> str:
        """Join the run namespace and ``group`` into an entity prefix.

        Returns:
            The ``/``-joined non-empty parts of ``(prefix, group)``.
        """
        return "/".join(part for part in (self._prefix, group) if part)

    def _run(self, action: str, fn: Callable[[], None]) -> None:
        """Run a logging action, suppressing failures unless ``strict``.

        Keeps a visualization failure from crashing training: the first
        suppressed error emits one warning; the rest are silenced.

        Args:
            action: Short name of the action, used in the warning message.
            fn: Zero-argument callable performing the Rerun calls.

        Raises:
            LoggingInputError: If ``fn`` reports malformed input. Re-raised
                even when not ``strict``, since it signals a caller bug.
        """
        try:
            fn()
        except LoggingInputError:
            # A caller bug (non-scalar metric, mismatched list lengths, ...),
            # not a transient sink failure. Always surface it so it gets fixed,
            # even in best-effort mode -- suppressing it would silently drop
            # data and leave the user debugging a phantom.
            raise
        except Exception:
            # Broad by design: visualization logging must never crash training.
            if self._strict:
                raise
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"RerunLogger: {action!r} failed and was suppressed; "
                    "further logging errors are silenced (pass strict=True to "
                    "raise).",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def log(
        self,
        values: Mapping[str, SupportsFloat | Tensor],
        *,
        step: int | None = None,
        epoch: int | None = None,
        group: str = "train",
        every: int | None = None,
        last: bool = False,
    ) -> None:
        """Log scalar metrics for this run.

        Delegates to :func:`log_scalars`, routing each metric to
        ``{prefix}/{group}/{name}`` so a dashboard
        (:func:`time_series_view`) groups it by run and section.

        Args:
            values: Mapping from metric name to scalar value (Python or numpy
                number, or single-element tensor). See :func:`log_scalars`.
            step: Optimizer/iteration step (``"step"`` timeline).
            epoch: Training epoch (``"epoch"`` timeline).
            group: Section under this run, e.g. ``"train"`` or ``"val"``.
            every: If set, only log when the driving counter (``step`` if
                given, else ``epoch``) is a multiple of it -- a cheap throttle
                for hot loops. Ignored when neither ``step`` nor ``epoch`` is
                given.
            last: Force this call through the ``every`` throttle. Pass
                ``last=(step == total_steps - 1)`` on the final iteration so a
                run's end-of-training metrics are never dropped just because
                the last step is not a multiple of ``every``.
        """
        if not self.enabled:
            return
        counter = step if step is not None else epoch
        if (
            every is not None
            and counter is not None
            and not last
            and counter % every != 0
        ):
            return
        self._run(
            "log",
            lambda: log_scalars(
                values,
                step=step,
                epoch=epoch,
                prefix=self._entity_prefix(group),
                recording=self._rec,
            ),
        )

    def log_config(self, config: Mapping[str, object]) -> None:
        """Attach the run's hyperparameters/config to the recording.

        Records ``config`` as a Rerun recording property so the run's settings
        (learning rate, batch size, augmentations, ...) travel with the
        recording and show in the viewer's properties panel -- the context you
        need to tell runs apart when comparing them later. Call once at
        startup.

        Nested mappings (e.g. a Hydra/OmegaConf config) are flattened into
        dot-separated keys, so ``{"optimizer": {"lr": 1e-3}}`` is recorded as
        ``optimizer.lr``. Numbers are recorded as numbers (so runs can be
        sorted or compared on, say, ``lr`` later); everything else falls back
        to its text representation.

        Args:
            config: Mapping of config name to value. Values may themselves be
                nested mappings.
        """
        if not self.enabled:
            return

        flat = _flatten_config(config)

        def _send() -> None:
            for key, value in flat.items():
                # Name the component after the property itself, not a shared
                # "value": Rerun enforces one type per component name across the
                # whole recording, so a single shared name would force every
                # config entry to the first one's type and silently drop the
                # rest (e.g. strings after a float). ``drop_untyped_nones`` is
                # passed explicitly so the dynamic ``**`` only has to satisfy
                # AnyValues' keyword-values parameter.
                values = rr.AnyValues(
                    drop_untyped_nones=True, **{key: _config_value(value)}
                )
                rr.send_property(key, values, recording=self._rec)

        self._run("log_config", _send)

    def style_series(
        self,
        name: str,
        *,
        group: str = "train",
        legend: str | None = None,
        color: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
        width: float | None = None,
    ) -> None:
        """Style one of this run's metric series.

        Resolves the entity from this logger's namespace so the run prefix
        and ``group`` need not be repeated. See :func:`style_series`.

        Args:
            name: Metric name to style (the same name passed to :meth:`log`).
            group: Section the metric was logged under.
            legend: Legend name for the series.
            color: RGB or RGBA color (0-255 per channel).
            width: Line width in points.
        """
        if not self.enabled:
            return
        entity = "/".join(part for part in (self._entity_prefix(group), name) if part)
        self._run(
            "style_series",
            lambda: style_series(
                entity, name=legend, color=color, width=width, recording=self._rec
            ),
        )

    def set_time(self, *, step: int | None = None, epoch: int | None = None) -> None:
        """Move this recording's timeline cursor.

        Call before the scene methods (:meth:`log_point_cloud`,
        :meth:`log_boxes_3d`, ...) when re-logging 3D data over training, so
        the geometry lands on the same ``step``/``epoch`` as the scalar curves
        and plays back in lockstep. No-op when disabled.

        A cursor set here persists: only the timelines you pass are moved, and
        each subsequent log carries *every* active cursor. After logging on the
        ``"epoch"`` timeline, switching to ``set_time(step=...)`` still stamps
        the stale ``"epoch"`` value too; call :meth:`reset_time` first to clear
        it when changing which timeline drives a sequence of logs.

        Args:
            step: Sequence value for the ``"step"`` timeline.
            epoch: Sequence value for the ``"epoch"`` timeline.
        """
        if not self.enabled:
            return

        def _set() -> None:
            if step is not None:
                rr.set_time("step", sequence=step, recording=self._rec)
            if epoch is not None:
                rr.set_time("epoch", sequence=epoch, recording=self._rec)

        self._run("set_time", _set)

    def reset_time(self) -> None:
        """Clear all timeline cursors on this recording.

        Subsequent logs carry no timeline until the next :meth:`set_time` (or
        ``step``/``epoch`` on :meth:`log`). Use this when switching which
        timeline drives a run of logs -- e.g. after logging per-epoch
        validation metrics, reset before re-logging predictions on the
        ``"step"`` timeline so they are not stamped with a stale ``epoch``
        cursor. No-op when disabled.
        """
        if not self.enabled:
            return
        self._run("reset_time", lambda: rr.reset_time(recording=self._rec))

    def log_point_cloud(
        self,
        entity: str,
        points: PointCloud3D | Tensor,
        *,
        color_by_distance: bool = True,
        static: bool = False,
    ) -> None:
        """Rank-aware, best-effort wrapper around :func:`log_point_cloud`.

        Routes into this logger's recording and is a no-op when disabled.
        ``entity`` is an absolute Rerun path (not namespaced by ``prefix``),
        since 3D scene data is typically shared across a recording.

        Args:
            entity: Rerun entity path (e.g. ``"world/lidar"``).
            points: Point cloud ``[N, 3+C]``. See :func:`log_point_cloud`.
            color_by_distance: Color points by distance from origin.
            static: Log without a timeline (constant geometry).
        """
        if not self.enabled:
            return
        self._run(
            "log_point_cloud",
            lambda: log_point_cloud(
                entity,
                points,
                color_by_distance=color_by_distance,
                static=static,
                recording=self._rec,
            ),
        )

    def log_boxes_3d(
        self,
        entity: str,
        boxes: BoundingBoxes3D,
        *,
        labels: list[str] | None = None,
        class_ids: list[int] | None = None,
        label_to_id: dict[str, int] | None = None,
        scores: list[float] | Tensor | None = None,
        score_threshold: float | None = None,
        fill_mode: "rr.components.FillModeLike | None" = None,
        show_labels: bool | None = None,
        log_heading: bool = True,
        static: bool = False,
    ) -> None:
        """Rank-aware, best-effort wrapper around :func:`log_boxes_3d`.

        Routes into this logger's recording and is a no-op when disabled.
        ``entity`` is an absolute Rerun path (not namespaced by ``prefix``).
        See :func:`log_boxes_3d` for the argument semantics.

        Args:
            entity: Rerun entity path (e.g. ``"world/pred/boxes"``).
            boxes: Bounding boxes in any supported format.
            labels: Per-box label strings for display.
            class_ids: Per-box class IDs for coloring.
            label_to_id: Mapping from class name to class ID.
            scores: Per-box confidence scores.
            score_threshold: Drop boxes scoring below this. Requires ``scores``.
            fill_mode: Box fill mode (e.g. ``"majorwireframe"``).
            show_labels: Force per-box labels on/off.
            log_heading: Log heading arrows for rotated boxes.
            static: Log without a timeline (constant geometry, e.g. ground
                truth on a fixed sample inspected over training).
        """
        if not self.enabled:
            return
        self._run(
            "log_boxes_3d",
            lambda: log_boxes_3d(
                entity,
                boxes,
                labels=labels,
                class_ids=class_ids,
                label_to_id=label_to_id,
                scores=scores,
                score_threshold=score_threshold,
                fill_mode=fill_mode,
                show_labels=show_labels,
                log_heading=log_heading,
                static=static,
                recording=self._rec,
            ),
        )

    def log_cameras(
        self,
        entity_prefix: str,
        images: CameraImages | Tensor,
        intrinsics: CameraIntrinsics | Tensor | None = None,
        extrinsics: CameraExtrinsics | Tensor | None = None,
        *,
        jpeg_quality: int | None = None,
    ) -> None:
        """Rank-aware, best-effort wrapper around :func:`log_cameras`.

        Routes into this logger's recording and is a no-op when disabled.
        ``entity_prefix`` is an absolute Rerun path (not namespaced by
        ``prefix``). See :func:`log_cameras`.

        Args:
            entity_prefix: Rerun entity path prefix (e.g. ``"world/cam"``).
            images: Camera images ``[N_cams, C, H, W]``.
            intrinsics: Intrinsic matrices ``[N_cams, 3, 3]``.
            extrinsics: Extrinsic matrices ``[N_cams, 4, 4]``.
            jpeg_quality: If set, JPEG-encode each image at this quality.
        """
        if not self.enabled:
            return
        self._run(
            "log_cameras",
            lambda: log_cameras(
                entity_prefix,
                images,
                intrinsics,
                extrinsics,
                jpeg_quality=jpeg_quality,
                recording=self._rec,
            ),
        )

    def log_sample(
        self,
        inputs: SampleInputs,
        targets: SampleTargets | None = None,
        *,
        predictions: Prediction3D | None = None,
        entity_prefix: str = "world",
        label_to_id: dict[str, int] | None = None,
        score_threshold: float | None = None,
        jpeg_quality: int | None = None,
    ) -> None:
        """Rank-aware, best-effort wrapper around :func:`log_sample`.

        Routes into this logger's recording and is a no-op when disabled.
        ``entity_prefix`` is an absolute Rerun path (not namespaced by
        ``prefix``). See :func:`log_sample`.

        Args:
            inputs: Sample inputs (points, images, intrinsics, extrinsics).
            targets: Ground-truth targets.
            predictions: Model predictions.
            entity_prefix: Root entity path for the sample (e.g. ``"world"``).
            label_to_id: Mapping from class name to class ID.
            score_threshold: Drop predictions scoring below this.
            jpeg_quality: If set, JPEG-encode camera images at this quality.
        """
        if not self.enabled:
            return
        self._run(
            "log_sample",
            lambda: log_sample(
                inputs,
                targets,
                predictions=predictions,
                entity_prefix=entity_prefix,
                label_to_id=label_to_id,
                score_threshold=score_threshold,
                jpeg_quality=jpeg_quality,
                recording=self._rec,
            ),
        )

    def flush(self) -> None:
        """Flush buffered data to the sink so partial results are visible."""
        if not self.enabled or self._rec is None:
            return
        self._run("flush", self._rec.flush)

    def close(self) -> None:
        """Flush and disconnect this recording's sink (idempotent)."""
        if not self.enabled or self._closed or self._rec is None:
            return
        self._closed = True
        self.flush()
        self._run("close", self._rec.disconnect)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _scalar_value(name: str, value: SupportsFloat | Tensor) -> float:
    """Coerce a scalar metric value to a Python float.

    Returns:
        The value as a float.

    Raises:
        LoggingInputError: If ``value`` is a tensor that does not hold exactly
            one element.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1:
            msg = f"metric {name!r} must be a scalar but has {value.numel()} elements"
            raise LoggingInputError(msg)
        return value.detach().cpu().item()
    return float(value)


def _flatten_config(
    config: Mapping[str, object], _parent: str = ""
) -> dict[str, object]:
    """Flatten a (possibly nested) config mapping into dot-separated keys.

    ``{"optimizer": {"lr": 1e-3}}`` becomes ``{"optimizer.lr": 1e-3}`` so each
    leaf can be recorded as its own Rerun property.

    Returns:
        A flat mapping from dotted key to leaf value.
    """
    flat: dict[str, object] = {}
    for key, value in config.items():
        full = f"{_parent}.{key}" if _parent else key
        if isinstance(value, Mapping):
            flat.update(_flatten_config(value, full))
        else:
            flat[full] = value
    return flat


def _config_value(value: object) -> int | float | str:
    """Coerce an arbitrary config value to a Rerun-recordable scalar.

    ``int``/``float``/``str`` are the scalar types Rerun records natively as a
    recording property. Numbers pass through as numbers so runs can be compared
    on them later; booleans and everything else fall back to a readable text
    representation.

    Returns:
        ``value`` as an ``int``/``float`` when numeric, else its ``str``.
    """
    if isinstance(value, bool):
        # bool is an int subclass; keep the readable "True"/"False" text.
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Tensor) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return str(value)


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
        LoggingInputError: If ``fmt`` is not a supported format.
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
        raise LoggingInputError(msg)

    return centers, sizes, yaws
