"""High-level Rerun logger orchestrating vision3d logging for training runs."""

import contextlib
import numbers
import warnings
import weakref
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Self, SupportsFloat

from torch import Tensor

from vision3d.datasets import SampleInputs, SampleTargets
from vision3d.metrics import Prediction3D
from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

from ._errors import LoggingInputError
from ._rerun import rr
from ._scalars import log_scalars, style_series
from ._scene import log_boxes_3d, log_cameras, log_point_cloud, log_sample

if TYPE_CHECKING:
    from pathlib import Path

    from rerun import RecordingStream
    from rerun.blueprint import BlueprintLike


def _finalize_recording(rec: "RecordingStream") -> None:
    """Flush and disconnect a recording, swallowing any failure.

    Backs :class:`RerunLogger`'s :func:`weakref.finalize` fallback. Kept at
    module level so the finalizer holds no reference to the logger.

    Args:
        rec: The recording stream to finalize.
    """
    # Runs at GC/interpreter shutdown, where warnings may be torn down: swallow.
    with contextlib.suppress(Exception):
        rec.flush()
        rec.disconnect()


class RerunLogger:
    """High-level Rerun logger for training metrics.

    Wraps the Rerun recording lifecycle (sink selection, an optional dashboard
    blueprint, rank-aware disabling, flushing) around :func:`log_scalars` and
    :func:`style_series` so a training loop can log metrics in one line.

    Pass at most one sink: ``save_path`` (write an ``.rrd`` file), ``grpc_url``
    (stream to a running viewer), or ``spawn`` (launch a local viewer); with
    none the recording is buffered in memory. Pass ``rank`` for distributed
    training -- the logger is a no-op off rank 0, so it can be called from
    shared loop code. Logging is best-effort: operational failures are caught
    and warned once per action, while malformed input raises
    :class:`LoggingInputError`; pass ``strict=True`` to propagate everything.
    Use as a context manager or call :meth:`close` for a guaranteed flush. Each
    logger owns a private recording stream and never becomes Rerun's global
    default, so several can coexist in one process without cross-talk.

    Args:
        name: Rerun application id for the recording.
        save_path: Path to write an ``.rrd`` file to.
        grpc_url: URL of a Rerun gRPC sink to stream to.
        spawn: Launch a local Rerun viewer and stream to it.
        blueprint: Optional dashboard layout (see :func:`time_series_view`,
            :func:`lidar_view`). ``None`` lets Rerun auto-arrange views.
        namespace: Entity-path namespace prepended to every metric (e.g.
            ``"runs/baseline"``), combined with each call's ``group``.
        rank: Process rank; the logger only records on rank 0.
        enabled: Master switch; ``False`` disables logging entirely.
        strict: If ``True``, let logging errors propagate instead of
            suppressing them best-effort.
        recording_id: Optional stable recording id (e.g. to group runs).

    Raises:
        LoggingInputError: If more than one of ``save_path``, ``grpc_url``, or
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
        namespace: str = "",
        rank: int = 0,
        enabled: bool = True,
        strict: bool = False,
        recording_id: str | None = None,
    ) -> None:
        # Validate before the rank guard so every rank rejects a bad config.
        if sum((save_path is not None, grpc_url is not None, spawn)) > 1:
            msg = "pass at most one of save_path, grpc_url, spawn"
            raise LoggingInputError(msg)
        self.enabled = enabled and rank == 0
        self._namespace = namespace.strip("/")
        self._strict = strict
        self._warned: set[str] = set()
        self._closed = False
        self._rec: RecordingStream | None = None
        self._finalizer: weakref.finalize[..., object] | None = None
        if not self.enabled:
            return
        # Own a private stream rather than rr.init, which would register it as
        # Rerun's global default; make_default stays off so loggers never
        # cross-talk. Every call below targets this stream explicitly.
        self._rec = rr.RecordingStream(name, recording_id=recording_id)
        if spawn:
            self._rec.spawn()
        elif save_path is not None:
            self._rec.save(str(save_path))
        elif grpc_url is not None:
            self._rec.connect_grpc(grpc_url)
        if blueprint is not None:
            self._rec.send_blueprint(blueprint)
        # Fallback flush + disconnect if the caller never closes; close()
        # detaches it so the sink is finalized exactly once.
        self._finalizer = weakref.finalize(self, _finalize_recording, self._rec)

    @property
    def recording(self) -> "RecordingStream | None":
        """This logger's recording stream, or ``None`` when disabled.

        Escape hatch for raw ``rr.*`` calls with no method here. Guard them
        with ``if logger.recording is not None:`` so they stay off non-zero
        ranks; prefer the rank-aware scene methods for 3D data.
        """
        return self._rec

    def _entity_prefix(self, group: str) -> str:
        """Join the run namespace and ``group`` into an entity prefix.

        Returns:
            The ``/``-joined non-empty parts of ``(namespace, group)``.
        """
        return "/".join(part for part in (self._namespace, group.strip("/")) if part)

    def _run(self, action: str, fn: Callable[[], None]) -> None:
        """Run a logging action, suppressing failures unless ``strict``.

        The first suppressed failure of each ``action`` warns once; repeats are
        silenced, so a flood stays quiet while a later, different failure still
        surfaces.

        Args:
            action: Short name of the action, used in the warning and as the
                rate-limit key.
            fn: Zero-argument callable performing the Rerun calls.

        Raises:
            LoggingInputError: If ``fn`` reports malformed input. Re-raised
                even when not ``strict``, since it signals a caller bug.
        """
        try:
            fn()
        except LoggingInputError:
            # Caller bug, not a transient sink failure: always surface it.
            raise
        except Exception:
            # Broad by design: visualization logging must never crash training.
            if self._strict:
                raise
            if action not in self._warned:
                self._warned.add(action)
                warnings.warn(
                    f"RerunLogger: {action!r} failed and was suppressed; "
                    f"further {action!r} errors are silenced (pass strict=True "
                    "to raise).",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def _scene(
        self, action: str, fn: Callable[..., None], *args: object, **kwargs: object
    ) -> None:
        """Dispatch a scene logger onto this recording, rank-aware and safe.

        Shared body of the scene wrappers: skip off rank 0, target this
        logger's recording, and route through :meth:`_run`.

        Args:
            action: Rate-limit key and warning label for :meth:`_run`.
            fn: Free scene-logging function to call (e.g.
                :func:`log_point_cloud`).
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``; ``recording`` is
                injected automatically.
        """
        if not self.enabled:
            return
        self._run(action, lambda: fn(*args, recording=self._rec, **kwargs))

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
        ``{namespace}/{group}/{name}``.

        Args:
            values: Mapping from metric name to scalar value (Python or numpy
                number, or single-element tensor). See :func:`log_scalars`.
            step: Optimizer/iteration step (``"step"`` timeline).
            epoch: Training epoch (``"epoch"`` timeline).
            group: Section under this run, e.g. ``"train"`` or ``"val"``.
            every: If set, only log when the driving counter (``step`` if
                given, else ``epoch``) is a multiple of it. Ignored when
                neither ``step`` nor ``epoch`` is given.
            last: Force this call through the ``every`` throttle, e.g. on the
                final iteration so end-of-training metrics are not dropped.

        Raises:
            LoggingInputError: If ``every`` is set to less than 1.
        """
        if not self.enabled:
            return
        if every is not None and every < 1:
            # every=0 would divide by zero in the modulo below; negatives never
            # match. A bad throttle is a caller bug, not a reason to crash.
            msg = f"every must be >= 1, got {every}"
            raise LoggingInputError(msg)
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

        Records ``config`` as Rerun recording properties so the run's settings
        travel with the recording and show in the viewer. Call once at startup.
        Nested mappings are flattened to dot-separated keys
        (``{"optimizer": {"lr": 1e-3}}`` -> ``optimizer.lr``); numbers stay
        numeric, everything else falls back to its text representation.

        Args:
            config: Mapping of config name to value. Values may themselves be
                nested mappings.
        """
        if not self.enabled:
            return

        flat = _flatten_config(config)

        def _send() -> None:
            for key, value in flat.items():
                # Name each component after its own key: Rerun enforces one type
                # per component name, so a shared name would coerce every entry
                # to the first one's type and drop the rest.
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

        Resolves the entity from this logger's namespace so ``namespace`` and
        ``group`` need not be repeated. See :func:`style_series`.

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

        Call before the scene methods when re-logging 3D data over training so
        the geometry lands on the same ``step``/``epoch`` as the scalar curves.
        Cursors persist: only the timelines you pass are moved, and each log
        carries every active cursor, so call :meth:`reset_time` first when
        changing which timeline drives a sequence of logs. No-op when disabled.

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
        ``step``/``epoch`` on :meth:`log`). Use when switching which timeline
        drives a run of logs, to avoid stamping a stale cursor. No-op when
        disabled.
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
        ``entity`` is an absolute Rerun path (not namespaced by ``namespace``).

        Args:
            entity: Rerun entity path (e.g. ``"world/lidar"``).
            points: Point cloud ``[N, 3+C]``. See :func:`log_point_cloud`.
            color_by_distance: Color points by distance from origin.
            static: Log without a timeline (constant geometry).
        """
        self._scene(
            "log_point_cloud",
            log_point_cloud,
            entity,
            points,
            color_by_distance=color_by_distance,
            static=static,
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
        fill_mode: rr.components.FillModeLike | None = None,
        show_labels: bool | None = None,
        log_heading: bool = True,
        static: bool = False,
    ) -> None:
        """Rank-aware, best-effort wrapper around :func:`log_boxes_3d`.

        Routes into this logger's recording and is a no-op when disabled.
        ``entity`` is an absolute Rerun path (not namespaced by ``namespace``).
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
        self._scene(
            "log_boxes_3d",
            log_boxes_3d,
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
        ``namespace``). See :func:`log_cameras`.

        Args:
            entity_prefix: Rerun entity path prefix (e.g. ``"world/cam"``).
            images: Camera images ``[N_cams, C, H, W]``.
            intrinsics: Intrinsic matrices ``[N_cams, 3, 3]``.
            extrinsics: Extrinsic matrices ``[N_cams, 4, 4]``.
            jpeg_quality: If set, JPEG-encode each image at this quality.
        """
        self._scene(
            "log_cameras",
            log_cameras,
            entity_prefix,
            images,
            intrinsics,
            extrinsics,
            jpeg_quality=jpeg_quality,
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
        ``namespace``). See :func:`log_sample`.

        Args:
            inputs: Sample inputs (points, images, intrinsics, extrinsics).
            targets: Ground-truth targets.
            predictions: Model predictions.
            entity_prefix: Root entity path for the sample (e.g. ``"world"``).
            label_to_id: Mapping from class name to class ID.
            score_threshold: Drop predictions scoring below this.
            jpeg_quality: If set, JPEG-encode camera images at this quality.
        """
        self._scene(
            "log_sample",
            log_sample,
            inputs,
            targets,
            predictions=predictions,
            entity_prefix=entity_prefix,
            label_to_id=label_to_id,
            score_threshold=score_threshold,
            jpeg_quality=jpeg_quality,
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
        # Flip state only after the sink work succeeds, so a failing flush in
        # strict mode propagates with the finalizer still alive to retry.
        self.flush()
        self._run("close", self._rec.disconnect)
        self._closed = True
        if self._finalizer is not None:
            self._finalizer.detach()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _flatten_config(
    config: Mapping[str, object], _parent: str = ""
) -> dict[str, object]:
    """Flatten a (possibly nested) config mapping into dot-separated keys.

    ``{"optimizer": {"lr": 1e-3}}`` becomes ``{"optimizer.lr": 1e-3}``.

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

    Numbers pass through as numbers so runs can be compared on them later;
    booleans and everything else fall back to a readable text representation.

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
