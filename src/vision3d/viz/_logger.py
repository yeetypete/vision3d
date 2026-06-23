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

    Backs :class:`RerunLogger`'s :func:`weakref.finalize` fallback, so a sink
    is still flushed when the caller forgets :meth:`RerunLogger.close`. Kept at
    module level (not a bound method) so the finalizer holds no reference to the
    logger. Errors are swallowed rather than warned: this runs at GC or
    interpreter shutdown, where the warnings machinery may be torn down and
    there is no loop left to protect.

    Args:
        rec: The recording stream to finalize.
    """
    with contextlib.suppress(Exception):
        rec.flush()
        rec.disconnect()


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
    suppressed with a warning, so a visualization hiccup never crashes a long
    training run. The warning fires once per failing operation (e.g. ``log``,
    ``log_config``) and is then silenced for that operation, so a flood of the
    same failure stays quiet while a later, different failure still surfaces.
    Pass ``strict=True`` to let such errors propagate
    instead. *Input* errors are treated differently: malformed arguments (a
    non-scalar metric, mismatched per-box list lengths, ...) raise
    :class:`LoggingInputError` and always propagate, even in best-effort mode,
    since they signal a caller bug that suppression would only hide.

    Record the run's hyperparameters once with :meth:`log_config` so they
    travel with the recording -- essential context when comparing runs later.

    Construct the logger and call :meth:`log` -- no other bookkeeping is
    required. A fallback finalizer flushes and closes the sink at garbage
    collection or interpreter exit, so the simple form below just works. For a
    guaranteed flush even when the loop raises (and so a buffered ``.rrd`` is
    never left truncated), use it as a context manager or call :meth:`close`
    when done.

    Limitations:

    * **One active logger per process.** Constructing a logger calls
      ``rerun.init``, which also registers it as Rerun's global recording, so a
      second logger supersedes the first for bare ``rr.*`` calls and
      :attr:`recording` access. The logger's own methods always target their
      own stream, so this only matters if you reach for the global recording
      directly. For run comparison, log to one recording under different
      ``namespace`` values rather than creating several loggers.
    * **Not thread-safe.** Methods assume a single calling thread per rank (the
      usual training-loop case); do not share one logger across threads (e.g. a
      multi-worker data pipeline) without external synchronization.

    Example:
        >>> logger = RerunLogger(  # doctest: +SKIP
        ...     "bevfusion", save_path="run.rrd", rank=rank
        ... )
        >>> logger.log_config({"lr": 1e-3, "batch_size": 4})  # doctest: +SKIP
        >>> for step, batch in enumerate(loader):  # doctest: +SKIP
        ...     loss = train_step(batch)
        ...     logger.log({"loss/total": loss, "lr": lr}, step=step, every=50)

        For a guaranteed flush even if the loop raises, use the context manager:

        >>> with RerunLogger(
        ...     "bevfusion", save_path="run.rrd"
        ... ) as logger:  # doctest: +SKIP
        ...     for step, batch in enumerate(loader):
        ...         logger.log({"loss/total": train_step(batch)}, step=step)

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
        namespace: Entity-path namespace prepended to every metric (e.g.
            ``"runs/baseline"`` to keep one run's curves together for
            cross-run comparison). Set once for the whole run, then combined
            with each call's ``group`` to form the entity prefix passed to
            :func:`log_scalars` -- distinct from that function's ``prefix``,
            which is the already-composed full prefix. Empty by default.
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
        # Validate config before the rank guard so every rank rejects a bad
        # configuration identically -- it is a pure argument check and must not
        # depend on which rank happens to run it.
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
        # Fallback finalizer: flush + disconnect the sink if the caller forgets
        # close() (and never uses the context manager). Without it, a buffered
        # save_path sink can be left truncated on exit. Runs at GC or interpreter
        # shutdown; close() detaches it so the sink is finalized exactly once.
        # Module-level target (not a bound method) so it holds no reference to
        # self and cannot keep the logger alive.
        self._finalizer = weakref.finalize(self, _finalize_recording, self._rec)

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
            The ``/``-joined non-empty parts of ``(namespace, group)``.
        """
        return "/".join(part for part in (self._namespace, group) if part)

    def _run(self, action: str, fn: Callable[[], None]) -> None:
        """Run a logging action, suppressing failures unless ``strict``.

        Keeps a visualization failure from crashing training: the first
        suppressed failure of each ``action`` emits one warning; repeats of
        that action are then silenced. Warning per action rather than once for
        the whole logger means a later, unrelated failure mode (e.g. a blueprint
        send after scalar logging has been failing) still surfaces once.

        Args:
            action: Short name of the action, used in the warning message and
                as the rate-limit key.
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

        The shared body of the scene wrappers (:meth:`log_point_cloud`,
        :meth:`log_boxes_3d`, :meth:`log_cameras`, :meth:`log_sample`): skip
        off rank 0, target this logger's recording rather than Rerun's global,
        and route through :meth:`_run` for best-effort error handling. Keeping
        it in one place means the four wrappers cannot drift on that policy.

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
        ``{namespace}/{group}/{name}`` so a dashboard
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

        Raises:
            LoggingInputError: If ``every`` is set to less than 1.
        """
        if not self.enabled:
            return
        if every is not None and every < 1:
            # Guard the modulo below: every=0 would be a ZeroDivisionError and
            # negatives never match. A bad throttle is a caller bug, so surface
            # it as LoggingInputError rather than crashing the training loop.
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

        Resolves the entity from this logger's namespace so the run
        ``namespace`` and ``group`` need not be repeated. See
        :func:`style_series`.

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
        ``entity`` is an absolute Rerun path (not namespaced by ``namespace``),
        since 3D scene data is typically shared across a recording.

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
        self._closed = True
        if self._finalizer is not None:
            # We are finalizing explicitly; cancel the GC/exit fallback so the
            # sink is not disconnected a second time.
            self._finalizer.detach()
        self.flush()
        self._run("close", self._rec.disconnect)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


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
