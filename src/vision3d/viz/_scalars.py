"""Log scalar training metrics as Rerun time series."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, SupportsFloat

from torch import Tensor

from ._errors import LoggingInputError
from ._rerun import rr

if TYPE_CHECKING:
    from rerun import RecordingStream


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
