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
    to its own entity (``{prefix}/{name}``), forming a time series for a
    :class:`~rerun.blueprint.TimeSeriesView`. Metric names may contain ``/`` to
    nest, e.g. ``{"loss/total": ..., "loss/cls": ...}`` groups under
    ``{prefix}/loss``. ``step``/``epoch`` move the recording's timeline cursor,
    which persists, so pass them on every call (or use :class:`RerunLogger`).

    Args:
        values: Mapping from metric name to scalar value (Python or numpy
            number, or single-element tensor). Tensor values must hold a
            single element and are moved to the CPU before logging.
        step: Optimizer/iteration step, logged on a ``"step"`` timeline.
        epoch: Training epoch, logged on an ``"epoch"`` timeline. May be
            combined with ``step`` to log on both timelines at once.
        prefix: Entity path prefix grouping these metrics (e.g. ``"train"``,
            ``"val"``). Pass ``""`` to log each metric at the root.
        recording: Target recording. ``None`` (default) uses Rerun's active
            global recording.
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
    control how the curve produced by :func:`log_scalars` appears. Most useful
    for overlaying multiple training runs in one plot: route each run to its
    own entity prefix and give it a stable legend name and distinct color.

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
