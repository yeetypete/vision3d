"""Tests for :func:`vision3d.viz.log_scalars` and :func:`style_series`.

These exercise entity routing, timeline handling, and scalar coercion
without a live Rerun recording: ``rr.log``, ``rr.Scalars``, and
``rr.set_time`` are spied on so the arguments handed to Rerun can be
asserted directly.
"""

import numpy as np
import pytest
import torch

import vision3d.viz._scalars as scalars_mod
from vision3d.viz._errors import LoggingInputError
from vision3d.viz._scalars import _scalar_value, log_scalars, style_series


class _Spy:
    """Capture ``rr.set_time``, ``rr.Scalars``, and ``rr.log`` calls."""

    def __init__(self) -> None:
        self.times: list[tuple[str, int | None]] = []
        self.logged: list[tuple[str, float]] = []

    def set_time(
        self, timeline: str, *, sequence: int | None = None, **_: object
    ) -> None:
        self.times.append((timeline, sequence))

    def scalars(self, value: float) -> float:
        # Pass the raw value straight through so ``log`` records it.
        return value

    def log(self, entity: str, archetype: float, **_: object) -> None:
        self.logged.append((entity, archetype))


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Patch Rerun's ``set_time``/``Scalars``/``log`` to capture calls.

    Returns:
        The :class:`_Spy` recording calls into Rerun.
    """
    s = _Spy()
    monkeypatch.setattr(scalars_mod.rr, "set_time", s.set_time)
    monkeypatch.setattr(scalars_mod.rr, "Scalars", s.scalars)
    monkeypatch.setattr(scalars_mod.rr, "log", s.log)
    return s


class TestEntityRouting:
    def test_prefix_prepended_to_each_name(self, spy: _Spy) -> None:
        log_scalars({"loss": 1.5, "lr": 0.01}, step=0)
        assert spy.logged == [("train/loss", 1.5), ("train/lr", 0.01)]

    def test_empty_prefix_logs_at_root(self, spy: _Spy) -> None:
        log_scalars({"loss": 1.5}, step=0, prefix="")
        assert spy.logged == [("loss", 1.5)]

    def test_nested_names_kept(self, spy: _Spy) -> None:
        log_scalars({"loss/cls": 0.3}, step=0, prefix="val")
        assert spy.logged == [("val/loss/cls", 0.3)]


class TestTimelines:
    def test_step_sets_step_timeline(self, spy: _Spy) -> None:
        log_scalars({"loss": 1.0}, step=7)
        assert spy.times == [("step", 7)]

    def test_epoch_sets_epoch_timeline(self, spy: _Spy) -> None:
        log_scalars({"loss": 1.0}, epoch=3)
        assert spy.times == [("epoch", 3)]

    def test_step_and_epoch_set_both(self, spy: _Spy) -> None:
        log_scalars({"loss": 1.0}, step=7, epoch=3)
        assert spy.times == [("step", 7), ("epoch", 3)]

    def test_no_timeline_when_neither_given(self, spy: _Spy) -> None:
        log_scalars({"loss": 1.0})
        assert spy.times == []
        assert spy.logged == [("train/loss", 1.0)]


class TestScalarCoercion:
    def test_python_number_passed_through(self) -> None:
        assert _scalar_value("loss", 2) == 2.0

    def test_single_element_tensor_extracted(self, spy: _Spy) -> None:
        log_scalars({"loss": torch.tensor(0.5)}, step=0)
        assert spy.logged == [("train/loss", pytest.approx(0.5))]

    def test_multi_element_tensor_raises(self) -> None:
        # A non-scalar metric is a caller bug: raise the dedicated input error
        # (a ValueError subclass) so RerunLogger can re-raise it best-effort.
        with pytest.raises(LoggingInputError, match="must be a scalar but has 2"):
            _scalar_value("loss", torch.tensor([1.0, 2.0]))

    def test_numpy_scalar_coerced(self, spy: _Spy) -> None:
        # vision3d.metrics returns numpy/python scalars; they must log cleanly.
        log_scalars({"mAP": np.float32(0.5)}, step=0, prefix="val")
        assert spy.logged == [("val/mAP", pytest.approx(0.5))]


class TestStyleSeries:
    def test_logs_series_lines_statically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, object, dict[str, object]]] = []
        captured: dict[str, object] = {}

        def fake_log(entity: str, archetype: object, **kwargs: object) -> None:
            calls.append((entity, archetype, kwargs))

        def fake_series_lines(**kwargs: object) -> str:
            captured.update(kwargs)
            return "series"

        monkeypatch.setattr(scalars_mod.rr, "log", fake_log)
        monkeypatch.setattr(scalars_mod.rr, "SeriesLines", fake_series_lines)

        style_series(
            "runs/baseline/loss", name="baseline", color=(255, 0, 0), width=2.0
        )

        assert captured == {"names": "baseline", "colors": (255, 0, 0), "widths": 2.0}
        assert len(calls) == 1
        entity, _, kwargs = calls[0]
        assert entity == "runs/baseline/loss"
        assert kwargs.get("static") is True

    def test_defaults_pass_through_as_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(scalars_mod.rr, "log", lambda *a, **k: None)
        monkeypatch.setattr(
            scalars_mod.rr, "SeriesLines", lambda **k: captured.update(k)
        )

        style_series("runs/baseline/loss")

        assert captured == {"names": None, "colors": None, "widths": None}
