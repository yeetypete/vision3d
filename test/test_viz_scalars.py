"""Tests for :func:`vision3d.viz.log_scalars`.

These exercise entity routing, timeline handling, and scalar coercion
without a live Rerun recording: ``rr.log``, ``rr.Scalars``, and
``rr.set_time`` are spied on so the arguments handed to Rerun can be
asserted directly.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import vision3d.viz._logging as logging_mod
from vision3d.viz import time_series_view
from vision3d.viz._logging import (
    MetricLogger,
    _scalar_value,
    log_scalars,
    style_series,
)


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
    monkeypatch.setattr(logging_mod.rr, "set_time", s.set_time)
    monkeypatch.setattr(logging_mod.rr, "Scalars", s.scalars)
    monkeypatch.setattr(logging_mod.rr, "log", s.log)
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
        with pytest.raises(ValueError, match="must be a scalar but has 2 elements"):
            _scalar_value("loss", torch.tensor([1.0, 2.0]))

    def test_numpy_scalar_coerced(self, spy: _Spy) -> None:
        # vision3d.metrics returns numpy/python scalars; they must log cleanly.
        log_scalars({"mAP": np.float64(0.5)}, step=0, prefix="val")
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

        monkeypatch.setattr(logging_mod.rr, "log", fake_log)
        monkeypatch.setattr(logging_mod.rr, "SeriesLines", fake_series_lines)

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
        monkeypatch.setattr(logging_mod.rr, "log", lambda *a, **k: None)
        monkeypatch.setattr(
            logging_mod.rr, "SeriesLines", lambda **k: captured.update(k)
        )

        style_series("runs/baseline/loss")

        assert captured == {"names": None, "colors": None, "widths": None}


class _RrSpy:
    """Spy standing in for the Rerun module inside :class:`MetricLogger`."""

    def __init__(self) -> None:
        self.inits: list[tuple[str, str | None, bool]] = []
        self.saves: list[str] = []
        self.grpc: list[str] = []
        self.blueprints: list[object] = []
        self.disconnects = 0
        self.flushes = 0
        self.logged: list[tuple[str, object]] = []
        self.times: list[tuple[str, int | None]] = []

    def init(
        self, name: str, *, recording_id: str | None = None, spawn: bool = False
    ) -> None:
        self.inits.append((name, recording_id, spawn))

    def save(self, path: str) -> None:
        self.saves.append(path)

    def connect_grpc(self, url: str) -> None:
        self.grpc.append(url)

    def send_blueprint(self, blueprint: object) -> None:
        self.blueprints.append(blueprint)

    def disconnect(self) -> None:
        self.disconnects += 1

    def get_global_data_recording(self) -> object:
        spy = self

        class _Rec:
            def flush(self) -> None:
                spy.flushes += 1

        return _Rec()

    def set_time(self, timeline: str, *, sequence: int | None = None) -> None:
        self.times.append((timeline, sequence))

    def Scalars(self, value: float) -> float:
        return value

    def SeriesLines(self, **kwargs: object) -> tuple[str, dict[str, object]]:
        return ("series", kwargs)

    def log(self, entity: str, archetype: object, **_: object) -> None:
        self.logged.append((entity, archetype))


@pytest.fixture
def rr_spy(monkeypatch: pytest.MonkeyPatch) -> _RrSpy:
    """Replace every Rerun call :class:`MetricLogger` makes with a spy.

    Returns:
        The :class:`_RrSpy` recording the lifecycle and logging calls.
    """
    spy = _RrSpy()
    for attr in (
        "init",
        "save",
        "connect_grpc",
        "send_blueprint",
        "disconnect",
        "get_global_data_recording",
        "set_time",
        "Scalars",
        "SeriesLines",
        "log",
    ):
        monkeypatch.setattr(logging_mod.rr, attr, getattr(spy, attr))
    return spy


class TestMetricLoggerLifecycle:
    def test_disabled_on_nonzero_rank(self, rr_spy: _RrSpy) -> None:
        logger = MetricLogger("run", save_path="x.rrd", rank=1)
        logger.log({"loss": 1.0}, step=0)
        logger.style_series("loss")
        logger.close()
        assert not logger.enabled
        assert rr_spy.inits == []  # never touched Rerun on a non-zero rank
        assert rr_spy.logged == []

    def test_disabled_when_enabled_false(self, rr_spy: _RrSpy) -> None:
        logger = MetricLogger("run", save_path="x.rrd", enabled=False)
        assert not logger.enabled
        assert rr_spy.inits == []

    def test_save_sink(self, rr_spy: _RrSpy) -> None:
        MetricLogger("run", save_path="out.rrd")
        assert rr_spy.inits == [("run", None, False)]
        assert rr_spy.saves == ["out.rrd"]

    def test_spawn_sink(self, rr_spy: _RrSpy) -> None:
        MetricLogger("run", spawn=True)
        assert rr_spy.inits == [("run", None, True)]
        assert rr_spy.saves == []

    def test_grpc_sink(self, rr_spy: _RrSpy) -> None:
        MetricLogger("run", grpc_url="rerun+http://host:9876/proxy")
        assert rr_spy.grpc == ["rerun+http://host:9876/proxy"]

    def test_blueprint_sent(self, rr_spy: _RrSpy) -> None:
        MetricLogger("run", blueprint=time_series_view(entity_prefix="train"))
        assert len(rr_spy.blueprints) == 1

    def test_multiple_sinks_raise(self) -> None:
        with pytest.raises(ValueError, match="at most one of"):
            MetricLogger("run", save_path="x.rrd", spawn=True)

    def test_context_manager_flushes_and_disconnects(self, rr_spy: _RrSpy) -> None:
        with MetricLogger("run", save_path="x.rrd"):
            pass
        assert rr_spy.flushes == 1
        assert rr_spy.disconnects == 1


class TestMetricLoggerLogging:
    def test_namespace_and_group_compose_entity(self, rr_spy: _RrSpy) -> None:
        logger = MetricLogger("run", spawn=True, prefix="runs/baseline")
        logger.log({"loss/total": 1.0}, step=3)
        assert rr_spy.times == [("step", 3)]
        assert rr_spy.logged == [("runs/baseline/train/loss/total", 1.0)]

    def test_group_override(self, rr_spy: _RrSpy) -> None:
        logger = MetricLogger("run", spawn=True)
        logger.log({"mAP": 0.5}, epoch=2, group="val")
        assert rr_spy.times == [("epoch", 2)]
        assert rr_spy.logged == [("val/mAP", 0.5)]

    def test_every_throttles_on_step(self, rr_spy: _RrSpy) -> None:
        logger = MetricLogger("run", spawn=True)
        logger.log({"loss": 1.0}, step=3, every=50)  # 3 % 50 != 0 -> skipped
        logger.log({"loss": 2.0}, step=100, every=50)  # 100 % 50 == 0 -> logged
        assert rr_spy.logged == [("train/loss", 2.0)]

    def test_style_series_resolves_namespaced_entity(self, rr_spy: _RrSpy) -> None:
        logger = MetricLogger("run", spawn=True, prefix="runs/baseline")
        logger.style_series("loss/total", legend="baseline", color=(1, 2, 3))
        entity, _ = rr_spy.logged[0]
        assert entity == "runs/baseline/train/loss/total"


class TestMetricLoggerIntegration:
    def test_writes_nonempty_rrd(self, tmp_path: Path) -> None:
        # End-to-end against a real recording: data must reach the file.
        path = tmp_path / "run.rrd"
        with MetricLogger("vision3d_test_run", save_path=path) as logger:
            for step in range(5):
                logger.log({"loss/total": float(step), "lr": 1e-3}, step=step)
        assert path.exists()
        assert path.stat().st_size > 0
