"""Tests for :func:`vision3d.viz.log_scalars`.

These exercise entity routing, timeline handling, and scalar coercion
without a live Rerun recording: ``rr.log``, ``rr.Scalars``, and
``rr.set_time`` are spied on so the arguments handed to Rerun can be
asserted directly.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import vision3d.viz._logging as logging_mod
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D
from vision3d.viz import time_series_view
from vision3d.viz._logging import (
    RerunLogger,
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


class _SpyRecording:
    """Stand-in for the ``RecordingStream`` a logger owns and targets."""

    def __init__(self, spy: "_RrSpy") -> None:
        self._spy = spy

    def flush(self, **_: object) -> None:
        self._spy.flushes += 1

    def disconnect(self) -> None:
        self._spy.disconnects += 1


class _RrSpy:
    """Spy standing in for the Rerun module inside :class:`RerunLogger`."""

    def __init__(self) -> None:
        self.inits: list[tuple[str, str | None, bool]] = []
        self.saves: list[str] = []
        self.grpc: list[str] = []
        self.blueprints: list[object] = []
        self.disconnects = 0
        self.flushes = 0
        self.logged: list[tuple[str, object]] = []
        self.times: list[tuple[str, int | None]] = []
        self.properties: list[tuple[str, object]] = []
        self._recording = _SpyRecording(self)

    def init(
        self, name: str, *, recording_id: str | None = None, spawn: bool = False
    ) -> None:
        self.inits.append((name, recording_id, spawn))

    def save(self, path: str, **_: object) -> None:
        self.saves.append(path)

    def connect_grpc(self, url: str, **_: object) -> None:
        self.grpc.append(url)

    def send_blueprint(self, blueprint: object, **_: object) -> None:
        self.blueprints.append(blueprint)

    def get_global_data_recording(self) -> _SpyRecording:
        return self._recording

    def set_time(
        self, timeline: str, *, sequence: int | None = None, **_: object
    ) -> None:
        self.times.append((timeline, sequence))

    def Scalars(self, value: float) -> float:
        return value

    def SeriesLines(self, **kwargs: object) -> tuple[str, dict[str, object]]:
        return ("series", kwargs)

    def log(self, entity: str, archetype: object, **_: object) -> None:
        self.logged.append((entity, archetype))

    def send_property(self, name: str, values: object, **_: object) -> None:
        self.properties.append((name, values))

    def AnyValues(self, **kwargs: object) -> tuple[str, dict[str, object]]:
        return ("any", kwargs)


@pytest.fixture
def rr_spy(monkeypatch: pytest.MonkeyPatch) -> _RrSpy:
    """Replace every Rerun call :class:`RerunLogger` makes with a spy.

    Returns:
        The :class:`_RrSpy` recording the lifecycle and logging calls.
    """
    spy = _RrSpy()
    for attr in (
        "init",
        "save",
        "connect_grpc",
        "send_blueprint",
        "get_global_data_recording",
        "set_time",
        "Scalars",
        "SeriesLines",
        "log",
        "send_property",
        "AnyValues",
    ):
        monkeypatch.setattr(logging_mod.rr, attr, getattr(spy, attr))
    return spy


class TestRerunLoggerLifecycle:
    def test_disabled_on_nonzero_rank(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", save_path="x.rrd", rank=1)
        logger.log({"loss": 1.0}, step=0)
        logger.style_series("loss")
        logger.close()
        assert not logger.enabled
        assert rr_spy.inits == []  # never touched Rerun on a non-zero rank
        assert rr_spy.logged == []

    def test_disabled_when_enabled_false(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", save_path="x.rrd", enabled=False)
        assert not logger.enabled
        assert rr_spy.inits == []

    def test_save_sink(self, rr_spy: _RrSpy) -> None:
        RerunLogger("run", save_path="out.rrd")
        assert rr_spy.inits == [("run", None, False)]
        assert rr_spy.saves == ["out.rrd"]

    def test_spawn_sink(self, rr_spy: _RrSpy) -> None:
        RerunLogger("run", spawn=True)
        assert rr_spy.inits == [("run", None, True)]
        assert rr_spy.saves == []

    def test_grpc_sink(self, rr_spy: _RrSpy) -> None:
        RerunLogger("run", grpc_url="rerun+http://host:9876/proxy")
        assert rr_spy.grpc == ["rerun+http://host:9876/proxy"]

    def test_blueprint_sent(self, rr_spy: _RrSpy) -> None:
        RerunLogger("run", blueprint=time_series_view(entity_prefix="train"))
        assert len(rr_spy.blueprints) == 1

    def test_multiple_sinks_raise(self) -> None:
        with pytest.raises(ValueError, match="at most one of"):
            RerunLogger("run", save_path="x.rrd", spawn=True)

    def test_context_manager_flushes_and_disconnects(self, rr_spy: _RrSpy) -> None:
        with RerunLogger("run", save_path="x.rrd"):
            pass
        assert rr_spy.flushes == 1
        assert rr_spy.disconnects == 1


class TestRerunLoggerLogging:
    def test_namespace_and_group_compose_entity(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True, prefix="runs/baseline")
        logger.log({"loss/total": 1.0}, step=3)
        assert rr_spy.times == [("step", 3)]
        assert rr_spy.logged == [("runs/baseline/train/loss/total", 1.0)]

    def test_group_override(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True)
        logger.log({"mAP": 0.5}, epoch=2, group="val")
        assert rr_spy.times == [("epoch", 2)]
        assert rr_spy.logged == [("val/mAP", 0.5)]

    def test_every_throttles_on_step(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True)
        logger.log({"loss": 1.0}, step=3, every=50)  # 3 % 50 != 0 -> skipped
        logger.log({"loss": 2.0}, step=100, every=50)  # 100 % 50 == 0 -> logged
        assert rr_spy.logged == [("train/loss", 2.0)]

    def test_last_bypasses_throttle(self, rr_spy: _RrSpy) -> None:
        # The final step rarely lands on a multiple of ``every``; ``last=True``
        # forces it through so end-of-training metrics are not dropped.
        logger = RerunLogger("run", spawn=True)
        logger.log({"loss": 1.0}, step=99, every=50, last=True)
        assert rr_spy.logged == [("train/loss", 1.0)]

    def test_style_series_resolves_namespaced_entity(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True, prefix="runs/baseline")
        logger.style_series("loss/total", legend="baseline", color=(1, 2, 3))
        entity, _ = rr_spy.logged[0]
        assert entity == "runs/baseline/train/loss/total"

    def test_close_is_idempotent(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True)
        logger.close()
        logger.close()
        assert rr_spy.disconnects == 1
        assert rr_spy.flushes == 1


class TestRerunLoggerErrorHandling:
    def test_best_effort_suppresses_and_warns_once(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logging_mod, "log_scalars", boom)
        logger = RerunLogger("run", spawn=True)

        # First failure is suppressed with a single warning (does not raise).
        with pytest.warns(RuntimeWarning, match="suppressed"):
            logger.log({"loss": 1.0}, step=0)

        # Subsequent failures are silent: turning warnings into errors must not
        # trip here, proving we only warn once.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            logger.log({"loss": 2.0}, step=1)

    def test_strict_propagates(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logging_mod, "log_scalars", boom)
        logger = RerunLogger("run", spawn=True, strict=True)
        with pytest.raises(RuntimeError, match="rerun down"):
            logger.log({"loss": 1.0}, step=0)


class TestRerunLoggerConfig:
    def test_log_config_keeps_numbers_numeric(self, rr_spy: _RrSpy) -> None:
        # Numbers stay numeric so runs can be sorted/compared on them later;
        # bools render as readable text rather than 1/0.
        logger = RerunLogger("run", spawn=True)
        logger.log_config({"lr": 1e-3, "batch_size": 4, "amp": True})
        # Each property names its component after itself (not a shared "value")
        # so mixed-type configs do not collide on Rerun's per-component type.
        assert rr_spy.properties == [
            ("lr", ("any", {"lr": 1e-3})),
            ("batch_size", ("any", {"batch_size": 4})),
            ("amp", ("any", {"amp": "True"})),
        ]

    def test_log_config_stringifies_non_numbers(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True)
        logger.log_config({"scheduler": "cosine", "milestones": [10, 20]})
        assert rr_spy.properties == [
            ("scheduler", ("any", {"scheduler": "cosine"})),
            ("milestones", ("any", {"milestones": "[10, 20]"})),
        ]

    def test_log_config_flattens_nested_mappings(self, rr_spy: _RrSpy) -> None:
        # A Hydra/OmegaConf-style nested config flattens to dotted keys.
        logger = RerunLogger("run", spawn=True)
        logger.log_config({"optimizer": {"name": "adamw", "lr": 1e-3}})
        assert rr_spy.properties == [
            ("optimizer.name", ("any", {"optimizer.name": "adamw"})),
            ("optimizer.lr", ("any", {"optimizer.lr": 1e-3})),
        ]

    def test_log_config_disabled_is_noop(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", save_path="x.rrd", rank=1)
        logger.log_config({"lr": 1e-3})
        assert rr_spy.properties == []


def _box() -> BoundingBoxes3D:
    """Build a single dummy 3D box for scene-method tests.

    Returns:
        A one-box :class:`BoundingBoxes3D` in XYZLWHY format.
    """
    return BoundingBoxes3D(torch.zeros(1, 7), format=BoundingBox3DFormat.XYZLWHY)


class TestRerunLoggerSceneMethods:
    """Scene methods are rank-aware and route into the logger's recording."""

    def test_scene_methods_noop_when_disabled(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Off rank 0 the scene methods must not touch Rerun at all, so they can
        # be called unconditionally from shared loop code.
        calls: list[str] = []
        for fn in ("log_point_cloud", "log_boxes_3d", "log_cameras", "log_sample"):
            monkeypatch.setattr(
                logging_mod, fn, lambda *a, _fn=fn, **k: calls.append(_fn)
            )
        logger = RerunLogger("run", save_path="x.rrd", rank=1)
        logger.log_point_cloud("world/lidar", torch.rand(4, 3))
        logger.log_boxes_3d("world/gt", _box())
        logger.log_cameras("world/cam", torch.rand(1, 3, 2, 2))
        logger.log_sample({})
        logger.set_time(step=0)
        assert calls == []
        assert rr_spy.times == []

    def test_scene_method_routes_to_this_recording(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            logging_mod,
            "log_point_cloud",
            lambda entity, points, **k: captured.update(entity=entity, **k),
        )
        logger = RerunLogger("run", spawn=True)
        logger.log_point_cloud("world/lidar", torch.rand(4, 3), static=True)
        assert captured["entity"] == "world/lidar"
        assert captured["static"] is True
        # Explicitly targets this logger's stream, never the global recording.
        assert captured["recording"] is logger.recording

    def test_set_time_sets_both_timelines(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True)
        logger.set_time(step=5, epoch=2)
        assert rr_spy.times == [("step", 5), ("epoch", 2)]

    def test_scene_method_is_best_effort(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logging_mod, "log_boxes_3d", boom)
        logger = RerunLogger("run", spawn=True)
        with pytest.warns(RuntimeWarning, match="suppressed"):
            logger.log_boxes_3d("world/pred", _box())


class TestRerunLoggerIntegration:
    def test_writes_nonempty_rrd(self, tmp_path: Path) -> None:
        # End-to-end against a real recording: data must reach the file.
        path = tmp_path / "run.rrd"
        with RerunLogger("vision3d_test_run", save_path=path) as logger:
            for step in range(5):
                logger.log({"loss/total": float(step), "lr": 1e-3}, step=step)
        assert path.exists()
        assert path.stat().st_size > 0
