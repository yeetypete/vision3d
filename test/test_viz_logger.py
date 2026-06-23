"""Tests for :class:`vision3d.viz.RerunLogger`.

These exercise the logger's lifecycle, rank-aware disabling, throttling,
best-effort error handling, config flattening, and scene-method routing. Rerun
is replaced with a spy (:class:`_RrSpy`) so the calls the logger makes can be
asserted without a live recording; one integration test writes a real ``.rrd``.
"""

import warnings
from pathlib import Path

import pytest
import torch

import vision3d.viz._logger as logger_mod
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D
from vision3d.viz import time_series_view
from vision3d.viz._errors import LoggingInputError
from vision3d.viz._logger import RerunLogger


class _SpyRecording:
    """Stand-in for the ``RecordingStream`` a logger owns and targets.

    The logger drives its sink through this stream's own methods (never the
    module-level ``rr.*`` helpers), so the sink calls are recorded here and
    funnelled back to the parent spy.
    """

    def __init__(self, spy: "_RrSpy") -> None:
        self._spy = spy

    def spawn(self, **_: object) -> None:
        self._spy.spawns += 1

    def save(self, path: str, **_: object) -> None:
        self._spy.saves.append(path)

    def connect_grpc(self, url: str, **_: object) -> None:
        self._spy.grpc.append(url)

    def send_blueprint(self, blueprint: object, **_: object) -> None:
        self._spy.blueprints.append(blueprint)

    def flush(self, **_: object) -> None:
        self._spy.flushes += 1

    def disconnect(self) -> None:
        self._spy.disconnects += 1


class _RrSpy:
    """Spy standing in for the Rerun module inside :class:`RerunLogger`."""

    def __init__(self) -> None:
        self.inits: list[tuple[str, str | None]] = []
        self.spawns = 0
        self.saves: list[str] = []
        self.grpc: list[str] = []
        self.blueprints: list[object] = []
        self.disconnects = 0
        self.flushes = 0
        self.logged: list[tuple[str, object]] = []
        self.times: list[tuple[str, int | None]] = []
        self.resets = 0
        self.properties: list[tuple[str, object]] = []
        self._recording = _SpyRecording(self)

    def RecordingStream(
        self, name: str, *, recording_id: str | None = None, **_: object
    ) -> _SpyRecording:
        # The logger owns a private stream rather than calling rr.init, so it
        # never registers a process-global recording. Each construction yields a
        # fresh stream -- as the real SDK does -- so two loggers in one process
        # cannot cross-talk; sink calls still funnel back here for assertions.
        self.inits.append((name, recording_id))
        self._recording = _SpyRecording(self)
        return self._recording

    def set_time(
        self, timeline: str, *, sequence: int | None = None, **_: object
    ) -> None:
        self.times.append((timeline, sequence))

    def reset_time(self, **_: object) -> None:
        self.resets += 1

    def Scalars(self, value: float) -> float:
        return value

    def SeriesLines(self, **kwargs: object) -> tuple[str, dict[str, object]]:
        return ("series", kwargs)

    def log(self, entity: str, archetype: object, **_: object) -> None:
        self.logged.append((entity, archetype))

    def send_property(self, name: str, values: object, **_: object) -> None:
        self.properties.append((name, values))

    def AnyValues(
        self, *, drop_untyped_nones: bool = True, **kwargs: object
    ) -> tuple[str, dict[str, object]]:
        # Consume the constant flag so assertions focus on the config fields.
        return ("any", kwargs)


@pytest.fixture
def rr_spy(monkeypatch: pytest.MonkeyPatch) -> _RrSpy:
    """Replace every Rerun call :class:`RerunLogger` makes with a spy.

    Returns:
        The :class:`_RrSpy` recording the lifecycle and logging calls.
    """
    spy = _RrSpy()
    for attr in (
        "RecordingStream",
        "set_time",
        "reset_time",
        "Scalars",
        "SeriesLines",
        "log",
        "send_property",
        "AnyValues",
    ):
        monkeypatch.setattr(logger_mod.rr, attr, getattr(spy, attr))
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
        assert rr_spy.inits == [("run", None)]
        assert rr_spy.saves == ["out.rrd"]
        assert rr_spy.spawns == 0

    def test_spawn_sink(self, rr_spy: _RrSpy) -> None:
        RerunLogger("run", spawn=True)
        assert rr_spy.inits == [("run", None)]
        assert rr_spy.spawns == 1
        assert rr_spy.saves == []

    def test_grpc_sink(self, rr_spy: _RrSpy) -> None:
        RerunLogger("run", grpc_url="rerun+http://host:9876/proxy")
        assert rr_spy.grpc == ["rerun+http://host:9876/proxy"]

    def test_blueprint_sent(self, rr_spy: _RrSpy) -> None:
        RerunLogger("run", blueprint=time_series_view(entity_prefix="train"))
        assert len(rr_spy.blueprints) == 1

    def test_loggers_own_distinct_streams(self, rr_spy: _RrSpy) -> None:
        # Each logger owns a private RecordingStream and never registers a
        # process-global recording, so two loggers in one process do not
        # cross-talk -- the footgun of routing through rr.init's global.
        a = RerunLogger("a", spawn=True)
        b = RerunLogger("b", spawn=True)
        assert a.recording is not b.recording
        assert rr_spy.inits == [("a", None), ("b", None)]

    def test_multiple_sinks_raise(self) -> None:
        with pytest.raises(ValueError, match="at most one of"):
            RerunLogger("run", save_path="x.rrd", spawn=True)

    def test_multiple_sinks_raise_off_rank_zero(self) -> None:
        # Config validation is a pure argument check: it must reject a bad
        # configuration on every rank, not just rank 0, so the error does not
        # depend on which process happens to run it.
        with pytest.raises(ValueError, match="at most one of"):
            RerunLogger("run", save_path="x.rrd", spawn=True, rank=1)

    def test_multiple_sinks_raise_when_disabled(self) -> None:
        # A logger should always be correctly configured: an invalid sink combo
        # is rejected even when logging is disabled outright, so the mistake
        # surfaces at construction rather than lurking until logging is enabled.
        with pytest.raises(ValueError, match="at most one of"):
            RerunLogger("run", save_path="x.rrd", spawn=True, enabled=False)

    def test_context_manager_flushes_and_disconnects(self, rr_spy: _RrSpy) -> None:
        with RerunLogger("run", save_path="x.rrd"):
            pass
        assert rr_spy.flushes == 1
        assert rr_spy.disconnects == 1

    def test_finalizer_flushes_when_not_closed(self, rr_spy: _RrSpy) -> None:
        # If the caller never calls close() (nor uses the context manager), the
        # fallback finalizer still flushes and disconnects the sink at GC /
        # interpreter exit, so a buffered save_path sink is not left truncated.
        logger = RerunLogger("run", save_path="x.rrd")
        assert logger._finalizer is not None
        assert logger._finalizer.alive
        logger._finalizer()  # simulate the finalizer firing at GC/exit
        assert rr_spy.flushes == 1
        assert rr_spy.disconnects == 1

    def test_close_detaches_finalizer(self, rr_spy: _RrSpy) -> None:
        # Explicit close() finalizes the sink and cancels the fallback so it
        # cannot disconnect a second time later.
        logger = RerunLogger("run", save_path="x.rrd")
        logger.close()
        assert logger._finalizer is not None
        assert not logger._finalizer.alive
        assert rr_spy.flushes == 1
        assert rr_spy.disconnects == 1


class TestRerunLoggerLogging:
    def test_namespace_and_group_compose_entity(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True, namespace="runs/baseline")
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

    def test_every_throttles_on_epoch_without_step(self, rr_spy: _RrSpy) -> None:
        # With no ``step``, the throttle falls back to ``epoch`` rather than
        # being silently ignored.
        logger = RerunLogger("run", spawn=True)
        logger.log({"mAP": 0.4}, epoch=3, every=2, group="val")  # 3 % 2 -> skipped
        logger.log({"mAP": 0.5}, epoch=4, every=2, group="val")  # 4 % 2 -> logged
        assert rr_spy.logged == [("val/mAP", 0.5)]

    def test_last_bypasses_throttle(self, rr_spy: _RrSpy) -> None:
        # The final step rarely lands on a multiple of ``every``; ``last=True``
        # forces it through so end-of-training metrics are not dropped.
        logger = RerunLogger("run", spawn=True)
        logger.log({"loss": 1.0}, step=99, every=50, last=True)
        assert rr_spy.logged == [("train/loss", 1.0)]

    @pytest.mark.parametrize("every", [0, -1])
    def test_every_below_one_raises(self, rr_spy: _RrSpy, every: int) -> None:
        # every=0 would be a ZeroDivisionError in the modulo throttle and
        # negatives never match: a caller bug, surfaced as LoggingInputError
        # rather than crashing (or silently disabling) the training loop.
        logger = RerunLogger("run", spawn=True)
        with pytest.raises(LoggingInputError, match="every must be >= 1"):
            logger.log({"loss": 1.0}, step=5, every=every)
        assert rr_spy.logged == []

    def test_style_series_resolves_namespaced_entity(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True, namespace="runs/baseline")
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

        monkeypatch.setattr(logger_mod, "log_scalars", boom)
        logger = RerunLogger("run", spawn=True)

        # First failure is suppressed with a single warning (does not raise).
        with pytest.warns(RuntimeWarning, match="suppressed"):
            logger.log({"loss": 1.0}, step=0)

        # Subsequent failures are silent: turning warnings into errors must not
        # trip here, proving we only warn once.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            logger.log({"loss": 2.0}, step=1)

    def test_warns_once_per_failing_action(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The warning is rate-limited per action, not once for the whole logger:
        # a different failing operation still surfaces once even after another
        # has already been silenced.
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logger_mod, "log_scalars", boom)
        monkeypatch.setattr(logger_mod, "log_boxes_3d", boom)
        logger = RerunLogger("run", spawn=True)

        # 'log' warns once, then goes quiet.
        with pytest.warns(RuntimeWarning, match="'log' failed"):
            logger.log({"loss": 1.0}, step=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            logger.log({"loss": 2.0}, step=1)

        # A distinct action ('log_boxes_3d') is not suppressed by 'log' having
        # already warned -- it surfaces its own first warning.
        with pytest.warns(RuntimeWarning, match="'log_boxes_3d' failed"):
            logger.log_boxes_3d("world/pred", _box())

    def test_strict_propagates(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logger_mod, "log_scalars", boom)
        logger = RerunLogger("run", spawn=True, strict=True)
        with pytest.raises(RuntimeError, match="rerun down"):
            logger.log({"loss": 1.0}, step=0)

    def test_input_error_propagates_even_when_not_strict(self, rr_spy: _RrSpy) -> None:
        # A non-scalar metric is a caller bug, not a transport hiccup: it must
        # surface even in best-effort mode rather than being swallowed.
        logger = RerunLogger("run", spawn=True)
        with pytest.raises(LoggingInputError, match="must be a scalar"):
            logger.log({"loss": torch.tensor([1.0, 2.0])}, step=0)
        # The bad value never reached Rerun as a logged scalar.
        assert rr_spy.logged == []


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
                logger_mod, fn, lambda *a, _fn=fn, **k: calls.append(_fn)
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
            logger_mod,
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

    def test_reset_time_clears_cursors(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", spawn=True)
        logger.reset_time()
        assert rr_spy.resets == 1

    def test_reset_time_noop_when_disabled(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger("run", save_path="x.rrd", rank=1)
        logger.reset_time()
        assert rr_spy.resets == 0

    def test_scene_method_is_best_effort(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logger_mod, "log_boxes_3d", boom)
        logger = RerunLogger("run", spawn=True)
        with pytest.warns(RuntimeWarning, match="suppressed"):
            logger.log_boxes_3d("world/pred", _box())

    def test_scene_method_input_error_propagates_when_not_strict(
        self, rr_spy: _RrSpy
    ) -> None:
        # Malformed box args are a caller bug, not a transport hiccup: they
        # must surface through the scene wrapper's best-effort boundary too,
        # not just the scalar path.
        logger = RerunLogger("run", spawn=True)
        with pytest.raises(LoggingInputError, match="score_threshold requires scores"):
            logger.log_boxes_3d("world/pred", _box(), score_threshold=0.5)
        # The bad call never reached Rerun as a logged archetype.
        assert rr_spy.logged == []


class TestRerunLoggerIntegration:
    def test_writes_nonempty_rrd(self, tmp_path: Path) -> None:
        # End-to-end against a real recording: data must reach the file.
        path = tmp_path / "run.rrd"
        with RerunLogger("vision3d_test_run", save_path=path) as logger:
            for step in range(5):
                logger.log({"loss/total": float(step), "lr": 1e-3}, step=step)
        assert path.exists()
        assert path.stat().st_size > 0
