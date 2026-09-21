"""Tests for :class:`vision3d.viz.RerunLogger`.

These exercise the logger's disable switch, entity composition,
throttling, best-effort error handling, config flattening, and scene-method
routing. Rerun is replaced with a spy (:class:`_RrSpy`) so the calls the logger
makes can be asserted without a live recording; the integration tests drive a
real :class:`rerun.RecordingStream`.
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

import vision3d.viz._logger as logger_mod
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D
from vision3d.viz._errors import LoggingInputError
from vision3d.viz._logger import RerunLogger
from vision3d.viz._rerun import rr

if TYPE_CHECKING:
    from rerun import RecordingStream


class _FakeRecording:
    """Stand-in for the ``RecordingStream`` a caller hands to a logger.

    The logger must treat it as an opaque token forwarded as ``recording=``,
    so any attribute access fails the test.
    """

    def __getattr__(self, name: str) -> object:
        msg = f"RerunLogger must not drive the caller's recording (rec.{name})"
        raise AssertionError(msg)


def _recording() -> "RecordingStream":
    """Build a recording stand-in.

    Returns:
        A :class:`_FakeRecording` typed as the real thing.
    """
    return cast("RecordingStream", _FakeRecording())


class _RrSpy:
    """Spy standing in for the Rerun module inside :class:`RerunLogger`."""

    def __init__(self) -> None:
        self.logged: list[tuple[str, object]] = []
        self.times: list[tuple[str, int | None]] = []
        self.resets = 0
        self.properties: list[tuple[str, object]] = []
        self.recordings: list[object] = []

    def set_time(
        self,
        timeline: str,
        *,
        sequence: int | None = None,
        recording: object = None,
        **_: object,
    ) -> None:
        self.times.append((timeline, sequence))
        self.recordings.append(recording)

    def reset_time(self, *, recording: object = None, **_: object) -> None:
        self.resets += 1
        self.recordings.append(recording)

    def Scalars(self, value: float) -> float:
        return value

    def SeriesLines(self, **kwargs: object) -> tuple[str, dict[str, object]]:
        return ("series", kwargs)

    def log(
        self, entity: str, archetype: object, *, recording: object = None, **_: object
    ) -> None:
        self.logged.append((entity, archetype))
        self.recordings.append(recording)

    def send_property(
        self, name: str, values: object, *, recording: object = None, **_: object
    ) -> None:
        self.properties.append((name, values))
        self.recordings.append(recording)

    def AnyValues(
        self, *, drop_untyped_nones: bool = True, **kwargs: object
    ) -> tuple[str, dict[str, object]]:
        # Consume the constant flag so assertions focus on the config fields.
        return ("any", kwargs)


@pytest.fixture
def rr_spy(monkeypatch: pytest.MonkeyPatch) -> _RrSpy:
    """Replace every Rerun call :class:`RerunLogger` makes with a spy.

    Returns:
        The :class:`_RrSpy` recording the logging calls.
    """
    spy = _RrSpy()
    for attr in (
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


class TestRerunLoggerRecording:
    def test_forwards_the_given_recording(self, rr_spy: _RrSpy) -> None:
        # Every call must name this stream explicitly rather than relying on
        # whichever recording happens to be active.
        rec = _recording()
        logger = RerunLogger(rec)
        logger.log({"loss": 1.0}, step=0)
        assert logger.recording is rec
        assert set(rr_spy.recordings) == {rec}

    def test_none_targets_the_active_recording(self, rr_spy: _RrSpy) -> None:
        # An explicit ``None`` is forwarded unresolved, leaving the choice of
        # recording to Rerun at call time.
        logger = RerunLogger(None)
        logger.log({"loss": 1.0}, step=0)
        assert logger.recording is None
        assert rr_spy.recordings == [None, None]

    def test_disabled_never_touches_rerun(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording(), enabled=False)
        logger.log({"loss": 1.0}, step=0)
        logger.style_series("loss")
        assert not logger.enabled
        assert rr_spy.logged == []


class TestRerunLoggerLogging:
    def test_namespace_and_group_compose_entity(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording(), namespace="runs/baseline")
        logger.log({"loss/total": 1.0}, step=3)
        assert rr_spy.times == [("step", 3)]
        assert rr_spy.logged == [("runs/baseline/train/loss/total", 1.0)]

    def test_group_override(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording())
        logger.log({"mAP": 0.5}, epoch=2, group="val")
        assert rr_spy.times == [("epoch", 2)]
        assert rr_spy.logged == [("val/mAP", 0.5)]

    def test_every_throttles_on_step(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording())
        logger.log({"loss": 1.0}, step=3, every=50)  # 3 % 50 != 0 -> skipped
        logger.log({"loss": 2.0}, step=100, every=50)  # 100 % 50 == 0 -> logged
        assert rr_spy.logged == [("train/loss", 2.0)]

    def test_every_throttles_on_epoch_without_step(self, rr_spy: _RrSpy) -> None:
        # With no ``step``, the throttle falls back to ``epoch`` rather than
        # being silently ignored.
        logger = RerunLogger(_recording())
        logger.log({"mAP": 0.4}, epoch=3, every=2, group="val")  # 3 % 2 -> skipped
        logger.log({"mAP": 0.5}, epoch=4, every=2, group="val")  # 4 % 2 -> logged
        assert rr_spy.logged == [("val/mAP", 0.5)]

    def test_last_bypasses_throttle(self, rr_spy: _RrSpy) -> None:
        # The final step rarely lands on a multiple of ``every``; ``last=True``
        # forces it through so end-of-training metrics are not dropped.
        logger = RerunLogger(_recording())
        logger.log({"loss": 1.0}, step=99, every=50, last=True)
        assert rr_spy.logged == [("train/loss", 1.0)]

    @pytest.mark.parametrize("every", [0, -1])
    def test_every_below_one_raises(self, rr_spy: _RrSpy, every: int) -> None:
        # every=0 would be a ZeroDivisionError in the modulo throttle and
        # negatives never match: a caller bug, surfaced as LoggingInputError
        # rather than crashing (or silently disabling) the training loop.
        logger = RerunLogger(_recording())
        with pytest.raises(LoggingInputError, match="every must be >= 1"):
            logger.log({"loss": 1.0}, step=5, every=every)
        assert rr_spy.logged == []

    @pytest.mark.parametrize("every", [0, -1])
    def test_every_below_one_raises_even_when_disabled(self, every: int) -> None:
        # Validated ahead of the ``enabled`` guard: the argument is wrong
        # whoever is logging, so every rank of a group reports it.
        logger = RerunLogger(_recording(), enabled=False)
        with pytest.raises(LoggingInputError, match="every must be >= 1"):
            logger.log({"loss": 1.0}, step=5, every=every)

    def test_style_series_resolves_namespaced_entity(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording(), namespace="runs/baseline")
        logger.style_series("loss/total", legend="baseline", color=(1, 2, 3))
        entity, _ = rr_spy.logged[0]
        assert entity == "runs/baseline/train/loss/total"


class TestRerunLoggerErrorHandling:
    def test_warns_once_per_failing_action(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Failures are suppressed with one warning per action, not one per
        # logger: a second failing operation still surfaces even after another
        # has been silenced. Covers both dispatch paths -- the scalar ``log``
        # and the ``_scene`` wrappers.
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logger_mod, "log_scalars", boom)
        monkeypatch.setattr(logger_mod, "log_boxes_3d", boom)
        logger = RerunLogger(_recording())

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

    def test_warning_blames_the_caller(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A warning blaming _logger.py would break ``module=``-keyed filters.
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logger_mod, "log_scalars", boom)
        monkeypatch.setattr(logger_mod, "log_boxes_3d", boom)
        logger = RerunLogger(_recording())

        with pytest.warns(RuntimeWarning) as scalar_path:
            logger.log({"loss": 1.0}, step=0)
        with pytest.warns(RuntimeWarning) as scene_path:
            logger.log_boxes_3d("world/pred", _box())

        assert scalar_path[0].filename == __file__
        assert scene_path[0].filename == __file__

    def test_raise_on_error_propagates(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        monkeypatch.setattr(logger_mod, "log_scalars", boom)
        logger = RerunLogger(_recording(), raise_on_error=True)
        with pytest.raises(RuntimeError, match="rerun down"):
            logger.log({"loss": 1.0}, step=0)

    def test_input_error_warns_when_suppressing(self, rr_spy: _RrSpy) -> None:
        # A non-scalar metric is a caller bug, but a disabled logger validates
        # nothing, so raising would fire on the logging rank alone: warn instead.
        logger = RerunLogger(_recording())
        with pytest.warns(RuntimeWarning, match="'log' got malformed input"):
            logger.log({"loss": torch.tensor([1.0, 2.0])}, step=0)
        # The bad value never reached Rerun as a logged scalar.
        assert rr_spy.logged == []

    def test_input_error_propagates_with_raise_on_error(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording(), raise_on_error=True)
        with pytest.raises(LoggingInputError, match="must be a scalar"):
            logger.log({"loss": torch.tensor([1.0, 2.0])}, step=0)

    def test_input_error_warns_after_an_operational_failure(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Malformed input has its own rate-limit key, so a caller bug is still
        # reported after a sink failure has silenced the same action.
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("rerun down")

        real_log_scalars = logger_mod.log_scalars
        monkeypatch.setattr(logger_mod, "log_scalars", boom)
        logger = RerunLogger(_recording())
        with pytest.warns(RuntimeWarning, match="'log' failed"):
            logger.log({"loss": 1.0}, step=0)

        monkeypatch.setattr(logger_mod, "log_scalars", real_log_scalars)
        with pytest.warns(RuntimeWarning, match="'log' got malformed input"):
            logger.log({"loss": torch.tensor([1.0, 2.0])}, step=1)


class TestRerunLoggerConfig:
    def test_log_config_keeps_numbers_numeric(self, rr_spy: _RrSpy) -> None:
        # Numbers stay numeric so runs can be sorted/compared on them later;
        # bools render as readable text rather than 1/0.
        logger = RerunLogger(_recording())
        logger.log_config({"lr": 1e-3, "batch_size": 4, "amp": True})
        # Each property names its component after itself (not a shared "value")
        # so mixed-type configs do not collide on Rerun's per-component type.
        assert rr_spy.properties == [
            ("lr", ("any", {"lr": 1e-3})),
            ("batch_size", ("any", {"batch_size": 4})),
            ("amp", ("any", {"amp": "True"})),
        ]

    def test_log_config_unwraps_scalar_tensors(self, rr_spy: _RrSpy) -> None:
        # 0.5 is exact in float32, so the assertion needs no tolerance.
        logger = RerunLogger(_recording())
        logger.log_config({"lr": torch.tensor(0.5)})
        assert rr_spy.properties == [("lr", ("any", {"lr": 0.5}))]

    def test_log_config_stringifies_non_numbers(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording())
        logger.log_config({"scheduler": "cosine", "milestones": [10, 20]})
        assert rr_spy.properties == [
            ("scheduler", ("any", {"scheduler": "cosine"})),
            ("milestones", ("any", {"milestones": "[10, 20]"})),
        ]

    def test_log_config_flattens_nested_mappings(self, rr_spy: _RrSpy) -> None:
        # A Hydra/OmegaConf-style nested config flattens to dotted keys.
        logger = RerunLogger(_recording())
        logger.log_config({"optimizer": {"name": "adamw", "lr": 1e-3}})
        assert rr_spy.properties == [
            ("optimizer.name", ("any", {"optimizer.name": "adamw"})),
            ("optimizer.lr", ("any", {"optimizer.lr": 1e-3})),
        ]

    def test_log_config_disabled_is_noop(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording(), enabled=False)
        logger.log_config({"lr": 1e-3})
        assert rr_spy.properties == []


def _box() -> BoundingBoxes3D:
    """Build a single dummy 3D box for scene-method tests.

    Returns:
        A one-box :class:`BoundingBoxes3D` in XYZLWHY format.
    """
    return BoundingBoxes3D(torch.zeros(1, 7), format=BoundingBox3DFormat.XYZLWHY)


class TestRerunLoggerSceneMethods:
    """Scene methods honour the switch and route into the logger's recording."""

    def test_scene_methods_noop_when_disabled(
        self, rr_spy: _RrSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A disabled logger's scene methods must not touch Rerun at all, so
        # they can be called unconditionally from shared loop code.
        calls: list[str] = []
        for fn in ("log_point_cloud", "log_boxes_3d", "log_cameras", "log_sample"):
            monkeypatch.setattr(
                logger_mod, fn, lambda *a, _fn=fn, **k: calls.append(_fn)
            )
        logger = RerunLogger(_recording(), enabled=False)
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
        rec = _recording()
        logger = RerunLogger(rec)
        logger.log_point_cloud("world/lidar", torch.rand(4, 3), static=True)
        assert captured["entity"] == "world/lidar"
        assert captured["static"] is True
        # Explicitly targets the caller's stream, never the active recording.
        assert captured["recording"] is rec

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"step": 5, "epoch": 2}, [("step", 5), ("epoch", 2)]),
            ({"step": 5}, [("step", 5)]),
            ({"epoch": 2}, [("epoch", 2)]),
        ],
    )
    def test_set_time_moves_only_the_given_timelines(
        self,
        rr_spy: _RrSpy,
        kwargs: dict[str, int],
        expected: list[tuple[str, int]],
    ) -> None:
        # Omitting a timeline must leave its cursor alone rather than reset it,
        # so per-step and per-epoch logging can interleave.
        logger = RerunLogger(_recording())
        logger.set_time(**kwargs)
        assert rr_spy.times == expected

    def test_reset_time_clears_cursors(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording())
        logger.reset_time()
        assert rr_spy.resets == 1

    def test_reset_time_noop_when_disabled(self, rr_spy: _RrSpy) -> None:
        logger = RerunLogger(_recording(), enabled=False)
        logger.reset_time()
        assert rr_spy.resets == 0

    def test_scene_method_input_error_warns_when_suppressing(
        self, rr_spy: _RrSpy
    ) -> None:
        # The scene wrappers share the scalar path's boundary: malformed box
        # args warn there too rather than raise.
        logger = RerunLogger(_recording())
        with pytest.warns(RuntimeWarning, match="'log_boxes_3d' got malformed input"):
            logger.log_boxes_3d("world/pred", _box(), score_threshold=0.5)
        # The bad call never reached Rerun as a logged archetype.
        assert rr_spy.logged == []

    def test_scene_method_input_error_propagates_with_raise_on_error(
        self, rr_spy: _RrSpy
    ) -> None:
        logger = RerunLogger(_recording(), raise_on_error=True)
        with pytest.raises(LoggingInputError, match="score_threshold requires scores"):
            logger.log_boxes_3d("world/pred", _box(), score_threshold=0.5)


class TestRerunLoggerIntegration:
    """End-to-end against a real recording."""

    def test_writes_nonempty_rrd(self, tmp_path: Path) -> None:
        # Exiting the ``with`` block flushes and finalizes the file sink, so
        # the .rrd is complete by the time it is inspected.
        path = tmp_path / "run.rrd"
        with rr.RecordingStream("vision3d_test_run") as rec:
            rec.save(str(path))
            logger = RerunLogger(rec)
            for step in range(5):
                logger.log({"loss/total": float(step), "lr": 1e-3}, step=step)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_logs_into_the_active_recording(self, tmp_path: Path) -> None:
        # A recording that logged nothing is the baseline: the metrics really
        # reached the file only if it grows beyond that.
        empty = tmp_path / "empty.rrd"
        with rr.RecordingStream("vision3d_test_active") as rec:
            rec.save(str(empty))

        used = tmp_path / "used.rrd"
        with rr.RecordingStream("vision3d_test_active") as rec:
            rec.save(str(used))
            logger = RerunLogger(None)
            for step in range(5):
                logger.log({"loss/total": float(step)}, step=step)
        assert used.stat().st_size > empty.stat().st_size
