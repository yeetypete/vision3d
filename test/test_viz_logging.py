"""Tests for :mod:`vision3d.viz._logging` box logging logic.

These exercise the pure label-building and score-filtering logic without a
live Rerun recording: ``rr.log`` and ``rr.Boxes3D`` are spied on so the
arguments handed to Rerun can be asserted directly.
"""

import pytest
import torch

import vision3d.viz._logging as logging_mod
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D
from vision3d.viz._logging import _build_labels, log_boxes_3d, log_point_cloud


def _boxes(n: int) -> BoundingBoxes3D:
    """Build ``n`` arbitrary yaw-rotated boxes.

    Returns:
        ``n`` random ``XYZLWHY`` boxes.
    """
    return BoundingBoxes3D(torch.rand(n, 7), format=BoundingBox3DFormat.XYZLWHY)


class _Spy:
    """Capture the kwargs passed to ``rr.Boxes3D`` and ``rr.log`` calls."""

    def __init__(self) -> None:
        self.boxes_kwargs: dict[str, object] | None = None
        self.logged: list[tuple[str, object]] = []

    def boxes3d(self, **kwargs: object) -> str:
        self.boxes_kwargs = kwargs
        return "boxes3d"

    def log(self, entity: str, archetype: object, **_: object) -> None:
        self.logged.append((entity, archetype))


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Patch Rerun's ``log``/``Boxes3D`` to capture calls.

    Returns:
        The :class:`_Spy` recording calls into Rerun.
    """
    s = _Spy()
    monkeypatch.setattr(logging_mod.rr, "log", s.log)
    monkeypatch.setattr(logging_mod.rr, "Boxes3D", s.boxes3d)
    return s


class TestBuildLabels:
    def test_resolves_class_ids_via_label_to_id(self) -> None:
        out = _build_labels(None, [1, 0], {"car": 0, "ped": 1}, [0.9, 0.1])
        assert out == ["ped 0.90", "car 0.10"]

    def test_scores_only_when_no_names_available(self) -> None:
        assert _build_labels(None, None, None, [0.25, 0.75]) == ["0.25", "0.75"]

    def test_unknown_class_id_falls_back_to_str(self) -> None:
        out = _build_labels(None, [7], {"car": 0}, [0.5])
        assert out == ["7 0.50"]


class TestScoreFiltering:
    def test_threshold_drops_low_scores_and_aligns_fields(self, spy: _Spy) -> None:
        log_boxes_3d(
            "world/pred/boxes",
            _boxes(3),
            labels=["a", "b", "c"],
            class_ids=[10, 11, 12],
            scores=[0.2, 0.9, 0.5],
            score_threshold=0.4,
            log_heading=False,
        )
        assert spy.boxes_kwargs is not None
        # Only boxes 1 and 2 survive the 0.4 threshold, in order.
        assert spy.boxes_kwargs["class_ids"] == [11, 12]
        assert spy.boxes_kwargs["labels"] == ["b 0.90", "c 0.50"]

    def test_all_filtered_out_clears_entity(self, spy: _Spy) -> None:
        log_boxes_3d(
            "world/pred/boxes",
            _boxes(2),
            scores=[0.1, 0.2],
            score_threshold=0.5,
            log_heading=False,
        )
        # Nothing should be drawn; the entity is cleared instead.
        assert spy.boxes_kwargs is None
        assert len(spy.logged) == 1
        entity, archetype = spy.logged[0]
        assert entity == "world/pred/boxes"
        assert isinstance(archetype, logging_mod.rr.Clear)


class TestValidation:
    def test_labels_length_mismatch_raises(self, spy: _Spy) -> None:
        with pytest.raises(ValueError, match="labels has length 1 but there are 2"):
            log_boxes_3d("world/pred/boxes", _boxes(2), labels=["only-one"])

    def test_scores_length_mismatch_raises(self, spy: _Spy) -> None:
        with pytest.raises(ValueError, match="scores has length 1 but there are 2"):
            log_boxes_3d("world/pred/boxes", _boxes(2), scores=[0.5])

    def test_class_ids_length_mismatch_raises(self, spy: _Spy) -> None:
        with pytest.raises(ValueError, match="class_ids has length 1 but there are 2"):
            log_boxes_3d("world/pred/boxes", _boxes(2), class_ids=[10])

    def test_score_threshold_without_scores_raises(self, spy: _Spy) -> None:
        with pytest.raises(ValueError, match="score_threshold requires scores"):
            log_boxes_3d("world/pred/boxes", _boxes(2), score_threshold=0.5)


class TestStatic:
    """``static=True`` should reach every ``rr.log`` an entity emits."""

    def test_point_cloud_propagates_static(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[bool | None] = []
        monkeypatch.setattr(
            logging_mod.rr,
            "log",
            lambda _entity, _archetype, **k: seen.append(k.get("static")),
        )
        log_point_cloud("world/lidar", torch.rand(5, 3), static=True)
        assert seen == [True]

    def test_boxes_propagate_static_to_geometry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, bool | None]] = []
        monkeypatch.setattr(
            logging_mod.rr,
            "log",
            lambda entity, _archetype, **k: seen.append((entity, k.get("static"))),
        )
        log_boxes_3d("world/gt/boxes", _boxes(2), static=True, log_heading=False)
        # The box geometry carries static=True.
        assert ("world/gt/boxes", True) in seen

    def test_static_defaults_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[bool | None] = []
        monkeypatch.setattr(
            logging_mod.rr,
            "log",
            lambda _entity, _archetype, **k: seen.append(k.get("static")),
        )
        log_point_cloud("world/lidar", torch.rand(5, 3))
        assert seen == [False]
