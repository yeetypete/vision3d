"""Tests for :mod:`vision3d.viz._logging` box logging logic.

These exercise the pure label-building and score-filtering logic without a
live Rerun recording: ``rr.log`` and ``rr.Boxes3D`` are spied on so the
arguments handed to Rerun can be asserted directly.
"""

import pytest
import torch

import vision3d.viz._logging as logging_mod
from vision3d.ops import extract_box3d_params
from vision3d.ops._rotation import _quaternion_to_matrix
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D
from vision3d.viz._logging import _build_labels, log_boxes_3d


def _boxes(
    n: int, fmt: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHY
) -> BoundingBoxes3D:
    """Build ``n`` random boxes in ``fmt``, with angles over the full circle.

    Returns:
        ``n`` random boxes in ``fmt``.
    """
    g = torch.Generator(torch.get_default_device()).manual_seed(0)
    center = torch.randn(n, 3, generator=g) * 10
    size = torch.rand(n, 3, generator=g) * 4 + 0.5
    angles = (torch.rand(n, 3, generator=g) * 2 - 1) * torch.pi
    raw = {
        BoundingBox3DFormat.XYZXYZ: torch.cat(
            [center - size / 2, center + size / 2], 1
        ),
        BoundingBox3DFormat.XYZLWH: torch.cat([center, size], 1),
        BoundingBox3DFormat.XYZLWHY: torch.cat([center, size, angles[:, :1]], 1),
        BoundingBox3DFormat.XYZLWHYPR: torch.cat([center, size, angles], 1),
    }[fmt]
    return BoundingBoxes3D(raw, format=fmt)


class _Spy:
    """Capture the kwargs passed to ``rr.Boxes3D``, ``rr.Arrows3D`` and ``rr.log``."""

    def __init__(self) -> None:
        self.boxes_kwargs: dict[str, object] | None = None
        self.arrows_kwargs: dict[str, object] | None = None
        self.logged: list[tuple[str, object]] = []

    def boxes3d(self, **kwargs: object) -> str:
        self.boxes_kwargs = kwargs
        return "boxes3d"

    def arrows3d(self, **kwargs: object) -> str:
        self.arrows_kwargs = kwargs
        return "arrows3d"

    def log(self, entity: str, archetype: object, **_: object) -> None:
        self.logged.append((entity, archetype))


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Patch Rerun's ``log``/``Boxes3D``/``Arrows3D`` to capture calls.

    Returns:
        The :class:`_Spy` recording calls into Rerun.
    """
    s = _Spy()
    monkeypatch.setattr(logging_mod.rr, "log", s.log)
    monkeypatch.setattr(logging_mod.rr, "Boxes3D", s.boxes3d)
    monkeypatch.setattr(logging_mod.rr, "Arrows3D", s.arrows3d)
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


class TestBoxPose:
    @pytest.mark.parametrize("fmt", list(BoundingBox3DFormat))
    def test_centers_sizes_and_rotation(
        self, spy: _Spy, fmt: BoundingBox3DFormat
    ) -> None:
        boxes = _boxes(16, fmt)
        log_boxes_3d("world/boxes", boxes, log_heading=False)
        assert spy.boxes_kwargs is not None
        centers, half_dims, rot = extract_box3d_params(
            boxes.as_subclass(torch.Tensor).cpu(), fmt
        )
        torch.testing.assert_close(spy.boxes_kwargs["centers"], centers)
        torch.testing.assert_close(spy.boxes_kwargs["sizes"], 2 * half_dims)
        # Axis-aligned formats log identity rotations, which overwrite any
        # rotations left on the entity by an earlier frame.
        quaternions = spy.boxes_kwargs["quaternions"]
        assert isinstance(quaternions, torch.Tensor)
        # Rerun's (x, y, z, w) back to (w, x, y, z).
        torch.testing.assert_close(
            _quaternion_to_matrix(quaternions[:, [3, 0, 1, 2]]), rot, atol=1e-5, rtol=0
        )

    def test_bfloat16_boxes(self, spy: _Spy) -> None:
        raw = torch.rand(4, 9, dtype=torch.bfloat16)
        log_boxes_3d(
            "world/boxes", BoundingBoxes3D(raw, format=BoundingBox3DFormat.XYZLWHYPR)
        )
        assert spy.boxes_kwargs is not None
        assert spy.arrows_kwargs is not None

    @pytest.mark.parametrize(
        "fmt", [BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR]
    )
    def test_heading_follows_box_forward_axis(
        self, spy: _Spy, fmt: BoundingBox3DFormat
    ) -> None:
        boxes = _boxes(16, fmt)
        log_boxes_3d("world/boxes", boxes)
        assert spy.arrows_kwargs is not None
        centers, half_dims, rot = extract_box3d_params(
            boxes.as_subclass(torch.Tensor).cpu(), fmt
        )
        forward = rot[:, :, 0]
        torch.testing.assert_close(
            spy.arrows_kwargs["origins"],
            centers + half_dims[:, :1] * forward,
            atol=1e-5,
            rtol=0,
        )
        vectors = spy.arrows_kwargs["vectors"]
        assert isinstance(vectors, torch.Tensor)
        torch.testing.assert_close(
            vectors / vectors.norm(dim=1, keepdim=True), forward, atol=1e-5, rtol=0
        )

    @pytest.mark.parametrize(
        ("fmt", "log_heading"),
        [
            (BoundingBox3DFormat.XYZLWH, True),
            (BoundingBox3DFormat.XYZXYZ, True),
            (BoundingBox3DFormat.XYZLWHY, False),
        ],
    )
    def test_no_heading_clears_stale_arrows(
        self, spy: _Spy, fmt: BoundingBox3DFormat, log_heading: bool
    ) -> None:
        log_boxes_3d("world/boxes", _boxes(4, fmt), log_heading=log_heading)
        assert spy.arrows_kwargs is None
        cleared = [e for e, a in spy.logged if isinstance(a, logging_mod.rr.Clear)]
        assert cleared == ["world/boxes/heading"]
