"""Tests for :mod:`vision3d.viz._logging` box and grid logging logic.

These exercise the pure label-building and score-filtering logic without a
live Rerun recording: ``rr.log`` and ``rr.Boxes3D`` are spied on so the
arguments handed to Rerun can be asserted directly.
"""

import math
from typing import Any

import pytest
import torch
from common_utils import make_grid_affine
from torchvision.utils import _generate_color_palette

import vision3d.viz._logging as logging_mod
from vision3d.datasets.nuscenes import _quaternion_to_rotation_matrix
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, GridMask3D
from vision3d.viz._logging import (
    _build_labels,
    log_boxes_3d,
    log_grid_mask,
    log_sample,
)


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


class _VoxelSpy:
    """Capture ``rr.VoxelGridMap`` calls together with their entity paths."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}
        self.cleared: list[str] = []
        self._pending: dict[str, Any] | None = None

    def voxel_grid_map(self, voxel_indices: object, **kwargs: Any) -> str:
        self._pending = {"voxel_indices": voxel_indices, **kwargs}
        return "voxel_grid_map"

    def log(self, entity: str, archetype: object, **_: object) -> None:
        if isinstance(archetype, logging_mod.rr.Clear):
            self.cleared.append(entity)
        elif archetype == "voxel_grid_map":
            assert self._pending is not None
            self.calls[entity] = self._pending


@pytest.fixture
def voxel_spy(monkeypatch: pytest.MonkeyPatch) -> _VoxelSpy:
    """Patch Rerun's ``log``/``VoxelGridMap`` to capture calls.

    Returns:
        The :class:`_VoxelSpy` recording calls into Rerun.
    """
    s = _VoxelSpy()
    monkeypatch.setattr(logging_mod.rr, "log", s.log)
    monkeypatch.setattr(logging_mod.rr, "VoxelGridMap", s.voxel_grid_map)
    return s


def _rotated_affine(*, reflect: bool = False) -> torch.Tensor:
    # Columns (d, h, w) with distinct cell sizes, rotated about an oblique axis.
    angle = 0.7
    axis = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    axis = axis / torch.linalg.norm(axis)
    k = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=torch.float64,
    )
    rotation = torch.eye(3, dtype=torch.float64) + math.sin(angle) * k
    rotation += (1 - math.cos(angle)) * (k @ k)
    affine = torch.eye(4, dtype=torch.float64)
    affine[:3, :3] = rotation @ torch.diag(
        torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    )
    if reflect:
        affine[:3, 1] = -affine[:3, 1]
    affine[:3, 3] = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    return affine


def _voxel_centers(call: dict[str, Any]) -> torch.Tensor:
    # Rebuild voxel centers the way Rerun places them.
    x, y, z, w = call["quaternion"].xyzw.tolist()
    rotation = torch.from_numpy(_quaternion_to_rotation_matrix([w, x, y, z]))
    size = torch.tensor(call["voxel_size"], dtype=torch.float64, device="cpu")
    translation = torch.tensor(call["translation"], dtype=torch.float64, device="cpu")
    indices = call["voxel_indices"].double()
    return ((indices + 0.5) * size) @ rotation.T + translation


def _cell_centers(grid: GridMask3D) -> torch.Tensor:
    # The affine lives on the CPU, so the cells are gathered there too.
    data = grid.as_subclass(torch.Tensor)[0].cpu()
    if data.ndim == 2:
        data = data.unsqueeze(0)
    cells = data.nonzero().double()
    ones = torch.ones(len(cells), 1, dtype=torch.float64, device="cpu")
    homogeneous = torch.cat([cells, ones], 1)
    return (homogeneous @ grid.affine.T)[:, :3]


class TestLogGridMask:
    @pytest.mark.parametrize("reflect", [False, True], ids=["rotation", "reflection"])
    @pytest.mark.parametrize("spatial", [(6, 7), (3, 6, 7)], ids=["planar", "volume"])
    def test_voxels_land_on_cell_centers(
        self, voxel_spy: _VoxelSpy, spatial: tuple[int, ...], reflect: bool
    ) -> None:
        data = torch.rand(1, *spatial) > 0.5
        grid = GridMask3D(data, affine=_rotated_affine(reflect=reflect))
        log_grid_mask("world/map", grid)
        torch.testing.assert_close(
            _voxel_centers(voxel_spy.calls["world/map"]), _cell_centers(grid)
        )

    @pytest.mark.parametrize("reflect", [False, True], ids=["rotation", "reflection"])
    def test_planar_grids_are_flat(self, voxel_spy: _VoxelSpy, reflect: bool) -> None:
        grid = GridMask3D(
            torch.ones(1, 2, 3, dtype=torch.bool),
            affine=_rotated_affine(reflect=reflect),
        )
        log_grid_mask("world/map", grid)
        call = voxel_spy.calls["world/map"]
        # The in-plane cell sizes are kept, and the 0.2 m layer becomes thin.
        size = call["voxel_size"]
        assert size[:2] == pytest.approx([0.5, 0.3])
        assert 0 < size[2] < 1e-3
        assert call["voxel_indices"][:, 2].eq(0).all()

    def test_bool_channels_get_one_entity_and_color_each(
        self, voxel_spy: _VoxelSpy
    ) -> None:
        data = torch.zeros(3, 4, 4, dtype=torch.bool)
        data[0, 1, 1] = data[1, 2, 2] = True
        grid = GridMask3D(data, affine=make_grid_affine())
        log_grid_mask(
            "world/map",
            grid,
            channel_names=["road", "walkway", "empty"],
            colors=[(1, 2, 3), (4, 5, 6)],
        )
        assert set(voxel_spy.calls) == {"world/map/road", "world/map/walkway"}
        assert voxel_spy.calls["world/map/road"]["colors"].tolist() == [[1, 2, 3]]
        assert voxel_spy.calls["world/map/walkway"]["colors"].tolist() == [[4, 5, 6]]
        # An empty channel clears what an earlier frame logged there.
        assert voxel_spy.cleared == ["world/map/empty"]

    def test_integer_cells_are_colored_by_value(self, voxel_spy: _VoxelSpy) -> None:
        data = torch.tensor([[[0, 1], [2, 0]]])
        grid = GridMask3D(data, affine=make_grid_affine())
        log_grid_mask("world/occ", grid, colors=[(0, 0, 0), (10, 0, 0), (0, 20, 0)])
        call = voxel_spy.calls["world/occ"]
        # Background cells (0) are skipped, in row-major order (1 then 2).
        assert call["voxel_indices"].tolist() == [[0, 1, 0], [1, 0, 0]]
        assert call["colors"].tolist() == [[10, 0, 0], [0, 20, 0]]

    def test_default_colors_match_torchvision_palette(
        self, voxel_spy: _VoxelSpy
    ) -> None:
        grid = GridMask3D(torch.tensor([[[0, 3], [7, 0]]]), affine=make_grid_affine())
        log_grid_mask("world/occ", grid)
        palette = _generate_color_palette(8)
        expected = [[int(v) for v in palette[i]] for i in (3, 7)]
        assert voxel_spy.calls["world/occ"]["colors"].tolist() == expected

    def test_float_channels_use_their_own_value_range(
        self, voxel_spy: _VoxelSpy
    ) -> None:
        data = torch.tensor([[[0.5, 1.0]], [[2.0, 4.0]]])
        grid = GridMask3D(data, affine=make_grid_affine())
        log_grid_mask("world/soft", grid, background=None)
        assert voxel_spy.calls["world/soft/0"]["value_range"] == (0.5, 1.0)
        assert voxel_spy.calls["world/soft/1"]["value_range"] == (2.0, 4.0)
        assert voxel_spy.calls["world/soft/0"]["colors"] is None

    def test_sheared_affine_raises(self, voxel_spy: _VoxelSpy) -> None:
        affine = make_grid_affine()
        affine[0, 1] = 0.3
        grid = GridMask3D(torch.ones(1, 2, 2, dtype=torch.bool), affine=affine)
        with pytest.raises(ValueError, match="orthogonal"):
            log_grid_mask("world/map", grid)

    def test_channel_names_length_mismatch_raises(self, voxel_spy: _VoxelSpy) -> None:
        grid = GridMask3D(torch.ones(2, 2, 2, dtype=torch.bool), affine=torch.eye(4))
        with pytest.raises(ValueError, match="channel_names has length 1"):
            log_grid_mask("world/map", grid, channel_names=["only-one"])

    def test_builds_real_archetype(self, spy: _Spy) -> None:
        # Construct the real archetype to check that Rerun accepts the inputs.
        grid = GridMask3D(torch.rand(2, 3, 4, 5) > 0.5, affine=_rotated_affine())
        log_grid_mask("world/map", grid)
        logged = [type(a) for _, a in spy.logged]
        assert logged == [logging_mod.rr.VoxelGridMap] * 2

    def test_log_sample_logs_map_target(self, voxel_spy: _VoxelSpy) -> None:
        grid = GridMask3D(torch.ones(1, 2, 2, dtype=torch.bool), affine=torch.eye(4))
        boxes = BoundingBoxes3D(torch.zeros(0, 7), format=BoundingBox3DFormat.XYZLWHY)
        targets = {"boxes": boxes, "labels": torch.zeros(0), "map": grid}
        log_sample({}, targets, map_colors=[(1, 2, 3)])
        call = voxel_spy.calls["world/gt/map"]
        assert call["opacity"] == pytest.approx(0.5)
        assert call["colors"].tolist() == [[1, 2, 3]]
