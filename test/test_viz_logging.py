"""Tests for :mod:`vision3d.viz._logging` point cloud, box, and grid logging logic.

These exercise the pure label-building and score-filtering logic without a
live Rerun recording: ``rr.log`` and ``rr.Boxes3D`` are spied on so the
arguments handed to Rerun can be asserted directly.
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from common_utils import make_grid_affine
from torchvision.utils import _generate_color_palette

import vision3d.viz._logging as logging_mod
from vision3d.ops import extract_box3d_params
from vision3d.ops._rotation import _quaternion_to_rotation_matrix
from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, GridMask3D
from vision3d.transforms.functional._geometry import _rotation_matrix
from vision3d.viz._logging import (
    _build_labels,
    log_boxes_3d,
    log_grid_mask,
    log_point_cloud,
    log_sample,
)


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
    """Capture the kwargs passed to Rerun's boxes, arrows, and ``rr.log``."""

    def __init__(self, boxes_from_fields: Callable[..., object]) -> None:
        self._boxes_from_fields = boxes_from_fields
        self.boxes_kwargs: dict[str, object] | None = None
        self.arrows_kwargs: dict[str, object] | None = None
        self.logged: list[tuple[str, object]] = []

    def boxes3d(self, **kwargs: object) -> object:
        self.boxes_kwargs = kwargs
        return self._boxes_from_fields(**kwargs)

    def arrows3d(self, **kwargs: object) -> str:
        self.arrows_kwargs = kwargs
        return "arrows3d"

    def log(self, entity: str, archetype: object, **_: object) -> None:
        self.logged.append((entity, archetype))


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Patch Rerun's ``log``/``Boxes3D.from_fields``/``Arrows3D`` to capture calls.

    Returns:
        The :class:`_Spy` recording calls into Rerun.
    """
    s = _Spy(logging_mod.rr.Boxes3D.from_fields)
    monkeypatch.setattr(logging_mod.rr, "log", s.log)
    monkeypatch.setattr(logging_mod.rr.Boxes3D, "from_fields", s.boxes3d)
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

    def test_resolves_class_ids_without_scores(self) -> None:
        out = _build_labels(None, [1, 0], {"car": 0, "ped": 1}, None)
        assert out == ["ped", "car"]

    def test_none_when_nothing_to_display(self) -> None:
        assert _build_labels(None, [1, 0], None, None) is None


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
        torch.testing.assert_close(spy.boxes_kwargs["half_sizes"], half_dims)
        quaternions = spy.boxes_kwargs["quaternions"]
        assert isinstance(quaternions, torch.Tensor)
        # Rerun's (x, y, z, w) back to (w, x, y, z).
        torch.testing.assert_close(
            _quaternion_to_rotation_matrix(quaternions[:, [3, 0, 1, 2]]), rot
        )

    def test_float64_sizes(self, spy: _Spy) -> None:
        # float32 has a 24-bit significand, so it rounds 2**24 + 1 to 2**24.
        raw = torch.tensor([[2**24, 0, 0, 2**24 + 1, 1, 1]], dtype=torch.float64)
        log_boxes_3d(
            "world/boxes", BoundingBoxes3D(raw, format=BoundingBox3DFormat.XYZXYZ)
        )
        assert spy.boxes_kwargs is not None
        half_sizes = spy.boxes_kwargs["half_sizes"]
        assert isinstance(half_sizes, torch.Tensor)
        assert half_sizes.tolist() == [[0.5, 0.5, 0.5]]

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
            spy.arrows_kwargs["origins"], centers + half_dims[:, :1] * forward
        )
        vectors = spy.arrows_kwargs["vectors"]
        assert isinstance(vectors, torch.Tensor)
        torch.testing.assert_close(vectors / vectors.norm(dim=1, keepdim=True), forward)

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


class TestPointCloud:
    @pytest.mark.parametrize("color_by_distance", [True, False])
    def test_clears_unset_colors(self, spy: _Spy, color_by_distance: bool) -> None:
        points = torch.randn(8, 4)
        log_point_cloud("world/lidar", points, color_by_distance=color_by_distance)
        assert len(spy.logged) == 1
        entity, archetype = spy.logged[0]
        assert entity == "world/lidar"
        assert isinstance(archetype, logging_mod.rr.Points3D)
        assert archetype.positions is not None
        assert len(archetype.positions) == 8
        # An empty batch clears colors from earlier frames. None would keep them.
        assert archetype.colors is not None
        assert len(archetype.colors) == (8 if color_by_distance else 0)


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
    """Patch Rerun's ``log`` and ``VoxelGridMap`` to capture calls.

    Returns:
        The :class:`_VoxelSpy` recording calls into Rerun.
    """
    s = _VoxelSpy()
    monkeypatch.setattr(logging_mod.rr, "log", s.log)
    monkeypatch.setattr(logging_mod.rr, "VoxelGridMap", s.voxel_grid_map)
    return s


# Distinct cell sizes along the grid axes (d, h, w).
_CELL_SIZE = (0.2, 0.5, 0.3)


def _rotated_affine(*, reflect: bool = False) -> torch.Tensor:
    # Columns (d, h, w) scaled by ``_CELL_SIZE``, rotated about an oblique axis.
    axis = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    rotation = _rotation_matrix(axis, 0.7)
    affine = torch.eye(4, dtype=torch.float64)
    affine[:3, :3] = rotation @ torch.diag(
        torch.tensor(_CELL_SIZE, dtype=torch.float64)
    )
    if reflect:
        affine[:3, 1] = -affine[:3, 1]
    affine[:3, 3] = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    return affine


def _random_mask(*shape: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    mask = torch.rand(*shape, generator=generator, device="cpu") > 0.5
    return mask.to(torch.get_default_device())


def _voxel_centers(call: dict[str, Any]) -> torch.Tensor:
    # Rebuild voxel centers the way Rerun places them.
    xyzw = torch.tensor(call["quaternion"].xyzw, dtype=torch.float64, device="cpu")
    # Rerun's (x, y, z, w) back to (w, x, y, z).
    rotation = _quaternion_to_rotation_matrix(xyzw[[3, 0, 1, 2]])
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
        data = _random_mask(1, *spatial)
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
        # The in-plane cell sizes are kept, and the d axis becomes thinner than a cell.
        size = call["voxel_size"]
        assert size[:2] == pytest.approx(_CELL_SIZE[1:])
        assert 0 < size[2] < min(size[:2])
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
        # Clearing the whole entity also clears earlier voxels of empty channels.
        assert voxel_spy.cleared == ["world/map"]

    def test_integer_cells_are_colored_by_value(self, voxel_spy: _VoxelSpy) -> None:
        data = torch.tensor([[[0, 1], [2, 0]]])
        grid = GridMask3D(data, affine=make_grid_affine())
        log_grid_mask("world/occ", grid, colors=[(0, 0, 0), (10, 0, 0), (0, 20, 0)])
        call = voxel_spy.calls["world/occ"]
        # Background cells (0) are skipped, in row-major order (1 then 2).
        assert call["voxel_indices"].tolist() == [[0, 1, 0], [1, 0, 0]]
        assert call["colors"].tolist() == [[10, 0, 0], [0, 20, 0]]

    def test_negative_ids_get_default_colors(self, voxel_spy: _VoxelSpy) -> None:
        grid = GridMask3D(torch.tensor([[[-1]]]), affine=make_grid_affine())
        log_grid_mask("world/occ", grid)
        # torchvision's palette repeats every 255 ids, so -1 is colored as 254.
        expected = list(_generate_color_palette(255)[254])
        assert voxel_spy.calls["world/occ"]["colors"].tolist() == [expected]

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
        grid = GridMask3D(_random_mask(2, 3, 4, 5), affine=_rotated_affine())
        log_grid_mask("world/map", grid)
        logged = [type(a) for _, a in spy.logged]
        rr = logging_mod.rr
        assert logged == [rr.Clear, rr.VoxelGridMap, rr.VoxelGridMap]

    def test_log_sample_logs_map_target(self, voxel_spy: _VoxelSpy) -> None:
        grid = GridMask3D(torch.ones(1, 2, 2, dtype=torch.bool), affine=torch.eye(4))
        boxes = BoundingBoxes3D(torch.zeros(0, 7), format=BoundingBox3DFormat.XYZLWHY)
        targets = {"boxes": boxes, "labels": torch.zeros(0), "map": grid}
        log_sample({}, targets, map_colors=[(1, 2, 3)])
        call = voxel_spy.calls["world/gt/map"]
        assert call["opacity"] == pytest.approx(0.5)
        assert call["colors"].tolist() == [[1, 2, 3]]


class TestBoxColors:
    @pytest.mark.parametrize("scores", [None, [0.9, 0.1]])
    def test_uses_annotation_context_colors(
        self, spy: _Spy, scores: list[float] | None
    ) -> None:
        log_boxes_3d(
            "world/boxes",
            _boxes(2),
            class_ids=[1, 0],
            label_to_id={"car": 0, "ped": 1},
            scores=scores,
            log_heading=False,
        )
        boxes = [a for e, a in spy.logged if e == "world/boxes" and isinstance(a, list)]
        assert len(boxes) == 1
        logged = {
            b.component_descriptor().component: b.as_arrow_array() for b in boxes[0]
        }
        # An empty batch would hide the colors provided by the AnnotationContext.
        assert "Boxes3D:colors" not in logged
        assert len(logged["Boxes3D:labels"]) == 2
        # Unset fields are still cleared.
        assert len(logged["Boxes3D:radii"]) == 0
