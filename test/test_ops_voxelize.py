"""Tests for vision3d.ops.voxelize."""

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from vision3d.ops import voxelize


def _python_voxelize_reference(
    points: Tensor,
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
    max_points_per_voxel: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-Python voxelize reference implementation matching the C++ op signature.

    Returns:
        ``(voxels, coords, num_points)`` matching the C++ op's shapes.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    dx, dy, dz = voxel_size
    nx = round((x_max - x_min) / dx)
    ny = round((y_max - y_min) / dy)
    nz = round((z_max - z_min) / dz)

    p = points.cpu()
    cell_to_voxel: dict[tuple[int, int, int], int] = {}
    voxel_points: list[list[Tensor]] = []
    for row in p:
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        if not (x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max):
            continue
        ix = min(int((x - x_min) / dx), nx - 1)
        iy = min(int((y - y_min) / dy), ny - 1)
        iz = min(int((z - z_min) / dz), nz - 1)
        key = (iz, iy, ix)
        if key not in cell_to_voxel:
            cell_to_voxel[key] = len(cell_to_voxel)
            voxel_points.append([])
        v = cell_to_voxel[key]
        if len(voxel_points[v]) < max_points_per_voxel:
            voxel_points[v].append(row)

    pcount = len(cell_to_voxel)
    c = p.shape[1]
    voxels = torch.zeros(pcount, max_points_per_voxel, c)
    coords = torch.empty(pcount, 3, dtype=torch.int64)
    num_points = torch.zeros(pcount, dtype=torch.int64)
    for (iz, iy, ix), v in cell_to_voxel.items():
        coords[v] = torch.tensor([iz, iy, ix], dtype=torch.int64)
        pts = voxel_points[v]
        num_points[v] = len(pts)
        for slot, pt in enumerate(pts):
            voxels[v, slot] = pt
    return voxels, coords, num_points


@pytest.mark.skip_device("cuda")  # TODO: remove when CUDA kernel is implemented
class TestVoxelize:
    """Behavior tests for the voxelize op."""

    def test_basic_3d_voxel_matches_reference(self, device: torch.device) -> None:
        torch.manual_seed(0)
        points = torch.rand(500, 4, device=device) * 10 - 5  # range [-5, 5]
        args = ((-5.0, -5.0, -5.0, 5.0, 5.0, 5.0), (1.0, 1.0, 1.0), 8)
        voxels, coords, num_points = voxelize(points, *args)
        ref_voxels, ref_coords, ref_num = _python_voxelize_reference(
            points,
            *args,
        )
        torch.testing.assert_close(voxels.cpu(), ref_voxels)
        torch.testing.assert_close(coords.cpu(), ref_coords)
        torch.testing.assert_close(num_points.cpu(), ref_num)

    def test_pillar_mode_has_single_z_slice(self, device: torch.device) -> None:
        torch.manual_seed(1)
        points = torch.rand(200, 4, device=device) * 10 - 5
        _, coords, _ = voxelize(
            points,
            point_cloud_range=(-5.0, -5.0, -5.0, 5.0, 5.0, 5.0),
            voxel_size=(1.0, 1.0, 10.0),  # dz spans the full z range -> nz=1
            max_points_per_voxel=8,
        )
        assert coords.shape[1] == 3
        assert torch.equal(coords[:, 0].unique(), torch.tensor([0], device=device))

    def test_out_of_range_points_dropped(self, device: torch.device) -> None:
        # Half the points are far outside the range box.
        in_pts = torch.tensor(
            [[0.5, 0.5, 0.5, 0.0], [1.5, 1.5, 1.5, 0.0]], device=device
        )
        out_pts = torch.tensor(
            [[100.0, 0.0, 0.0, 0.0], [0.0, 0.0, -50.0, 0.0]], device=device
        )
        points = torch.cat([in_pts, out_pts])
        _, _, num_points = voxelize(
            points, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0), (1.0, 1.0, 1.0), 8
        )
        assert int(num_points.sum()) == 2  # only the two in-range points

    def test_max_points_cap_drops_surplus(self, device: torch.device) -> None:
        # 10 points all landing in the same single voxel.
        points = torch.zeros(10, 4, device=device)
        points[:, :3] = 0.5  # everything in cell (0, 0, 0)
        _, _, num_points = voxelize(
            points, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 4
        )
        assert num_points.tolist() == [4]

    def test_empty_input_returns_zero_voxels(self, device: torch.device) -> None:
        points = torch.empty(0, 4, device=device)
        voxels, coords, num_points = voxelize(
            points, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 4
        )
        assert voxels.shape == (0, 4, 4)
        assert coords.shape == (0, 3)
        assert num_points.shape == (0,)

    def test_all_points_out_of_range(self, device: torch.device) -> None:
        points = torch.full((50, 4), 100.0, device=device)
        voxels, coords, num_points = voxelize(
            points, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 4
        )
        assert voxels.shape[0] == 0
        assert coords.shape[0] == 0
        assert num_points.shape[0] == 0

    def test_max_voxels_caps_unique_cells(self, device: torch.device) -> None:
        # 10 input points, each landing in its own voxel. Cap at 4 -> only the
        # first 4 voxels in input order should survive. The trailing 6 points'
        # would-be voxels are dropped.
        points = torch.stack(
            [torch.tensor([i + 0.5, 0.5, 0.5, 0.0]) for i in range(10)]
        ).to(device)
        voxels, coords, num_points = voxelize(
            points,
            (0.0, 0.0, 0.0, 10.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            max_points_per_voxel=4,
            max_voxels=4,
        )
        assert voxels.shape[0] == 4
        assert coords.shape[0] == 4
        assert num_points.tolist() == [1, 1, 1, 1]
        # First four voxels' ix should be 0..3 (input order preserved).
        assert coords[:, 2].tolist() == [0, 1, 2, 3]

    def test_max_voxels_none_means_no_cap(self, device: torch.device) -> None:
        # 50 unique-cell points, no cap: every cell is kept.
        points = torch.stack(
            [torch.tensor([i + 0.5, 0.5, 0.5, 0.0]) for i in range(50)]
        ).to(device)
        _, _, num_points = voxelize(
            points,
            (0.0, 0.0, 0.0, 50.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            max_points_per_voxel=4,
            max_voxels=None,
        )
        assert num_points.shape[0] == 50

    def test_max_voxels_keeps_in_voxel_points_after_cap(
        self, device: torch.device
    ) -> None:
        # Pattern: voxels A, B, A, B, A, B, C, D. Cap=2 keeps {A, B} and
        # both A's later points go into A; C and D are dropped entirely.
        points = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.0],  # A
                [1.5, 0.5, 0.5, 0.0],  # B
                [0.5, 0.5, 0.5, 0.0],  # A again
                [1.5, 0.5, 0.5, 0.0],  # B again
                [0.5, 0.5, 0.5, 0.0],  # A third time
                [2.5, 0.5, 0.5, 0.0],  # C - dropped (cap hit)
                [3.5, 0.5, 0.5, 0.0],  # D - dropped (cap hit)
            ],
            device=device,
        )
        voxels, _, num_points = voxelize(
            points,
            (0.0, 0.0, 0.0, 5.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            max_points_per_voxel=8,
            max_voxels=2,
        )
        assert voxels.shape[0] == 2
        assert num_points.tolist() == [3, 2]  # A got 3, B got 2

    def test_extra_feature_channels_preserved(self, device: torch.device) -> None:
        # 6-channel points: xyz + 3 extra. The op must echo the extras
        # into voxels[:, :, 3:6] unchanged.
        points = torch.tensor(
            [[0.1, 0.1, 0.1, 7.0, 8.0, 9.0]], device=device, dtype=torch.float32
        )
        voxels, _, num_points = voxelize(
            points, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 4
        )
        assert voxels.shape == (1, 4, 6)
        assert num_points.item() == 1
        torch.testing.assert_close(voxels[0, 0], points[0])
