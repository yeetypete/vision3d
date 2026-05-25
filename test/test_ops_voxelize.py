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
    max_voxels: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-Python voxelize reference matching the C++/CUDA op contract.

    Voxels are returned in ascending flat-cell-id (lexicographic ``(iz,
    iy, ix)``) order. Within a voxel, points are stored in input order
    up to ``max_points_per_voxel``. The cap on number of voxels picks
    the lowest-cell-id winners.

    Returns:
        ``(voxels, coords, num_points)`` matching the op's shapes.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    dx, dy, dz = voxel_size
    nx = round((x_max - x_min) / dx)
    ny = round((y_max - y_min) / dy)
    nz = round((z_max - z_min) / dz)

    p = points.cpu()
    pairs: list[tuple[int, int]] = []
    for i, row in enumerate(p):
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        if not (x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max):
            continue
        ix = min(int((x - x_min) / dx), nx - 1)
        iy = min(int((y - y_min) / dy), ny - 1)
        iz = min(int((z - z_min) / dz), nz - 1)
        pairs.append((iz * ny * nx + iy * nx + ix, i))
    pairs.sort(key=lambda kv: kv[0])

    unique_cells: list[int] = []
    voxel_points: list[list[int]] = []
    prev = -1
    for cell, pt in pairs:
        if cell != prev:
            if max_voxels is not None and len(unique_cells) >= max_voxels:
                break
            unique_cells.append(cell)
            voxel_points.append([])
            prev = cell
        if len(voxel_points[-1]) < max_points_per_voxel:
            voxel_points[-1].append(pt)

    pcount = len(unique_cells)
    c = p.shape[1]
    voxels = torch.zeros(pcount, max_points_per_voxel, c)
    coords = torch.zeros(pcount, 3, dtype=torch.int64)
    num_points = torch.zeros(pcount, dtype=torch.int64)
    for v, cell in enumerate(unique_cells):
        coords[v] = torch.tensor(
            [cell // (ny * nx), (cell // nx) % ny, cell % nx], dtype=torch.int64
        )
        pts = voxel_points[v]
        num_points[v] = len(pts)
        for slot, idx in enumerate(pts):
            voxels[v, slot] = p[idx]
    return voxels, coords, num_points


class TestVoxelize:
    """Behavior tests for the voxelize op (CPU and CUDA must agree exactly)."""

    def test_basic_3d_voxel_matches_reference(self, device: torch.device) -> None:
        torch.manual_seed(0)
        points = torch.rand(500, 4, device=device) * 10 - 5
        args = ((-5.0, -5.0, -5.0, 5.0, 5.0, 5.0), (1.0, 1.0, 1.0), 8)
        voxels, coords, num_points = voxelize(points, *args)
        ref_voxels, ref_coords, ref_num = _python_voxelize_reference(points, *args)
        torch.testing.assert_close(voxels.cpu(), ref_voxels.cpu())
        torch.testing.assert_close(coords.cpu(), ref_coords.cpu())
        torch.testing.assert_close(num_points.cpu(), ref_num.cpu())

    def test_pillar_mode_has_single_z_slice(self, device: torch.device) -> None:
        torch.manual_seed(1)
        points = torch.rand(200, 4, device=device) * 10 - 5
        _, coords, _ = voxelize(
            points,
            point_cloud_range=(-5.0, -5.0, -5.0, 5.0, 5.0, 5.0),
            voxel_size=(1.0, 1.0, 10.0),  # dz spans the full z range
            max_points_per_voxel=8,
        )
        assert coords.shape[1] == 3
        assert torch.equal(coords[:, 0].unique(), torch.tensor([0], device=device))

    def test_out_of_range_points_dropped(self, device: torch.device) -> None:
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
        assert int(num_points.sum()) == 2

    def test_max_points_cap_drops_surplus(self, device: torch.device) -> None:
        points = torch.zeros(10, 4, device=device)
        points[:, :3] = 0.5
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

    def test_max_voxels_caps_lowest_cell_ids(self, device: torch.device) -> None:
        # 10 input points, each landing in its own voxel (ix = 0..9). Cap at 4
        # keeps the lowest 4 cell ids. With this layout, ix = 0..3.
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
        assert num_points.tolist() == [1, 1, 1, 1]
        assert coords[:, 2].tolist() == [0, 1, 2, 3]

    def test_max_voxels_none_means_no_cap(self, device: torch.device) -> None:
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
        # Cells A=ix0, B=ix1, C=ix2, D=ix3. Cap=2 keeps the lowest two cell
        # ids (A and B). A has 3 points and B has 2. C and D are dropped.
        points = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.0],  # A
                [1.5, 0.5, 0.5, 0.0],  # B
                [0.5, 0.5, 0.5, 0.0],  # A again
                [1.5, 0.5, 0.5, 0.0],  # B again
                [0.5, 0.5, 0.5, 0.0],  # A third time
                [2.5, 0.5, 0.5, 0.0],  # C - dropped
                [3.5, 0.5, 0.5, 0.0],  # D - dropped
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
        assert num_points.tolist() == [3, 2]

    def test_extra_feature_channels_preserved(self, device: torch.device) -> None:
        points = torch.tensor(
            [[0.1, 0.1, 0.1, 7.0, 8.0, 9.0]], device=device, dtype=torch.float32
        )
        voxels, _, num_points = voxelize(
            points, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 4
        )
        assert voxels.shape == (1, 4, 6)
        assert num_points.item() == 1
        torch.testing.assert_close(voxels[0, 0], points[0])

    @pytest.mark.skip_device("cpu")
    def test_cpu_cuda_parity(self) -> None:
        # Stress test on a larger, more realistic input. CPU and CUDA must
        # produce bit-identical outputs (same row order, same per-voxel
        # point order).
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        # Construct on CPU explicitly: the conftest device fixture sets a
        # default-device context, but here we want both backends.
        points_cpu = torch.rand(20_000, 4, device="cpu") * 100 - 30
        args = ((0.0, -40.0, -3.0, 70.0, 40.0, 1.0), (0.5, 0.5, 4.0), 32)
        v_cpu, c_cpu, n_cpu = voxelize(points_cpu, *args)
        v_gpu, c_gpu, n_gpu = voxelize(points_cpu.cuda(), *args)
        torch.testing.assert_close(c_cpu, c_gpu.cpu())
        torch.testing.assert_close(n_cpu, n_gpu.cpu())
        torch.testing.assert_close(v_cpu, v_gpu.cpu())
