"""Bucket points into a 3D voxel grid (pillar or true voxel)."""

import torch
from torch import Tensor

from vision3d import _extension  # noqa: F401  # loads ``_C`` into torch.ops
from vision3d.ops import _meta_registrations  # noqa: F401  # registers fake kernels


@torch.no_grad()
def voxelize(
    points: Tensor,
    point_cloud_range: tuple[float, float, float, float, float, float],
    voxel_size: tuple[float, float, float],
    max_points_per_voxel: int = 32,
    max_voxels: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Bucket points into a 3D voxel grid.

    For PointPillars-style pillars set ``voxel_size = (dx, dy, z_max -
    z_min)`` so the grid has a single z-slice and every kept point
    lands at ``iz = 0``. For 3D voxel detectors (VoxelNet, SECOND,
    CenterPoint-Voxel) use a normal three-axis ``voxel_size``.

    The op is not differentiable.

    Args:
        points: ``[N, C]`` float32 point cloud. The first three columns
            must be ``(x, y, z)``. Remaining columns may be arbitrary
            per-point features. Points outside ``point_cloud_range``
            along any axis are discarded.
        point_cloud_range: ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
        voxel_size: ``(dx, dy, dz)`` voxel size.
        max_points_per_voxel: Cap on points stored per voxel. Surplus
            points are dropped in input order. Default: ``32``.
        max_voxels: Optional cap on the number of output voxels. ``None``
            (default) means no cap. When set, points that would have
            created a new voxel beyond ``max_voxels`` are silently
            dropped. Points landing in already-allocated voxels still
            get written.

    Returns:
        Tuple of:

        * ``voxels``: ``[P, max_points_per_voxel, C]`` per-voxel point
          buffers padded with zeros at unfilled slots.
        * ``coords``: ``[P, 3]`` ``int64`` ``(iz, iy, ix)`` voxel
          indices.
        * ``num_points``: ``[P]`` ``int64`` point counts per voxel,
          each in ``[1, max_points_per_voxel]``.

        ``P`` is the number of non-empty voxels (capped at ``max_voxels``
        when set). When no points fall inside ``point_cloud_range``,
        all three tensors have a leading dimension of zero.
    """
    range_t = torch.as_tensor(point_cloud_range, dtype=torch.float32, device="cpu")
    size_t = torch.as_tensor(voxel_size, dtype=torch.float32, device="cpu")
    return torch.ops.vision3d.voxelize(
        points,
        range_t,
        size_t,
        max_points_per_voxel,
        -1 if max_voxels is None else max_voxels,
    )
