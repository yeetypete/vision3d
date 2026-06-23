"""Functional kernel for temporal lidar sweep accumulation."""

from collections.abc import Sequence

import torch
from torch import Tensor


def accumulate_sweeps(
    sweeps: Sequence[Tensor],
    transforms: Tensor,
    time_offsets: Tensor,
) -> Tensor:
    """Accumulate and time-stamp a set of lidar sweeps.

    Each sweep is mapped into a common target frame by its own rigid
    transform, then all sweeps are concatenated into a single point cloud
    with a new trailing column holding the per-point time offset. This
    densifies a sparse lidar frame by folding in neighbouring sweeps while
    recording when each point was captured.

    The transform is applied to the ``(x, y, z)`` coordinates only. Feature
    columns (e.g. intensity) pass through unchanged and the time offset is
    appended after them, so a sweep of shape ``[N, 3+C]`` contributes rows of
    shape ``[N, 3+C+1]``.

    Args:
        sweeps: Sweeps to aggregate, each a ``[N_i, 3+C]`` point cloud whose
            first three columns are ``(x, y, z)`` in that sweep's own frame,
            followed by ``C`` feature columns. Every sweep must share the same
            number of feature columns.
        transforms: Rigid ``[S, 4, 4]`` homogeneous transforms, one per sweep,
            mapping that sweep's coordinates into the target frame.
        time_offsets: ``[S]`` per-sweep time offsets (e.g. seconds relative to
            the target frame) broadcast into the appended column.

    Returns:
        Aggregated ``[sum(N_i), 3+C+1]`` point cloud in the target frame, with
        the time offset as the last column. Rows follow the order of
        ``sweeps``.

    Raises:
        ValueError: If ``sweeps`` is empty, or if ``transforms`` or
            ``time_offsets`` do not have exactly one entry per sweep.
    """
    if len(sweeps) == 0:
        raise ValueError("accumulate_sweeps requires at least one sweep.")
    if transforms.shape[0] != len(sweeps) or time_offsets.shape[0] != len(sweeps):
        raise ValueError(
            f"Expected one transform and time offset per sweep, got "
            f"{len(sweeps)} sweeps, {transforms.shape[0]} transforms, and "
            f"{time_offsets.shape[0]} time offsets."
        )

    compensated: list[Tensor] = []
    for points, transform, dt in zip(sweeps, transforms, time_offsets):
        transform = transform.to(points.dtype)
        xyz = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        times = dt.to(points.dtype).reshape(1, 1).expand(points.shape[0], 1)
        compensated.append(torch.cat([xyz, points[:, 3:], times], dim=1))
    return torch.cat(compensated, dim=0)
