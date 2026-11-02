"""Functional kernels for point cloud transforms."""

import torch
from torch import Tensor

from vision3d.tensors import PointCloud3D

from ._registry import _get_kernel, register_kernel


def shuffle_points(inpt: Tensor) -> Tensor:
    """Randomly shuffle the points of a point cloud.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.

    Returns:
        Shuffled tensor.

    Raises:
        TypeError: If ``inpt`` is not an instance of
            :class:`~vision3d.tensors.PointCloud3D`.
    """  # noqa: DOC502
    kernel = _get_kernel(shuffle_points, type(inpt))
    return kernel(inpt)


def shuffle_points_point_cloud(points: Tensor) -> Tensor:
    """Randomly permute point order.

    Args:
        points: Point cloud ``[N, 3+C]``.

    Returns:
        Permuted point cloud with the same shape.
    """
    return points[torch.randperm(points.shape[0], device=points.device)]


@register_kernel(shuffle_points, PointCloud3D)
def _shuffle_points_kernel(points: Tensor) -> Tensor:
    return shuffle_points_point_cloud(points)


def sample_points(inpt: Tensor, *, n: int) -> Tensor:
    """Randomly sample a point cloud to ``n`` points.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        n: Target number of points.

    Returns:
        Sampled tensor.

    Raises:
        TypeError: If ``inpt`` is not an instance of
            :class:`~vision3d.tensors.PointCloud3D`.
    """  # noqa: DOC502
    kernel = _get_kernel(sample_points, type(inpt))
    return kernel(inpt, n=n)


def sample_points_point_cloud(points: Tensor, *, n: int) -> Tensor:
    """Randomly subsample (or oversample with replacement) to ``n`` points.

    Args:
        points: Point cloud ``[N, 3+C]``.
        n: Target number of points. If ``N >= n``, a random subset is
            selected. Otherwise points are sampled with replacement.

    Returns:
        Sampled point cloud ``[n, 3+C]``.
    """
    num_points = points.shape[0]
    if num_points >= n:
        indices = torch.randperm(num_points, device=points.device)[:n]
    else:
        indices = torch.randint(0, num_points, (n,), device=points.device)
    return points[indices]


@register_kernel(sample_points, PointCloud3D)
def _sample_points_kernel(points: Tensor, *, n: int) -> Tensor:
    return sample_points_point_cloud(points, n=n)


def jitter_points(inpt: Tensor, *, sigma: float) -> Tensor:
    """Add Gaussian noise to the xyz coordinates of a point cloud.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        sigma: Standard deviation of the noise.

    Returns:
        Jittered tensor.

    Raises:
        TypeError: If ``inpt`` is not an instance of
            :class:`~vision3d.tensors.PointCloud3D`.
    """  # noqa: DOC502
    kernel = _get_kernel(jitter_points, type(inpt))
    return kernel(inpt, sigma=sigma)


def jitter_points_point_cloud(points: Tensor, *, sigma: float) -> Tensor:
    """Add Gaussian noise to point xyz coordinates.

    Args:
        points: Point cloud ``[N, 3+C]``.
        sigma: Standard deviation of the noise.

    Returns:
        Jittered point cloud with the same shape. Non-xyz features
        are unchanged.
    """
    out = points.clone()
    out[:, :3] += torch.randn_like(out[:, :3]) * sigma
    return out


@register_kernel(jitter_points, PointCloud3D)
def _jitter_points_kernel(points: Tensor, *, sigma: float) -> Tensor:
    return jitter_points_point_cloud(points, sigma=sigma)
