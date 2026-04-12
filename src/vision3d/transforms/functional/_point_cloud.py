"""Functional kernels for point cloud transforms."""

from typing import TYPE_CHECKING

from vision3d.tensors import PointCloud3D

if TYPE_CHECKING:
    from torch import Tensor

from ._registry import register_kernel


def shuffle_points[N, C](inpt: Tensor[N, C], *, perm: Tensor[N]) -> Tensor[N, C]:
    """Dispatcher entry point for point shuffling.

    Returns:
        Input unchanged (passthrough for non-point types).
    """
    return inpt


def shuffle_points_point_cloud[N, C](
    points: Tensor[N, C], *, perm: Tensor[N]
) -> Tensor[N, C]:
    """Permute point order.

    Args:
        points: Point cloud ``[N, 3+C]``.
        perm: Permutation indices ``[N]``.

    Returns:
        Permuted point cloud with the same shape.
    """
    return points[perm]


@register_kernel(shuffle_points, PointCloud3D)
def _shuffle_points_kernel[N, C](
    points: Tensor[N, C], *, perm: Tensor[N]
) -> Tensor[N, C]:
    return shuffle_points_point_cloud(points, perm=perm)


def sample_points[N, C](inpt: Tensor[N, C], *, indices: Tensor[N]) -> Tensor[N, C]:
    """Dispatcher entry point for point sampling.

    Returns:
        Input unchanged (passthrough for non-point types).
    """
    return inpt


def sample_points_point_cloud[N, C, M](
    points: Tensor[N, C], *, indices: Tensor[M]
) -> Tensor[M, C]:
    """Select points by index.

    Args:
        points: Point cloud ``[N, 3+C]``.
        indices: Selection indices ``[M]``. May contain repeats for
            oversampling.

    Returns:
        Selected point cloud ``[M, 3+C]``.
    """
    return points[indices]


@register_kernel(sample_points, PointCloud3D)
def _sample_points_kernel[N, C, M](
    points: Tensor[N, C], *, indices: Tensor[M]
) -> Tensor[M, C]:
    return sample_points_point_cloud(points, indices=indices)


def jitter_points[N, C](inpt: Tensor[N, C], *, noise: Tensor[N, C]) -> Tensor[N, C]:
    """Dispatcher entry point for point jittering.

    Returns:
        Input unchanged (passthrough for non-point types).
    """
    return inpt


def jitter_points_point_cloud[N, C](
    points: Tensor[N, C], *, noise: Tensor[N, C]
) -> Tensor[N, C]:
    """Add noise to point xyz coordinates.

    Args:
        points: Point cloud ``[N, 3+C]``.
        noise: Additive noise ``[N, 3]``.

    Returns:
        Jittered point cloud with the same shape. Non-xyz features
        are unchanged.
    """
    out = points.clone()
    out[:, :3] += noise
    return out


@register_kernel(jitter_points, PointCloud3D)
def _jitter_points_kernel[N, C](
    points: Tensor[N, C], *, noise: Tensor[N, C]
) -> Tensor[N, C]:
    return jitter_points_point_cloud(points, noise=noise)
