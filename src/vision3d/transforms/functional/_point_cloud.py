"""Functional kernels for point cloud transforms."""

from torch import Tensor

from vision3d.tensors import PointCloud3D

from ._registry import _get_kernel, register_kernel


def shuffle_points(inpt: Tensor, *, perm: Tensor) -> Tensor:
    """Shuffle the points of a point cloud.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        perm: Permutation indices ``[N]``.

    Returns:
        Shuffled tensor.
    """
    kernel = _get_kernel(shuffle_points, type(inpt))
    return kernel(inpt, perm=perm)


def shuffle_points_point_cloud(points: Tensor, *, perm: Tensor) -> Tensor:
    """Permute point order.

    Args:
        points: Point cloud ``[N, 3+C]``.
        perm: Permutation indices ``[N]``.

    Returns:
        Permuted point cloud with the same shape.
    """
    return points[perm]


@register_kernel(shuffle_points, PointCloud3D)
def _shuffle_points_kernel(points: Tensor, *, perm: Tensor) -> Tensor:
    return shuffle_points_point_cloud(points, perm=perm)


def sample_points(inpt: Tensor, *, indices: Tensor) -> Tensor:
    """Select points of a point cloud by index.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        indices: Selection indices ``[M]``. May contain repeats for
            oversampling.

    Returns:
        Sampled tensor.
    """
    kernel = _get_kernel(sample_points, type(inpt))
    return kernel(inpt, indices=indices)


def sample_points_point_cloud(points: Tensor, *, indices: Tensor) -> Tensor:
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
def _sample_points_kernel(points: Tensor, *, indices: Tensor) -> Tensor:
    return sample_points_point_cloud(points, indices=indices)


def jitter_points(inpt: Tensor, *, noise: Tensor) -> Tensor:
    """Add noise to the xyz coordinates of a point cloud.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        noise: Additive noise ``[N, 3]``.

    Returns:
        Jittered tensor.
    """
    kernel = _get_kernel(jitter_points, type(inpt))
    return kernel(inpt, noise=noise)


def jitter_points_point_cloud(points: Tensor, *, noise: Tensor) -> Tensor:
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
def _jitter_points_kernel(points: Tensor, *, noise: Tensor) -> Tensor:
    return jitter_points_point_cloud(points, noise=noise)
