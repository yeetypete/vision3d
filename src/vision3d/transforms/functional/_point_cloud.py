"""Functional kernels for point cloud transforms."""

from torch import Tensor

from vision3d.tensors import PointCloud3D

from ._registry import register_kernel


def filter_close_points(inpt: Tensor, *, radius: float) -> Tensor:
    """Dispatcher entry point for close-point filtering.

    So this does not do anything but makes sure the programm does not crash when
    the input is not a point cloud.

    Returns:
        Input unchanged (passthrough for non-point types).
    """
    return inpt


def filter_close_points_point_cloud(points: Tensor, *, radius: float) -> Tensor:
    """Remove points inside a square region around the sensor origin.

    A point is removed when both ``abs(x) < radius`` and
    ``abs(y) < radius``. Its z coordinate and feature columns do not affect
    filtering. This is like the close-point removal used by the nuScenes
    devkit.

    Args:
        points: Point cloud ``[N, 3+C]``.
        radius: Half-width of the excluded square in the xy plane.

    Returns:
        Filtered point cloud ``[M, 3+C]`` with the original row order and
        feature columns preserved. Missing filtered points are removed.
    """
    close = (points[:, :2].abs() < radius).all(dim=1)
    return points[~close]


@register_kernel(filter_close_points, PointCloud3D)
def _filter_close_points_kernel(points: Tensor, *, radius: float) -> Tensor:
    return filter_close_points_point_cloud(points, radius=radius)


def shuffle_points(inpt: Tensor, *, perm: Tensor) -> Tensor:
    """Dispatcher entry point for point shuffling.

    Returns:
        Input unchanged (passthrough for non-point types).
    """
    return inpt


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
    """Dispatcher entry point for point sampling.

    Returns:
        Input unchanged (passthrough for non-point types).
    """
    return inpt


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
    """Dispatcher entry point for point jittering.

    Returns:
        Input unchanged (passthrough for non-point types).
    """
    return inpt


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
