"""Point cloud transform classes."""

from typing import Any, override

from vision3d.tensors import PointCloud3D

from ._transform import Transform, _RandomApplyTransform
from .functional._point_cloud import (
    jitter_points,
    sample_points,
    shuffle_points,
)


class PointShuffle(_RandomApplyTransform):
    """Randomly permute point order with probability ``p``.

    Each point cloud in the sample is shuffled independently, so clouds of
    different sizes are supported.

    Args:
        p: Probability of applying. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D,)

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the permutation.

        Returns:
            Shuffled input.
        """
        return self._call_kernel(shuffle_points, inpt)


class PointSample(Transform):
    """Subsample (or oversample with replacement) to exactly ``n`` points.

    If the point cloud has more than ``n`` points, a random subset is
    selected. If fewer, points are sampled with replacement to reach
    ``n``. Each point cloud in the sample is sampled independently, so clouds
    of different sizes are supported.

    Args:
        n: Target number of points.
    """

    _transformed_types = (PointCloud3D,)

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the sampling.

        Returns:
            Sampled input.
        """
        return self._call_kernel(sample_points, inpt, n=self.n)


class PointJitter(_RandomApplyTransform):
    """Add Gaussian noise to point xyz coordinates with probability ``p``.

    Each point cloud in the sample is jittered independently, i.e. the noise
    added to each cloud will be different.

    Args:
        sigma: Standard deviation of the Gaussian noise. Default: ``0.01``.
        p: Probability of applying. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D,)

    def __init__(self, sigma: float = 0.01, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.sigma = sigma

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the noise.

        Returns:
            Jittered input.
        """
        return self._call_kernel(jitter_points, inpt, sigma=self.sigma)
