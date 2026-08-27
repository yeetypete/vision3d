"""Point cloud transform classes."""

from typing import Any, override

import torch

from vision3d.tensors import PointCloud3D

from ._transform import Transform, _RandomApplyTransform
from .functional._point_cloud import (
    filter_close_points,
    jitter_points,
    sample_points,
    shuffle_points,
)


class ClosePointFilter(Transform):
    """This function filters out points close to the sensor origin.

    Points are removed from the open square
    ``abs(x) < radius and abs(y) < radius``. The z coordinate is ignored here
    aswell, like it is the case in the ego self-return filtering used by
    the nuScenes devkit.

    Args:
        radius: Half-width of the excluded square in the xy plane, in the
            point cloud's coordinate units. As a default we use '1.0'.

    Raises:
        ValueError: If ``radius`` is negative cause this would make no sense
            logically but could occur due to mathematical error.
    """

    _transformed_types = (PointCloud3D,)

    def __init__(self, radius: float = 1.0) -> None:
        super().__init__()
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}.")
        self.radius = radius

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Filter close point-cloud returns.

        Returns:
            Point cloud containing only points outside the exclusion region.
        """
        return self._call_kernel(filter_close_points, inpt, radius=self.radius)


class PointShuffle(_RandomApplyTransform):
    """Randomly permute point order with probability ``p``.

    Args:
        p: Probability of applying. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D,)

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample a random permutation.

        Returns:
            Dict with ``"perm"`` key.
        """
        n = flat_inputs[0].shape[0]
        return {"perm": torch.randperm(n)}

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the permutation.

        Returns:
            Shuffled input.
        """
        return self._call_kernel(shuffle_points, inpt, perm=params["perm"])


class PointSample(Transform):
    """Subsample (or oversample with replacement) to exactly ``n`` points.

    If the point cloud has more than ``n`` points, a random subset is
    selected. If fewer, points are sampled with replacement to reach
    ``n``.

    Args:
        n: Target number of points.
    """

    _transformed_types = (PointCloud3D,)

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample indices to reach exactly ``n`` points.

        Returns:
            Dict with ``"indices"`` key.
        """
        num_points = flat_inputs[0].shape[0]
        if num_points >= self.n:
            indices = torch.randperm(num_points)[: self.n]
        else:
            indices = torch.randint(0, num_points, (self.n,))
        return {"indices": indices}

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the sampling.

        Returns:
            Sampled input.
        """
        return self._call_kernel(sample_points, inpt, indices=params["indices"])


class PointJitter(_RandomApplyTransform):
    """Add Gaussian noise to point xyz coordinates with probability ``p``.

    Args:
        sigma: Standard deviation of the Gaussian noise. Default: ``0.01``.
        p: Probability of applying. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D,)

    def __init__(self, sigma: float = 0.01, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.sigma = sigma

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample Gaussian noise.

        Returns:
            Dict with ``"noise"`` key containing ``[N, 3]`` tensor.
        """
        n = flat_inputs[0].shape[0]
        return {"noise": torch.randn(n, 3) * self.sigma}

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the noise.

        Returns:
            Jittered input.
        """
        return self._call_kernel(jitter_points, inpt, noise=params["noise"])
