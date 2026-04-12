"""Point cloud transform classes."""

from typing import Any, override

import torch
from torch import Tensor

from ._transform import RandomTransform, Transform
from .functional._point_cloud import (
    jitter_points,
    sample_points,
    shuffle_points,
)


class PointShuffle(RandomTransform):
    """Randomly permute point order with probability ``p``.

    Args:
        p: Probability of applying. Default: ``0.5``.
    """

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample a random permutation.

        Returns:
            Dict with ``"perm"`` key.
        """
        n = max(inpt.shape[0] for inpt in flat_inputs if isinstance(inpt, Tensor))
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

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample indices to reach exactly ``n`` points.

        Returns:
            Dict with ``"indices"`` key.
        """
        num_points = max(
            inpt.shape[0] for inpt in flat_inputs if isinstance(inpt, Tensor)
        )
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


class PointJitter(RandomTransform):
    """Add Gaussian noise to point xyz coordinates with probability ``p``.

    Args:
        sigma: Standard deviation of the Gaussian noise. Default: ``0.01``.
        p: Probability of applying. Default: ``0.5``.
    """

    def __init__(self, sigma: float = 0.01, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.sigma = sigma

    @override
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample Gaussian noise.

        Returns:
            Dict with ``"noise"`` key containing ``[N, 3]`` tensor.
        """
        n = max(inpt.shape[0] for inpt in flat_inputs if isinstance(inpt, Tensor))
        return {"noise": torch.randn(n, 3) * self.sigma}

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the noise.

        Returns:
            Jittered input.
        """
        return self._call_kernel(jitter_points, inpt, noise=params["noise"])
