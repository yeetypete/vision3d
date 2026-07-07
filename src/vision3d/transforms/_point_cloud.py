"""Point cloud transform classes."""

from collections.abc import Mapping, Sequence
from typing import Any, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.tv_tensors import TVTensor

from vision3d.ops import points_in_boxes_3d_indices
from vision3d.tensors import BoundingBoxes3D, PointCloud3D

from ._transform import Transform, _RandomApplyTransform
from .functional._point_cloud import (
    jitter_points,
    sample_points,
    shuffle_points,
)


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


def _normalize_keep(keep: int | tuple[int, int]) -> tuple[int, int]:
    """Expand *keep* into a ``(min, max)`` integer range.

    Args:
        keep: A single count (fixed) or a ``(min, max)`` range.

    Returns:
        The ``(min, max)`` pair.

    Raises:
        TypeError: If *keep* is not an int or a pair of ints.
        ValueError: If the range is negative or has ``min > max``.
    """
    if isinstance(keep, bool):
        msg = "`keep` should be an int or an (int, int) pair."
        raise TypeError(msg)
    if isinstance(keep, int):
        lo, hi = keep, keep
    else:
        seq = tuple(keep)
        if len(seq) != 2 or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in seq
        ):
            msg = "`keep` should be an int or an (int, int) pair."
            raise TypeError(msg)
        lo, hi = seq
    if lo < 0:
        msg = "`keep` values should be non-negative."
        raise ValueError(msg)
    if lo > hi:
        msg = "`keep` min must not exceed max."
        raise ValueError(msg)
    return lo, hi


def _normalize_keep_ratio(
    keep_ratio: float | tuple[float, float],
) -> tuple[float, float]:
    """Expand *keep_ratio* into a ``(min, max)`` float range.

    Args:
        keep_ratio: A single fraction (fixed) or a ``(min, max)`` range.

    Returns:
        The ``(min, max)`` pair.

    Raises:
        TypeError: If *keep_ratio* is not a float or a pair of floats.
        ValueError: If a value is outside ``[0, 1]`` or has ``min > max``.
    """
    if isinstance(keep_ratio, bool):
        msg = "`keep_ratio` should be a float or a (float, float) pair."
        raise TypeError(msg)
    if isinstance(keep_ratio, (int, float)):
        lo, hi = float(keep_ratio), float(keep_ratio)
    else:
        seq = tuple(keep_ratio)
        if len(seq) != 2 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in seq
        ):
            msg = "`keep_ratio` should be a float or a (float, float) pair."
            raise TypeError(msg)
        lo, hi = seq[0], seq[1]
    if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
        msg = "`keep_ratio` values should lie in [0.0, 1.0]."
        raise ValueError(msg)
    if lo > hi:
        msg = "`keep_ratio` min must not exceed max."
        raise ValueError(msg)
    return lo, hi


def _is_label_tensor(obj: Any, num_boxes: int) -> bool:
    """Return whether *obj* looks like a per-box integer label tensor.

    Args:
        obj: Candidate object.
        num_boxes: Expected length.

    Returns:
        True if *obj* is a plain (non-:class:`~torchvision.tv_tensors.TVTensor`)
        1D integer tensor of length ``num_boxes``.
    """
    return (
        isinstance(obj, Tensor)
        and not isinstance(obj, TVTensor)
        and obj.ndim == 1
        and obj.shape[0] == num_boxes
        and not torch.is_floating_point(obj)
        and not torch.is_complex(obj)
    )


def _find_by_key(obj: Any, key: str) -> Any:
    """Recursively search *obj* for the first value stored under *key*.

    Walks mappings and sequences (but not tensors) so that both the
    single-dict (``{"labels": ...}``) and split (``(inputs, targets)``)
    calling conventions are covered.

    Returns:
        The first matching value, or ``None`` if *key* is absent.
    """
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_by_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_by_key(value, key)
            if found is not None:
                return found
    return None


class ObjectPointsSample(_RandomApplyTransform):
    """Thin the points inside each object to simulate sparse LiDAR returns.

    Subsamples the points enclosed by each box down to a per-object target,
    optionally to zero, to probe how well a fusion model copes when an
    object has few or no LiDAR points. Points are assigned to boxes with
    :func:`~vision3d.ops.points_in_boxes_3d_indices` (first box wins on
    overlap); background points, boxes, and labels are left unchanged.

    The target is set by exactly one of ``keep`` or ``keep_ratio``. Each
    accepts a scalar for a fixed target or a ``(min, max)`` range sampled
    per object; as in torchvision a scalar is the exact value, not a
    symmetric range. A target of ``0`` removes all of an object's points; a
    target at or above the current count is a no-op (never oversamples).

    Args:
        keep: Absolute number of points to retain per object, as a fixed
            count or a ``(min, max)`` range. Mutually exclusive with
            ``keep_ratio``. Default: ``None``.
        keep_ratio: Fraction of each object's points to retain
            (``round(ratio * count)``), as a fixed ratio or a ``(min, max)``
            range. Mutually exclusive with ``keep``. Default: ``None``.
        p_object: Per-object probability of thinning. Default: ``1.0``.
        labels: Class labels to thin; ``None`` thins every object. When set,
            an integer per-box label tensor must accompany the boxes: it is
            taken from a ``"labels"`` entry in the input when present,
            otherwise the sole integer per-box tensor. Default: ``None``.
        p: Probability of applying the transform. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D,)

    def __init__(
        self,
        keep: int | tuple[int, int] | None = None,
        keep_ratio: float | tuple[float, float] | None = None,
        p_object: float = 1.0,
        labels: Sequence[int] | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        if (keep is None) == (keep_ratio is None):
            msg = "Exactly one of `keep` or `keep_ratio` must be set."
            raise ValueError(msg)
        if not (0.0 <= p_object <= 1.0):
            msg = "`p_object` should be a float in [0.0, 1.0]."
            raise ValueError(msg)

        self.keep = _normalize_keep(keep) if keep is not None else None
        self.keep_ratio = (
            _normalize_keep_ratio(keep_ratio) if keep_ratio is not None else None
        )
        self.labels = tuple(labels) if labels is not None else None
        self.p_object = p_object

        self._label_set = set(self.labels) if self.labels is not None else None

    def _sample_target(self, count: int, device: torch.device) -> int:
        """Draw the number of points to retain for an object of ``count`` points.

        All draws use ``device`` so the whole transform shares a single RNG
        stream (seeding that device's generator makes it reproducible).

        Returns:
            Target point count, clamped to ``[0, count]``.
        """
        if self.keep is not None:
            lo, hi = self.keep
            target = int(torch.randint(lo, hi + 1, (), device=device).item())
        else:
            assert self.keep_ratio is not None
            lo, hi = self.keep_ratio
            ratio = (
                lo
                if lo == hi
                else float(torch.empty((), device=device).uniform_(lo, hi).item())
            )
            target = round(ratio * count)
        return max(0, min(target, count))

    def _keep_indices(
        self, points: PointCloud3D, boxes: BoundingBoxes3D, labels: Tensor | None
    ) -> Tensor:
        """Compute the global indices of points that survive thinning.

        Background points and points of untouched objects are always kept;
        surviving indices stay in ascending (original) order.

        Returns:
            1D long tensor of indices into ``points``.
        """
        device = points.device
        n = points.shape[0]
        assign = points_in_boxes_3d_indices(points, boxes, boxes.format)  # [N]
        keep_mask = torch.ones(n, dtype=torch.bool, device=device)
        select = torch.rand(boxes.shape[0], device=device) < self.p_object

        for j in range(boxes.shape[0]):
            if not bool(select[j]):
                continue
            if self._label_set is not None and (
                labels is None or int(labels[j].item()) not in self._label_set
            ):
                continue
            idx = (assign == j).nonzero(as_tuple=True)[0]
            count = idx.numel()
            if count == 0:
                continue
            target = self._sample_target(count, device)
            if target >= count:
                continue
            survivors = torch.randperm(count, device=device)[:target]
            drop = torch.ones(count, dtype=torch.bool, device=device)
            drop[survivors] = False
            keep_mask[idx[drop]] = False

        return keep_mask.nonzero(as_tuple=True)[0]

    def _find_labels(
        self, inputs: Any, flat_inputs: list[Any], num_boxes: int
    ) -> Tensor | None:
        """Locate the per-box integer label tensor for the class filter.

        Prefers a value stored under a ``"labels"`` key (searched through any
        mappings in the input) and falls back to the sole integer per-box
        tensor. Non-integer per-box tensors (e.g. float ``scores``) are never
        treated as labels.

        Returns:
            The label tensor, or ``None`` if none is present.
        """
        keyed = _find_by_key(inputs, "labels")
        if _is_label_tensor(keyed, num_boxes):
            return keyed
        return next((o for o in flat_inputs if _is_label_tensor(o, num_boxes)), None)

    @override
    def forward(self, *inputs: Any) -> Any:
        """Thin object points across a single sample.

        Accepts a pytree containing one
        :class:`~vision3d.tensors.PointCloud3D` and one
        :class:`~vision3d.tensors.BoundingBoxes3D`, plus an optional
        integer label tensor. Boxes and labels pass through by identity.

        Returns:
            The input structure with the point cloud subsampled.

        Raises:
            TypeError: If ``labels`` filtering is requested but no matching
                label tensor is present.
        """
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        self.check_inputs(flat_inputs)

        points_idx = next(
            (i for i, o in enumerate(flat_inputs) if isinstance(o, PointCloud3D)), None
        )
        boxes = next((o for o in flat_inputs if isinstance(o, BoundingBoxes3D)), None)

        # Validate labels before the random gate so a missing-labels
        # misconfiguration is reported deterministically, not just when the
        # transform happens to fire.
        labels = None
        if self._label_set is not None and boxes is not None and boxes.shape[0] > 0:
            labels = self._find_labels(inputs, flat_inputs, boxes.shape[0])
            if labels is None:
                msg = (
                    f"{type(self).__name__}() with `labels` set requires an "
                    f"integer label tensor of length {boxes.shape[0]} alongside "
                    f"the boxes."
                )
                raise TypeError(msg)

        if torch.rand(1) >= self.p:
            return inputs

        if points_idx is None or boxes is None or boxes.shape[0] == 0:
            return inputs
        points = flat_inputs[points_idx]

        indices = self._keep_indices(points, boxes, labels)

        flat_outputs = list(flat_inputs)
        flat_outputs[points_idx] = self._call_kernel(
            sample_points, points, indices=indices
        )
        return tree_unflatten(flat_outputs, spec)
