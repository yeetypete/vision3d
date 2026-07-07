"""Point cloud transform classes."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.tv_tensors import TVTensor

from vision3d.ops import points_in_boxes_3d
from vision3d.ops._points_in_boxes_3d import _first_true_index
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


def _normalize_range(
    value: Any,
    name: str,
    *,
    scalar_types: type | tuple[type, ...],
    cast: Callable[[Any], Any],
    type_desc: str,
    minimum: float,
    maximum: float,
    bounds_msg: str,
) -> tuple[Any, Any]:
    """Expand a scalar-or-pair argument into a validated ``(min, max)`` range.

    A scalar becomes a fixed ``(v, v)`` range; a two-element sequence is
    taken verbatim. Shared by the ``keep`` and ``keep_ratio`` arguments.

    Args:
        value: A single value or a ``(min, max)`` pair.
        name: Argument name, used in error messages.
        scalar_types: Accepted scalar type(s) (``bool`` is always rejected).
        cast: Applied to each value (e.g. ``int`` or ``float``).
        type_desc: Human-readable type description for the ``TypeError``.
        minimum: Inclusive lower bound each value must satisfy.
        maximum: Inclusive upper bound each value must satisfy.
        bounds_msg: ``ValueError`` message when a value is out of bounds.

    Returns:
        The validated ``(min, max)`` pair.

    Raises:
        TypeError: If *value* is not a scalar of *scalar_types* or a pair
            thereof.
        ValueError: If a value is out of bounds or has ``min > max``.
    """
    type_msg = f"`{name}` should be {type_desc}."
    if isinstance(value, bool):
        raise TypeError(type_msg)
    if isinstance(value, scalar_types):
        lo = hi = cast(value)
    else:
        seq = tuple(value)
        if len(seq) != 2 or not all(
            isinstance(v, scalar_types) and not isinstance(v, bool) for v in seq
        ):
            raise TypeError(type_msg)
        lo, hi = cast(seq[0]), cast(seq[1])
    if not (minimum <= lo <= maximum and minimum <= hi <= maximum):
        raise ValueError(bounds_msg)
    if lo > hi:
        msg = f"`{name}` min must not exceed max."
        raise ValueError(msg)
    return lo, hi


def _default_labels_getter(inputs: Any) -> Any:
    """Locate the per-box label tensor in a detection-style sample.

    Mirrors torchvision's default ``labels_getter`` heuristic: accepts a
    mapping holding a ``labels``-like key, or an ``(inputs, targets)`` pair
    whose second element is such a mapping or a plain label tensor. A key is
    "label-like" if it equals ``"labels"`` (case-insensitive) or, failing
    that, merely contains ``"label"``.

    Returns:
        The located value, or ``None`` if no label-like entry is present.
    """
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
        inputs = inputs[1]
    if isinstance(inputs, Tensor) and not isinstance(inputs, TVTensor):
        return inputs
    if isinstance(inputs, Mapping):
        keys = [k for k in inputs if isinstance(k, str)]
        for key in keys:
            if key.lower() == "labels":
                return inputs[key]
        for key in keys:
            if "label" in key.lower():
                return inputs[key]
    return None


def _parse_labels_getter(
    labels_getter: str | Callable[[Any], Any] | None,
) -> Callable[[Any], Any]:
    """Resolve the ``labels_getter`` argument into a callable.

    Returns:
        A function mapping the input sample to its label tensor (or ``None``).

    Raises:
        ValueError: If *labels_getter* is not ``"default"``, a callable, or
            ``None``.
    """
    if labels_getter == "default":
        return _default_labels_getter
    if callable(labels_getter):
        return labels_getter
    if labels_getter is None:
        return lambda _: None
    msg = "`labels_getter` should be 'default', a callable, or None."
    raise ValueError(msg)


class ObjectPointsSample(_RandomApplyTransform):
    """Thin the points inside each object to simulate sparse LiDAR returns.

    Subsamples the points enclosed by each box down to a per-object target,
    optionally to zero, to probe how well a fusion model copes when an
    object has few or no LiDAR points. Background points, boxes, and labels
    are left unchanged.

    Following the torchvision single-sample convention (see
    :func:`torchvision.transforms.v2._utils.get_bounding_boxes`), the
    transform operates on the first :class:`~vision3d.tensors.PointCloud3D`
    and the first :class:`~vision3d.tensors.BoundingBoxes3D` in the sample;
    every box in that tensor is considered. Each point is assigned to the
    first *eligible* box that contains it, where a box is eligible if it is
    selected by ``p_object`` and (when ``labels`` is set) has an allowed
    class. Assigning among eligible boxes only means a point in the overlap
    of an ineligible and an eligible box is still thinned with the eligible
    one, rather than escaping thinning via a lower-indexed ineligible box.

    The target is set by exactly one of ``keep`` or ``keep_ratio``. Each
    accepts a scalar for a fixed target or a ``(min, max)`` range sampled
    per object; as in torchvision a scalar is the exact value, not a
    symmetric range. A target of ``0`` removes all of an object's points; a
    target at or above the current count is a no-op (never oversamples).

    All randomness is drawn on the point cloud's device, so seeding that
    device's generator alone makes the transform reproducible.

    Args:
        keep: Absolute number of points to retain per object, as a fixed
            count or a ``(min, max)`` range. Mutually exclusive with
            ``keep_ratio``. Default: ``None``.
        keep_ratio: Fraction of each object's points to retain
            (``round(ratio * count)``), as a fixed ratio or a ``(min, max)``
            range. Mutually exclusive with ``keep``. Default: ``None``.
        p_object: Per-object probability of thinning. Default: ``1.0``.
        labels: Class labels to thin; ``None`` thins every object. When set,
            an integer per-box label tensor must be locatable in the sample
            via ``labels_getter``. Default: ``None``.
        labels_getter: How to find the per-box label tensor when ``labels``
            is set. ``"default"`` looks for a ``labels``-like key in a
            mapping (or the second element of an ``(inputs, targets)`` pair),
            matching torchvision's heuristic; a callable receives the sample
            and returns the tensor; ``None`` disables lookup (invalid when
            ``labels`` is set). Default: ``"default"``.
        p: Probability of applying the transform. Default: ``0.5``.
    """

    _transformed_types = (PointCloud3D,)

    def __init__(
        self,
        keep: int | tuple[int, int] | None = None,
        keep_ratio: float | tuple[float, float] | None = None,
        p_object: float = 1.0,
        labels: Sequence[int] | None = None,
        labels_getter: str | Callable[[Any], Any] | None = "default",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        if (keep is None) == (keep_ratio is None):
            msg = "Exactly one of `keep` or `keep_ratio` must be set."
            raise ValueError(msg)
        if not (0.0 <= p_object <= 1.0):
            msg = "`p_object` should be a float in [0.0, 1.0]."
            raise ValueError(msg)
        if labels is not None and len(labels) == 0:
            msg = "`labels` must be non-empty when set (use None to thin all classes)."
            raise ValueError(msg)
        if labels is not None and labels_getter is None:
            msg = "`labels_getter` must not be None when `labels` is set."
            raise ValueError(msg)

        self.keep = (
            _normalize_range(
                keep,
                "keep",
                scalar_types=int,
                cast=int,
                type_desc="an int or an (int, int) pair",
                minimum=0,
                maximum=float("inf"),
                bounds_msg="`keep` values should be non-negative.",
            )
            if keep is not None
            else None
        )
        self.keep_ratio = (
            _normalize_range(
                keep_ratio,
                "keep_ratio",
                scalar_types=(int, float),
                cast=float,
                type_desc="a float or a (float, float) pair",
                minimum=0.0,
                maximum=1.0,
                bounds_msg="`keep_ratio` values should lie in [0.0, 1.0].",
            )
            if keep_ratio is not None
            else None
        )
        self.labels = tuple(labels) if labels is not None else None
        self.labels_getter = labels_getter
        self.p_object = p_object

        # Cached CPU tensor of the allowed classes; moved to the point cloud's
        # device (and label dtype) once per call rather than rebuilt each time.
        self._wanted_labels = (
            torch.tensor(self.labels, dtype=torch.long)
            if self.labels is not None
            else None
        )
        self._labels_getter = _parse_labels_getter(labels_getter)

    def _sample_targets(self, counts: Tensor, device: torch.device) -> Tensor:
        """Draw the retain-count for each box given its per-box point ``counts``.

        Draws are vectorised over all boxes on ``device`` so no per-object
        host-device synchronisation is needed.

        Returns:
            1D long tensor of per-box targets, each clamped to ``[0, count]``.
        """
        m = counts.shape[0]
        if self.keep is not None:
            lo, hi = self.keep
            targets = torch.randint(lo, hi + 1, (m,), device=device)
        else:
            assert self.keep_ratio is not None
            lo, hi = self.keep_ratio
            if lo == hi:
                ratios = torch.full((m,), lo, device=device)
            else:
                ratios = torch.empty(m, device=device).uniform_(lo, hi)
            targets = torch.round(ratios * counts.to(ratios.dtype)).to(torch.long)
        # Draws are non-negative (randint lo >= 0; round of a non-negative
        # product), so capping at the box's own count is the only bound needed.
        return torch.minimum(targets, counts)

    def _keep_indices(
        self, points: PointCloud3D, boxes: BoundingBoxes3D, labels: Tensor | None
    ) -> Tensor:
        """Compute the global indices of points that survive thinning.

        Background points and points of untouched (ineligible) objects are
        always kept; surviving indices stay in ascending (original) order.

        Returns:
            1D long tensor of indices into ``points``.
        """
        device = points.device
        n = points.shape[0]
        m = boxes.shape[0]

        # Per-box eligibility: selected by p_object and (when filtering) an
        # allowed class. Computed vectorised so nothing below needs per-box
        # host-device syncs.
        eligible = torch.rand(m, device=device) < self.p_object
        if self._wanted_labels is not None:
            assert labels is not None
            wanted = self._wanted_labels.to(device=device, dtype=labels.dtype)
            eligible = eligible & torch.isin(labels.to(device), wanted)
        if not bool(eligible.any()):
            return torch.arange(n, device=device)

        # Assign each point to the first *eligible* box containing it.
        mask = points_in_boxes_3d(points, boxes, boxes.format)  # [N, M]
        assign = _first_true_index(mask & eligible.unsqueeze(0))  # [N], -1 = none
        fg = (assign >= 0).nonzero(as_tuple=True)[0]  # foreground indices, ascending
        if fg.numel() == 0:
            return torch.arange(n, device=device)

        fg_assign = assign[fg]  # box index of each foreground point
        counts = torch.bincount(fg_assign, minlength=m)  # [M]
        targets = self._sample_targets(counts, device)

        # Vectorised subsample: order foreground points by (box, random key)
        # so each box's points are contiguous and in random order, then keep
        # the first `target` of each box via an intra-box rank. Background and
        # untouched objects (target >= count) survive untouched.
        keys = torch.rand(fg.numel(), device=device)
        order = torch.argsort(keys)
        order = order[torch.argsort(fg_assign[order], stable=True)]
        sorted_assign = fg_assign[order]
        starts = torch.zeros(m, dtype=torch.long, device=device)
        starts[1:] = counts.cumsum(0)[:-1]  # offset of each box's block
        rank = torch.arange(fg.numel(), device=device) - starts[sorted_assign]
        drop = fg[order][rank >= targets[sorted_assign]]

        keep_mask = torch.ones(n, dtype=torch.bool, device=device)
        keep_mask[drop] = False
        return keep_mask.nonzero(as_tuple=True)[0]

    def _resolve_labels(self, inputs: Any, num_boxes: int) -> Tensor:
        """Locate and validate the per-box integer label tensor.

        Returns:
            A plain 1D integer tensor of length ``num_boxes``.

        Raises:
            TypeError: If ``labels_getter`` does not yield such a tensor.
        """
        found = self._labels_getter(inputs)
        if (
            isinstance(found, Tensor)
            and not isinstance(found, TVTensor)
            and found.ndim == 1
            and found.shape[0] == num_boxes
            and not torch.is_floating_point(found)
            and not torch.is_complex(found)
        ):
            return found
        msg = (
            f"{type(self).__name__}() with `labels` set requires an integer "
            f"label tensor of length {num_boxes} in the sample (located via "
            f"`labels_getter`)."
        )
        raise TypeError(msg)

    @override
    def forward(self, *inputs: Any) -> Any:
        """Thin object points across a single sample.

        Accepts a pytree containing one
        :class:`~vision3d.tensors.PointCloud3D` and one
        :class:`~vision3d.tensors.BoundingBoxes3D`, plus an optional
        integer label tensor. Boxes and labels pass through by identity.
        When ``labels`` filtering is requested but no matching label tensor
        is found, :meth:`_resolve_labels` raises :class:`TypeError`.

        Returns:
            The input structure with the point cloud subsampled.
        """
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        self.check_inputs(flat_inputs)

        points_idx = next(
            (i for i, o in enumerate(flat_inputs) if isinstance(o, PointCloud3D)), None
        )
        boxes = next((o for o in flat_inputs if isinstance(o, BoundingBoxes3D)), None)
        points = flat_inputs[points_idx] if points_idx is not None else None
        device = points.device if points is not None else torch.device("cpu")

        # Validate labels before the random gate so a missing-labels
        # misconfiguration is reported deterministically, not just when the
        # transform happens to fire.
        labels = None
        if self.labels is not None and boxes is not None and boxes.shape[0] > 0:
            labels = self._resolve_labels(inputs, boxes.shape[0])

        # Gate on the point cloud's device so a single seeded generator drives
        # both the apply decision and the per-object draws.
        if torch.rand(1, device=device) >= self.p:
            return inputs

        if points is None or boxes is None or boxes.shape[0] == 0:
            return inputs
        assert points_idx is not None

        indices = self._keep_indices(points, boxes, labels)
        if indices.numel() == points.shape[0]:
            return inputs  # nothing thinned; skip reallocating the point cloud

        flat_outputs = list(flat_inputs)
        flat_outputs[points_idx] = self._call_kernel(
            sample_points, points, indices=indices
        )
        return tree_unflatten(flat_outputs, spec)
