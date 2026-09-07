"""The contract every vision3d transform must satisfy.

Transforms must:

- Accept any sample structure: a dict, an ``(inputs, targets)`` pair, a list, a
  nested dict, a ``NamedTuple``, or two positional arguments.
- Return the sample in the structure it arrived in.
- Choose what to act on from each leaf's type, never from a key name or a
  position, so that one transform serves every dataset.
- Pass through leaves they do not claim, by identity.
- No-op on a sample that carries none of the types they claim.
- Survive pickling, which ``DataLoader`` requires under its ``spawn`` start
  method.
"""

import math
import pickle
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, override

import pytest
import torch
from common_utils import (
    check_transform,
    make_bounding_boxes_3d,
    make_camera_extrinsics,
    make_camera_images,
    make_camera_intrinsics,
    make_fusion_sample,
    make_lidar_sample,
    make_point_cloud_3d,
)
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import TVTensor

import vision3d.tensors as _tensors
import vision3d.transforms as T
from vision3d.tensors import BoundingBox3DFormat, PointCloud3D
from vision3d.transforms import Transform


def _labels_by_type(batch: Any) -> tuple[Tensor, ...]:
    """Locate the fixtures' labels as the plain (non-TVTensor) tensor leaves.

    The structural containers below hold no label-like key for the default
    heuristic to find, so labels are identified by type instead.

    Returns:
        The plain tensor leaves of ``batch``.
    """
    return tuple(
        leaf
        for leaf in tree_flatten(batch)[0]
        if isinstance(leaf, Tensor) and not isinstance(leaf, TVTensor)
    )


_ALL_FORMATS = (
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWHYPR,
)
# Axis-aligned formats cannot represent a rotation.
_ROTATED_FORMATS = (BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR)


@dataclass(frozen=True)
class Contract:
    """How to build a transform, and which parts of the contract apply to it.

    Attributes:
        build: Constructs the transform.
        cameras: Whether the transform tolerates camera tensors in a sample.
        hooks: Whether ``make_params``/``transform`` are the real per-leaf hooks.
            ``False`` for transforms that hand-roll ``forward``.
        per_leaf_params: Whether ``make_params`` returns entries keyed by leaf
            identity rather than one value shared by every leaf.
        build_p0: Constructs the transform with ``p=0.0``, for transforms gated on
            a probability. ``None`` for transforms with no ``p``.
        formats: Box formats the transform accepts.
    """

    build: Callable[[], Transform]
    cameras: bool = True
    hooks: bool = True
    per_leaf_params: bool = False
    build_p0: Callable[[], Transform] | None = None
    formats: tuple[BoundingBox3DFormat, ...] = _ALL_FORMATS


_WIDE_RANGE = (-100.0, -100.0, -100.0, 100.0, 100.0, 100.0)

CONTRACTS: dict[str, Contract] = {
    "PointShuffle": Contract(
        lambda: T.PointShuffle(p=1.0),
        per_leaf_params=True,
        build_p0=lambda: T.PointShuffle(p=0.0),
    ),
    "PointSample": Contract(lambda: T.PointSample(n=17), per_leaf_params=True),
    "PointJitter": Contract(
        lambda: T.PointJitter(sigma=0.1, p=1.0),
        per_leaf_params=True,
        build_p0=lambda: T.PointJitter(sigma=0.1, p=0.0),
    ),
    # Refuses camera tensors by design, so it is only audited camera-free.
    "RandomFlip3D": Contract(
        lambda: T.RandomFlip3D(axis="x", p=1.0),
        cameras=False,
        build_p0=lambda: T.RandomFlip3D(axis="x", p=0.0),
    ),
    "RandomTranslate3D": Contract(
        lambda: T.RandomTranslate3D(translation_range=1.0, p=1.0),
        build_p0=lambda: T.RandomTranslate3D(translation_range=1.0, p=0.0),
    ),
    "RandomRotate3D": Contract(
        lambda: T.RandomRotate3D(angle_range=math.pi / 4, p=1.0),
        build_p0=lambda: T.RandomRotate3D(angle_range=math.pi / 4, p=0.0),
        formats=_ROTATED_FORMATS,
    ),
    "RandomScale3D": Contract(
        lambda: T.RandomScale3D(scale_range=(1.5, 1.5), p=1.0),
        build_p0=lambda: T.RandomScale3D(scale_range=(1.5, 1.5), p=0.0),
    ),
    "RangeFilter3D": Contract(
        lambda: T.RangeFilter3D(_WIDE_RANGE, labels_getter=None), hooks=False
    ),
    "CopyPaste3D": Contract(
        lambda: T.CopyPaste3D(
            target_counts={0: 3}, min_points=1, labels_getter=_labels_by_type
        ),
        hooks=False,
        build_p0=lambda: T.CopyPaste3D(
            target_counts={0: 3}, min_points=1, labels_getter=_labels_by_type, p=0.0
        ),
    ),
}


def _names(
    *,
    cameras: bool | None = None,
    hooks: bool | None = None,
    per_leaf: bool | None = None,
) -> list[str]:
    """Registered names whose contract matches every supplied flag.

    Returns:
        Transform names in registration order, for use as pytest parameters.
    """
    return [
        name
        for name, contract in CONTRACTS.items()
        if (cameras is None or contract.cameras is cameras)
        and (hooks is None or contract.hooks is hooks)
        and (per_leaf is None or contract.per_leaf_params is per_leaf)
    ]


def _build(name: str) -> Transform:
    """Construct the registered transform.

    Returns:
        A new transform instance.
    """
    return CONTRACTS[name].build()


ALL_TRANSFORMS = _names()
CAMERA_SAFE_TRANSFORMS = _names(cameras=True)
HOOK_DRIVEN_TRANSFORMS = _names(hooks=True)
PER_LEAF_PARAM_TRANSFORMS = _names(per_leaf=True)
SHARED_PARAM_TRANSFORMS = _names(hooks=True, per_leaf=False)
PROBABILISTIC_TRANSFORMS = [n for n, c in CONTRACTS.items() if c.build_p0 is not None]
TRANSFORM_FORMAT_CASES = [
    pytest.param(name, fmt, id=f"{name}-{fmt.value}")
    for name in ALL_TRANSFORMS
    for fmt in CONTRACTS[name].formats
]


Sample = namedtuple("Sample", ["points", "boxes", "labels"])


class TypedSample(NamedTuple):
    points: PointCloud3D
    boxes: Any
    labels: Tensor


def _core_leaves() -> tuple[PointCloud3D, Any, Tensor]:
    """Build one point cloud, one box set, and matching labels.

    Returns:
        ``(points, boxes, labels)``.
    """
    return (
        make_point_cloud_3d(num_points=40),
        make_bounding_boxes_3d(num_boxes=3),
        torch.tensor([0, 1, 2]),
    )


def _structures() -> dict[str, Any]:
    """Place the same three leaves inside a variety of pytree containers.

    Returns:
        Mapping of container description to the assembled sample.
    """
    points, boxes, labels = _core_leaves()
    return {
        "dict": {"points": points, "boxes": boxes, "labels": labels},
        "tuple_of_dicts": ({"points": points}, {"boxes": boxes, "labels": labels}),
        "list": [points, boxes, labels],
        "nested": {
            "inputs": {"points": points},
            "targets": {"boxes": boxes, "labels": labels},
        },
        "namedtuple": Sample(points=points, boxes=boxes, labels=labels),
        "typed_namedtuple": TypedSample(points=points, boxes=boxes, labels=labels),
        "deeply_nested": {"a": [{"b": (points,)}, {"c": [boxes]}], "d": {"e": labels}},
    }


STRUCTURE_NAMES = list(_structures())


class TestRegistryCoverage:
    @pytest.mark.skip_device("cuda")
    def test_every_transform_is_registered(self) -> None:
        exported = {
            name
            for name in T.__all__
            if isinstance(getattr(T, name), type)
            and issubclass(getattr(T, name), Transform)
            and getattr(T, name) is not Transform
        }
        assert exported == set(CONTRACTS), (
            "every transform exported by vision3d.transforms must have a "
            "CONTRACTS entry so the pytree contract is checked automatically"
        )

    def test_fusion_sample_covers_all_tvtensors(self) -> None:
        # The audits below rely on make_fusion_sample holding one of every
        # vision3d TVTensor. A new type missing from it would be passed through
        # silently instead of audited.
        declared = {
            cls
            for cls in vars(_tensors).values()
            if isinstance(cls, type)
            and issubclass(cls, TVTensor)
            and cls is not TVTensor
        }
        present = {
            type(v) for v in make_fusion_sample().values() if isinstance(v, TVTensor)
        }
        missing = declared - present
        assert not missing, (
            f"New TVTensor(s) {sorted(c.__name__ for c in missing)} not in "
            f"make_fusion_sample(). Add to common_utils.make_fusion_sample, "
            f"then audit each transform's _transformed_types / check_inputs."
        )


class TestStructurePreserved:
    """A transform returns the exact container structure it was given."""

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    @pytest.mark.parametrize("structure", STRUCTURE_NAMES)
    def test_structure_round_trips(self, name: str, structure: str) -> None:
        check_transform(_build(name), _structures()[structure])

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    def test_lidar_sample(self, name: str) -> None:
        check_transform(_build(name), make_lidar_sample())

    @pytest.mark.parametrize("name", CAMERA_SAFE_TRANSFORMS)
    def test_fusion_sample(self, name: str) -> None:
        check_transform(_build(name), make_fusion_sample())

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    def test_two_positional_args_match_a_two_tuple(self, name: str) -> None:
        points, boxes, labels = _core_leaves()
        inputs = {"points": points}
        targets = {"boxes": boxes, "labels": labels}

        torch.manual_seed(0)
        splat = _build(name)(inputs, targets)
        torch.manual_seed(0)
        packed = _build(name)((inputs, targets))

        assert tree_flatten(splat)[1] == tree_flatten(packed)[1]
        for a, b in zip(tree_flatten(splat)[0], tree_flatten(packed)[0], strict=True):
            torch.testing.assert_close(a.as_subclass(Tensor), b.as_subclass(Tensor))


class TestPickleable:
    """DataLoader pickles its transforms under the ``spawn`` start method."""

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    @pytest.mark.skip_device("cuda")
    def test_round_trips_through_pickle(self, name: str) -> None:
        transform = _build(name)
        restored = pickle.loads(pickle.dumps(transform))
        assert type(restored) is type(transform)
        # extra_repr lists the public config, so this catches a lost attribute.
        assert repr(restored) == repr(transform)

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    @pytest.mark.skip_device("cuda")
    def test_unpickled_transform_still_runs(self, name: str) -> None:
        restored = pickle.loads(pickle.dumps(_build(name)))
        sample = make_lidar_sample()
        assert set(restored(sample)) == set(sample)


class TestNoMatchingLeaves:
    """A sample holding none of a transform's claimed types is a no-op."""

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    def test_empty_sample(self, name: str) -> None:
        assert _build(name)({}) == {}

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    def test_plain_tensors_only(self, name: str) -> None:
        sample = {"meta": torch.tensor([1.0, 2.0]), "count": torch.tensor(3)}
        out = _build(name)(sample)
        for key, value in sample.items():
            assert out[key] is value

    @pytest.mark.parametrize("name", PER_LEAF_PARAM_TRANSFORMS)
    def test_point_transform_passes_through_camera_only(self, name: str) -> None:
        sample = {
            "images": make_camera_images(num_cameras=2, height=16, width=16),
            "extrinsics": make_camera_extrinsics(num_cameras=2),
            "intrinsics": make_camera_intrinsics(num_cameras=2),
        }
        out = _build(name)(sample)
        assert set(out) == set(sample)
        for key, value in sample.items():
            assert out[key] is value

    @pytest.mark.parametrize("name", PER_LEAF_PARAM_TRANSFORMS)
    def test_point_transform_passes_through_boxes_only(self, name: str) -> None:
        sample = {
            "boxes": make_bounding_boxes_3d(num_boxes=3),
            "labels": torch.tensor([0, 1, 2]),
        }
        out = _build(name)(sample)
        for key, value in sample.items():
            assert out[key] is value


class TestUnclaimedTypesPassThrough:
    """Leaves outside ``_transformed_types`` are returned by identity."""

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    def test_unknown_leaf_types_untouched(self, name: str) -> None:
        points, boxes, labels = _core_leaves()
        marker = object()
        sample = {
            "points": points,
            "boxes": boxes,
            "labels": labels,
            "name": "scene-0001",
            "token": marker,
            "index": 7,
            "flag": None,
        }
        out = _build(name)(sample)
        assert out["name"] == "scene-0001"
        assert out["token"] is marker
        assert out["index"] == 7
        assert out["flag"] is None


class TestForwardRequiresInput:
    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    @pytest.mark.skip_device("cuda")
    def test_no_inputs_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match="at least one input sample"):
            _build(name)()


class TestTransformedTypesDeclared:
    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    @pytest.mark.skip_device("cuda")
    def test_transformed_types_is_a_tuple(self, name: str) -> None:
        transform = _build(name)
        assert isinstance(transform._transformed_types, tuple)
        assert transform._transformed_types


class TestMultiplePointClouds:
    """Samples may hold more than one point cloud, of differing length."""

    def _two_clouds(self, n_a: int = 40, n_b: int = 13) -> dict[str, Any]:
        """Build a sample with two independent point clouds.

        Returns:
            Dict with ``lidar`` and ``radar`` point clouds.
        """
        return {
            "lidar": make_point_cloud_3d(num_points=n_a),
            "radar": make_point_cloud_3d(num_points=n_b),
        }

    @pytest.mark.parametrize("name", PER_LEAF_PARAM_TRANSFORMS)
    def test_differing_lengths_handled_independently(self, name: str) -> None:
        out = _build(name)(self._two_clouds())
        assert isinstance(out["lidar"], PointCloud3D)
        assert isinstance(out["radar"], PointCloud3D)

    @pytest.mark.parametrize("name", SHARED_PARAM_TRANSFORMS)
    def test_point_clouds_stay_aligned(self, name: str) -> None:
        # A scene transform samples its rotation, offset or scale factor once
        # per call and applies that one value to every leaf, so the sensors stay
        # in a common frame.
        cloud = make_point_cloud_3d(num_points=8)
        out = _build(name)({"lidar": cloud.clone(), "radar": cloud.clone()})
        torch.testing.assert_close(
            out["lidar"].as_subclass(Tensor), out["radar"].as_subclass(Tensor)
        )

    def test_point_sample_resamples_each_cloud_to_n(self) -> None:
        out = T.PointSample(n=17)(self._two_clouds(n_a=40, n_b=13))
        assert out["lidar"].shape[0] == 17
        assert out["radar"].shape[0] == 17

    def test_shuffle_draws_independent_permutations(self) -> None:
        torch.manual_seed(0)
        rows = torch.arange(64, dtype=torch.float32).reshape(16, 4)
        out = T.PointShuffle(p=1.0)(
            {"lidar": PointCloud3D(rows.clone()), "radar": PointCloud3D(rows.clone())}
        )
        assert not torch.equal(
            out["lidar"].as_subclass(Tensor), out["radar"].as_subclass(Tensor)
        )

    def test_jitter_draws_independent_noise(self) -> None:
        torch.manual_seed(0)
        base = torch.zeros(16, 4)
        out = T.PointJitter(sigma=1.0, p=1.0)(
            {"lidar": PointCloud3D(base.clone()), "radar": PointCloud3D(base.clone())}
        )
        assert not torch.equal(
            out["lidar"].as_subclass(Tensor), out["radar"].as_subclass(Tensor)
        )


class TestAllRandomnessInMakeParams:
    """``transform`` must be a pure function of ``(inpt, params)``.

    Every random draw belongs in ``make_params``, so ``params`` fully describes
    the augmentation applied. A transform sampling inside ``transform`` would
    give different answers on two calls with the same params.
    """

    @pytest.mark.parametrize("name", HOOK_DRIVEN_TRANSFORMS)
    def test_transform_is_deterministic_given_params(self, name: str) -> None:
        transform = _build(name)
        sample = {
            "lidar": make_point_cloud_3d(num_points=40),
            "radar": make_point_cloud_3d(num_points=13),
            "boxes": make_bounding_boxes_3d(num_boxes=3),
        }
        claimed = self._claimed(transform, sample)
        params = transform.make_params(claimed)

        for inpt in claimed:
            first = transform.transform(inpt, params)
            second = transform.transform(inpt, params)
            torch.testing.assert_close(
                first.as_subclass(Tensor), second.as_subclass(Tensor)
            )

    @pytest.mark.parametrize("name", PER_LEAF_PARAM_TRANSFORMS)
    def test_params_cover_every_claimed_leaf(self, name: str) -> None:
        transform = _build(name)
        sample = {
            "lidar": make_point_cloud_3d(num_points=40),
            "radar": make_point_cloud_3d(num_points=13),
        }
        claimed = self._claimed(transform, sample)
        params = transform.make_params(claimed)

        assert len(params) == 1
        (per_leaf,) = params.values()
        assert set(per_leaf) == {id(inpt) for inpt in claimed}

    @pytest.mark.parametrize("name", PER_LEAF_PARAM_TRANSFORMS)
    def test_make_params_on_empty_leaf_list(self, name: str) -> None:
        # make_params runs even when nothing matched, so it must tolerate an
        # empty list rather than indexing into it.
        params = _build(name).make_params([])

        assert len(params) == 1
        (per_leaf,) = params.values()
        assert per_leaf == {}

    @staticmethod
    def _claimed(transform: Transform, sample: Any) -> list[Any]:
        """Return the sample leaves the transform claims.

        Returns:
            The leaves ``transform`` will be called on.
        """
        flat_inputs, _ = tree_flatten(sample)
        needs = transform._needs_transform_list(flat_inputs)
        return [inpt for inpt, nt in zip(flat_inputs, needs, strict=True) if nt]


class TestBoxFormatPreserved:
    @pytest.mark.parametrize(("name", "fmt"), TRANSFORM_FORMAT_CASES)
    def test_format_survives(self, name: str, fmt: BoundingBox3DFormat) -> None:
        sample = make_lidar_sample(format=fmt)
        out = _build(name)(sample)
        assert out["boxes"].format == fmt


class TestProbabilityGate:
    """A transform built with ``p=0.0`` returns the sample it was given."""

    @pytest.mark.parametrize("name", PROBABILISTIC_TRANSFORMS)
    def test_p_zero_is_identity(self, name: str) -> None:
        build_p0 = CONTRACTS[name].build_p0
        assert build_p0 is not None
        sample = make_lidar_sample()
        out = build_p0()(sample)
        for actual, original in zip(
            tree_flatten(out)[0], tree_flatten(sample)[0], strict=True
        ):
            assert actual is original


class TestInputNotMutated:
    """A transform returns new tensors rather than editing the caller's."""

    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    def test_input_sample_untouched(self, name: str) -> None:
        sample = make_lidar_sample()
        before = [leaf.clone() for leaf in tree_flatten(sample)[0]]
        _build(name)(sample)
        for leaf, original in zip(tree_flatten(sample)[0], before, strict=True):
            torch.testing.assert_close(
                leaf.as_subclass(Tensor), original.as_subclass(Tensor)
            )


class TestDtypePreserved:
    @pytest.mark.parametrize("name", ALL_TRANSFORMS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_survives(self, name: str, dtype: torch.dtype) -> None:
        sample = {
            "points": make_point_cloud_3d(num_points=20, dtype=dtype),
            "boxes": make_bounding_boxes_3d(num_boxes=3, dtype=dtype),
            "labels": torch.tensor([0, 1, 2]),
        }
        out = _build(name)(sample)
        assert out["points"].dtype == dtype
        assert out["boxes"].dtype == dtype


class _IdentityTransform(Transform):
    """Trivial transform that returns inputs unchanged; used to probe dispatch."""

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return inpt


class _PointCloudOnly(_IdentityTransform):
    _transformed_types = (PointCloud3D,)


class TestDispatchByType:
    """``_transformed_types`` decides which leaves reach ``transform``."""

    def test_default_dispatches_all_tvtensors(self) -> None:
        sample = make_fusion_sample()
        out = _IdentityTransform()(sample)
        assert set(out) == set(sample)

    def test_plain_tensors_pass_through(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = _IdentityTransform()({"labels": labels})
        assert out["labels"] is labels

    def test_only_listed_tvtensors_dispatched(self) -> None:
        dispatched: list[type] = []

        class _Probe(_PointCloudOnly):
            @override
            def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
                dispatched.append(type(inpt))
                return inpt

        _Probe()(make_fusion_sample())
        assert dispatched == [PointCloud3D]

    def test_unlisted_tvtensors_pass_through_unchanged(self) -> None:
        sample = make_fusion_sample()
        out = _PointCloudOnly()(sample)
        for key in ("images", "extrinsics", "intrinsics"):
            assert out[key] is sample[key]
