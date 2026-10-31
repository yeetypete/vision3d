"""Tests for point cloud transforms."""

import functools

import torch
from common_utils import (
    check_transform,
    make_bounding_boxes_3d,
    make_fusion_sample,
    make_lidar_sample,
    make_point_cloud_3d,
)

from vision3d.tensors import BoundingBox3DFormat, PointCloud3D
from vision3d.transforms import PointJitter, PointSample, PointShuffle
from vision3d.transforms.functional import (
    jitter_points,
    jitter_points_point_cloud,
    sample_points,
    sample_points_point_cloud,
    shuffle_points,
    shuffle_points_point_cloud,
)

_make_sample = functools.partial(make_lidar_sample, num_points=100)


class TestShufflePointsKernel:
    def test_output_is_permutation(self) -> None:
        points = torch.randn(50, 4)
        out = shuffle_points_point_cloud(points)
        assert out.shape == points.shape
        assert torch.equal(out.sort(dim=0).values, points.sort(dim=0).values)

    def test_does_not_modify_input(self) -> None:
        points = torch.randn(50, 4)
        original = points.clone()
        shuffle_points_point_cloud(points)
        assert torch.equal(points, original)

    def test_dtype_preserved(self) -> None:
        points = torch.randn(20, 4, dtype=torch.float64)
        out = shuffle_points_point_cloud(points)
        assert out.dtype == torch.float64


class TestSamplePointsKernel:
    def test_downsample(self) -> None:
        points = torch.randn(100, 4)
        out = sample_points_point_cloud(points, n=30)
        assert out.shape == (30, 4)

    def test_oversample(self) -> None:
        points = torch.randn(10, 4)
        out = sample_points_point_cloud(points, n=50)
        assert out.shape == (50, 4)

    def test_exact_size_is_permutation(self) -> None:
        points = torch.randn(50, 4)
        out = sample_points_point_cloud(points, n=50)
        assert torch.equal(out.sort(dim=0).values, points.sort(dim=0).values)

    def test_does_not_modify_input(self) -> None:
        points = torch.randn(50, 4)
        original = points.clone()
        sample_points_point_cloud(points, n=10)
        assert torch.equal(points, original)

    def test_dtype_preserved(self) -> None:
        points = torch.randn(20, 4, dtype=torch.float64)
        out = sample_points_point_cloud(points, n=5)
        assert out.dtype == torch.float64


class TestJitterPointsKernel:
    def test_only_xyz_modified(self) -> None:
        points = torch.randn(50, 6)
        out = jitter_points_point_cloud(points, sigma=0.1)
        assert torch.equal(out[:, 3:], points[:, 3:])
        assert not torch.equal(out[:, :3], points[:, :3])

    def test_zero_sigma_is_identity(self) -> None:
        points = torch.randn(50, 4)
        out = jitter_points_point_cloud(points, sigma=0.0)
        assert torch.equal(out, points)

    def test_does_not_modify_input(self) -> None:
        points = torch.randn(50, 4)
        original = points.clone()
        jitter_points_point_cloud(points, sigma=0.1)
        assert torch.equal(points, original)

    def test_dtype_preserved(self) -> None:
        points = torch.randn(20, 4, dtype=torch.float64)
        out = jitter_points_point_cloud(points, sigma=0.1)
        assert out.dtype == torch.float64


class TestShufflePointsDispatch:
    def test_dispatches_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=20)
        out = shuffle_points(pc)
        assert isinstance(out, PointCloud3D)

    def test_passthrough_bounding_boxes(self) -> None:
        boxes = make_bounding_boxes_3d(
            format=BoundingBox3DFormat.XYZLWHYPR, num_boxes=3
        )
        out = shuffle_points(boxes)
        assert out is boxes

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = shuffle_points(labels)
        assert out is labels


class TestSamplePointsDispatch:
    def test_dispatches_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=20)
        out = sample_points(pc, n=10)
        assert isinstance(out, PointCloud3D)

    def test_passthrough_bounding_boxes(self) -> None:
        boxes = make_bounding_boxes_3d(
            format=BoundingBox3DFormat.XYZLWHYPR, num_boxes=3
        )
        out = sample_points(boxes, n=10)
        assert out is boxes

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = sample_points(labels, n=10)
        assert out is labels


class TestJitterPointsDispatch:
    def test_dispatches_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=20)
        out = jitter_points(pc, sigma=0.1)
        assert isinstance(out, PointCloud3D)

    def test_passthrough_bounding_boxes(self) -> None:
        boxes = make_bounding_boxes_3d(
            format=BoundingBox3DFormat.XYZLWHYPR, num_boxes=3
        )
        out = jitter_points(boxes, sigma=0.1)
        assert out is boxes

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = jitter_points(labels, sigma=0.1)
        assert out is labels


class TestPointShuffle:
    def test_transform(self) -> None:
        check_transform(PointShuffle(p=1.0), make_fusion_sample())

    def test_output_same_shape(self) -> None:
        sample = _make_sample()
        out = PointShuffle(p=1.0)(sample)
        assert out["points"].shape == sample["points"].shape

    def test_output_is_permutation(self) -> None:
        sample = _make_sample()
        out = PointShuffle(p=1.0)(sample)
        original_sorted = sample["points"].sort(dim=0).values
        shuffled_sorted = out["points"].sort(dim=0).values
        assert torch.allclose(original_sorted, shuffled_sorted)

    def test_p_zero_is_identity(self) -> None:
        sample = _make_sample()
        out = PointShuffle(p=0.0)(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_clouds_of_different_sizes(self) -> None:
        sample = {
            "lidar": make_point_cloud_3d(num_points=100),
            "radar": make_point_cloud_3d(num_points=60),
        }
        out = PointShuffle(p=1.0)(sample)
        assert out["lidar"].shape == sample["lidar"].shape
        assert out["radar"].shape == sample["radar"].shape


class TestPointSample:
    def test_transform(self) -> None:
        check_transform(PointSample(n=10), make_fusion_sample())

    def test_downsample(self) -> None:
        sample = _make_sample()
        out = PointSample(n=10)(sample)
        assert out["points"].shape[0] == 10
        assert out["points"].shape[1] == sample["points"].shape[1]

    def test_upsample_with_replacement(self) -> None:
        sample = _make_sample()
        out = PointSample(n=200)(sample)
        assert out["points"].shape[0] == 200

    def test_exact_size_is_permutation(self) -> None:
        sample = _make_sample()
        out = PointSample(n=100)(sample)
        assert out["points"].shape[0] == 100
        original_sorted = sample["points"].sort(dim=0).values
        sampled_sorted = out["points"].sort(dim=0).values
        assert torch.allclose(original_sorted, sampled_sorted)

    def test_clouds_of_different_sizes(self) -> None:
        sample = {
            "lidar": make_point_cloud_3d(num_points=100),
            "radar": make_point_cloud_3d(num_points=60),
        }
        out = PointSample(n=80)(sample)
        assert out["lidar"].shape[0] == 80
        assert out["radar"].shape[0] == 80


class TestPointJitter:
    def test_transform(self) -> None:
        check_transform(PointJitter(sigma=0.1, p=1.0), make_fusion_sample())

    def test_output_same_shape(self) -> None:
        sample = _make_sample()
        out = PointJitter(sigma=0.1, p=1.0)(sample)
        assert out["points"].shape == sample["points"].shape

    def test_only_xyz_modified(self) -> None:
        sample = _make_sample()
        out = PointJitter(sigma=0.1, p=1.0)(sample)
        assert torch.equal(out["points"][:, 3:], sample["points"][:, 3:])

    def test_xyz_actually_changed(self) -> None:
        sample = _make_sample()
        out = PointJitter(sigma=1.0, p=1.0)(sample)
        assert not torch.equal(out["points"][:, :3], sample["points"][:, :3])

    def test_noise_magnitude_scales_with_sigma(self) -> None:
        torch.manual_seed(0)
        sample = _make_sample()
        out_small = PointJitter(sigma=0.001, p=1.0)(sample)

        torch.manual_seed(0)
        sample2 = _make_sample()
        out_large = PointJitter(sigma=1.0, p=1.0)(sample2)

        diff_small = (out_small["points"][:, :3] - sample["points"][:, :3]).abs().mean()
        diff_large = (
            (out_large["points"][:, :3] - sample2["points"][:, :3]).abs().mean()
        )
        assert diff_large > diff_small * 10

    def test_p_zero_is_identity(self) -> None:
        sample = _make_sample()
        out = PointJitter(sigma=0.1, p=0.0)(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_clouds_of_different_sizes(self) -> None:
        sample = {
            "lidar": make_point_cloud_3d(num_points=100),
            "radar": make_point_cloud_3d(num_points=60),
        }
        out = PointJitter(sigma=0.1, p=1.0)(sample)
        assert out["lidar"].shape == sample["lidar"].shape
        assert out["radar"].shape == sample["radar"].shape
