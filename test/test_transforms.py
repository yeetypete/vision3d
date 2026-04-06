import pytest
import torch
from common_utils import (
    make_bounding_boxes_3d,
    make_camera_extrinsics,
    make_camera_images,
    make_camera_intrinsics,
    make_point_cloud_3d,
)

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)
from vision3d.transforms import RandomFlip3D
from vision3d.transforms.functional import (
    flip_3d,
    flip_3d_bounding_boxes,
    flip_3d_camera_extrinsics,
    flip_3d_point_cloud,
)

ALL_FORMATS = list(BoundingBox3DFormat)
ALL_AXES = ["x", "y", "z"]

# Which YPR angle indices to negate per flip axis (independent reference).
# yaw=6 (around Z), pitch=7 (around Y), roll=8 (around X).
# A flip negates angles rotating around axes OTHER than the flip axis.
_REF_NEGATE_YPR: dict[str, list[int]] = {
    "x": [6, 7],
    "y": [6, 8],
    "z": [7, 8],
}


# Reference implementations
def _reference_flip_point_cloud(points: torch.Tensor, axis: str) -> torch.Tensor:
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    out = points.clone()
    out[..., idx] = -out[..., idx]
    return out


def _reference_flip_bounding_boxes(
    boxes: torch.Tensor, format: BoundingBox3DFormat, axis: str
) -> torch.Tensor:
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    out = boxes.clone()

    if format is BoundingBox3DFormat.XYZXYZ:
        lo, hi = idx, idx + 3
        out[..., lo], out[..., hi] = -boxes[..., hi], -boxes[..., lo]
    else:
        out[..., idx] = -out[..., idx]
        if format is BoundingBox3DFormat.XYZLWHY:
            if axis in ("x", "y"):
                out[..., 6] = -out[..., 6]
        elif format is BoundingBox3DFormat.XYZLWHYPR:
            for angle_idx in _REF_NEGATE_YPR[axis]:
                out[..., angle_idx] = -out[..., angle_idx]

    return out


# Kernel tests
class TestFlip3DPointCloudKernel:
    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_correctness(self, axis: str) -> None:
        points = torch.rand(50, 3) * 200 - 100
        actual = flip_3d_point_cloud(points, axis=axis)
        expected = _reference_flip_point_cloud(points, axis)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_preserves_features(self, axis: str) -> None:
        points = torch.rand(10, 6)
        actual = flip_3d_point_cloud(points, axis=axis)
        torch.testing.assert_close(actual[:, 3:], points[:, 3:])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype: torch.dtype) -> None:
        points = torch.rand(10, 3, dtype=dtype)
        actual = flip_3d_point_cloud(points, axis="x")
        assert actual.dtype == dtype

    def test_does_not_modify_input(self) -> None:
        points = torch.rand(10, 3)
        original = points.clone()
        flip_3d_point_cloud(points, axis="x")
        torch.testing.assert_close(points, original)

    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_double_flip_identity(self, axis: str) -> None:
        points = torch.rand(10, 3)
        double_flipped = flip_3d_point_cloud(
            flip_3d_point_cloud(points, axis=axis), axis=axis
        )
        torch.testing.assert_close(double_flipped, points)

    def test_batch_dims(self) -> None:
        points = torch.rand(2, 10, 4)
        actual = flip_3d_point_cloud(points, axis="y")
        assert actual.shape == (2, 10, 4)
        torch.testing.assert_close(actual[..., 1], -points[..., 1])
        torch.testing.assert_close(actual[..., 0], points[..., 0])


class TestFlip3DBoundingBoxesKernel:
    @pytest.mark.parametrize("axis", ALL_AXES)
    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_correctness(self, format: BoundingBox3DFormat, axis: str) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=5)
        raw = bbox.as_subclass(torch.Tensor)
        actual = flip_3d_bounding_boxes(raw, format=format, axis=axis)
        expected = _reference_flip_bounding_boxes(raw, format, axis)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype: torch.dtype) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR, dtype=dtype)
        actual = flip_3d_bounding_boxes(
            bbox.as_subclass(torch.Tensor),
            format=BoundingBox3DFormat.XYZLWHYPR,
            axis="x",
        )
        assert actual.dtype == dtype

    def test_does_not_modify_input(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        original = bbox.clone()
        flip_3d_bounding_boxes(
            bbox.as_subclass(torch.Tensor),
            format=BoundingBox3DFormat.XYZLWHYPR,
            axis="x",
        )
        torch.testing.assert_close(bbox, original)

    @pytest.mark.parametrize("axis", ALL_AXES)
    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_double_flip_identity(self, format: BoundingBox3DFormat, axis: str) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        raw = bbox.as_subclass(torch.Tensor)
        double_flipped = flip_3d_bounding_boxes(
            flip_3d_bounding_boxes(raw, format=format, axis=axis),
            format=format,
            axis=axis,
        )
        torch.testing.assert_close(double_flipped, raw)

    def test_xyzxyz_flip_swaps_bounds(self) -> None:
        boxes = torch.tensor([[1.0, 2, 3, 10, 20, 30]])
        flipped = flip_3d_bounding_boxes(
            boxes, format=BoundingBox3DFormat.XYZXYZ, axis="x"
        )
        expected = torch.tensor([[-10.0, 2, 3, -1, 20, 30]])
        torch.testing.assert_close(flipped, expected)

    def test_xyzlwhypr_flip_x_negates_yaw_and_pitch(self) -> None:
        boxes = torch.tensor([[5.0, 10, 15, 4, 6, 8, 0.3, 0.5, 0.7]])
        flipped = flip_3d_bounding_boxes(
            boxes, format=BoundingBox3DFormat.XYZLWHYPR, axis="x"
        )
        expected = torch.tensor([[-5.0, 10, 15, 4, 6, 8, -0.3, -0.5, 0.7]])
        torch.testing.assert_close(flipped, expected)

    def test_xyzlwhypr_flip_y_negates_yaw_and_roll(self) -> None:
        boxes = torch.tensor([[5.0, 10, 15, 4, 6, 8, 0.3, 0.5, 0.7]])
        flipped = flip_3d_bounding_boxes(
            boxes, format=BoundingBox3DFormat.XYZLWHYPR, axis="y"
        )
        expected = torch.tensor([[5.0, -10, 15, 4, 6, 8, -0.3, 0.5, -0.7]])
        torch.testing.assert_close(flipped, expected)

    def test_xyzlwhypr_flip_z_negates_pitch_and_roll(self) -> None:
        boxes = torch.tensor([[5.0, 10, 15, 4, 6, 8, 0.3, 0.5, 0.7]])
        flipped = flip_3d_bounding_boxes(
            boxes, format=BoundingBox3DFormat.XYZLWHYPR, axis="z"
        )
        expected = torch.tensor([[5.0, 10, -15, 4, 6, 8, 0.3, -0.5, -0.7]])
        torch.testing.assert_close(flipped, expected)

    def test_xyzlwhy_flip_z_keeps_yaw(self) -> None:
        boxes = torch.tensor([[5.0, 10, 15, 4, 6, 8, 0.3]])
        flipped = flip_3d_bounding_boxes(
            boxes, format=BoundingBox3DFormat.XYZLWHY, axis="z"
        )
        expected = torch.tensor([[5.0, 10, -15, 4, 6, 8, 0.3]])
        torch.testing.assert_close(flipped, expected)


# Functional tests
class TestFlip3DDispatch:
    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_dispatches_point_cloud(self, axis: str) -> None:
        pc = make_point_cloud_3d(num_points=10)
        out = flip_3d(pc, axis=axis)
        assert isinstance(out, PointCloud3D)

    @pytest.mark.parametrize("axis", ALL_AXES)
    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_dispatches_bounding_boxes(
        self, format: BoundingBox3DFormat, axis: str
    ) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        out = flip_3d(bbox, axis=axis)
        assert isinstance(out, BoundingBoxes3D)
        assert out.format == format

    def test_dispatches_camera_extrinsics(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        out = flip_3d(ext, axis="x")
        assert isinstance(out, CameraExtrinsics)

    def test_passthrough_camera_images(self) -> None:
        imgs = make_camera_images(num_cameras=2, height=32, width=32)
        out = flip_3d(imgs, axis="x")
        assert out is imgs

    def test_passthrough_camera_intrinsics(self) -> None:
        intr = make_camera_intrinsics(num_cameras=2)
        out = flip_3d(intr, axis="x")
        assert out is intr

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = flip_3d(labels, axis="x")
        assert out is labels


# Extrinsics kernel tests
class TestFlip3DCameraExtrinsicsKernel:
    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_correctness(self, axis: str) -> None:
        """Flipping negates the column corresponding to the flipped axis."""
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        ext = torch.eye(4).unsqueeze(0).expand(3, -1, -1).clone()
        actual = flip_3d_camera_extrinsics(ext, axis=axis)
        expected = ext.clone()
        expected[..., :, idx].neg_()
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_double_flip_identity(self, axis: str) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        raw = ext.as_subclass(torch.Tensor)
        double_flipped = flip_3d_camera_extrinsics(
            flip_3d_camera_extrinsics(raw, axis=axis), axis=axis
        )
        torch.testing.assert_close(double_flipped, raw)

    def test_does_not_modify_input(self) -> None:
        ext = make_camera_extrinsics(num_cameras=2)
        original = ext.clone()
        flip_3d_camera_extrinsics(ext.as_subclass(torch.Tensor), axis="x")
        torch.testing.assert_close(ext, original)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype: torch.dtype) -> None:
        ext = make_camera_extrinsics(num_cameras=2, dtype=dtype)
        actual = flip_3d_camera_extrinsics(ext.as_subclass(torch.Tensor), axis="y")
        assert actual.dtype == dtype

    def test_non_identity_extrinsics(self) -> None:
        """Verify correctness with a non-trivial rotation+translation matrix."""
        ext = torch.tensor(
            [
                [0.0, -1.0, 0.0, 5.0],
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, -2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).unsqueeze(0)
        # Flip X: negate column 0
        actual = flip_3d_camera_extrinsics(ext, axis="x")
        expected = torch.tensor(
            [
                [0.0, -1.0, 0.0, 5.0],
                [-1.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, -2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).unsqueeze(0)
        torch.testing.assert_close(actual, expected)


# Transform tests
def _make_sample(
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
) -> dict[str, torch.Tensor]:
    return {
        "points": make_point_cloud_3d(num_points=20),
        "boxes": make_bounding_boxes_3d(format=format, num_boxes=3),
        "labels": torch.tensor([0, 1, 2]),
    }


def _make_fusion_sample(
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
) -> dict[str, torch.Tensor]:
    return {
        "points": make_point_cloud_3d(num_points=20),
        "boxes": make_bounding_boxes_3d(format=format, num_boxes=3),
        "labels": torch.tensor([0, 1, 2]),
        "images": make_camera_images(num_cameras=4, height=32, width=32),
        "extrinsics": make_camera_extrinsics(num_cameras=4),
        "intrinsics": make_camera_intrinsics(num_cameras=4),
    }


class TestRandomFlip3D:
    def test_p_one_always_flips(self) -> None:
        sample = _make_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert not torch.equal(out["points"], sample["points"])

    def test_p_zero_never_flips(self) -> None:
        sample = _make_sample()
        transform = RandomFlip3D(axis="x", p=0.0)
        out = transform(sample)
        assert torch.equal(out["points"], sample["points"])
        assert torch.equal(out["boxes"], sample["boxes"])

    def test_labels_passthrough(self) -> None:
        sample = _make_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert torch.equal(out["labels"], sample["labels"])

    def test_preserves_types(self) -> None:
        sample = _make_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)

    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_preserves_format(self, format: BoundingBox3DFormat) -> None:
        sample = _make_sample(format=format)
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert out["boxes"].format == format

    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_point_cloud_correctness_vs_functional(self, axis: str) -> None:
        sample = _make_sample()
        transform = RandomFlip3D(axis=axis, p=1.0)
        out = transform(sample)

        expected = flip_3d_point_cloud(
            sample["points"].as_subclass(torch.Tensor), axis=axis
        )
        torch.testing.assert_close(out["points"].as_subclass(torch.Tensor), expected)

    @pytest.mark.parametrize("axis", ALL_AXES)
    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_bbox_correctness_vs_functional(
        self, format: BoundingBox3DFormat, axis: str
    ) -> None:
        sample = _make_sample(format=format)
        transform = RandomFlip3D(axis=axis, p=1.0)
        out = transform(sample)

        expected = flip_3d_bounding_boxes(
            sample["boxes"].as_subclass(torch.Tensor), format=format, axis=axis
        )
        torch.testing.assert_close(out["boxes"].as_subclass(torch.Tensor), expected)

    def test_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            RandomFlip3D(axis="w")


class TestRandomFlip3DFusion:
    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_extrinsics_updated(self, axis: str) -> None:
        sample = _make_fusion_sample()
        transform = RandomFlip3D(axis=axis, p=1.0)
        out = transform(sample)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert not torch.equal(out["extrinsics"], sample["extrinsics"])

    @pytest.mark.parametrize("axis", ALL_AXES)
    def test_extrinsics_correctness_vs_functional(self, axis: str) -> None:
        sample = _make_fusion_sample()
        transform = RandomFlip3D(axis=axis, p=1.0)
        out = transform(sample)

        expected = flip_3d_camera_extrinsics(
            sample["extrinsics"].as_subclass(torch.Tensor), axis=axis
        )
        torch.testing.assert_close(
            out["extrinsics"].as_subclass(torch.Tensor), expected
        )

    def test_images_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert isinstance(out["images"], CameraImages)
        assert torch.equal(out["images"], sample["images"])

    def test_intrinsics_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert isinstance(out["intrinsics"], CameraIntrinsics)
        assert torch.equal(out["intrinsics"], sample["intrinsics"])

    def test_labels_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert torch.equal(out["labels"], sample["labels"])

    def test_all_types_preserved(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)
        assert isinstance(out["images"], CameraImages)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert isinstance(out["intrinsics"], CameraIntrinsics)
