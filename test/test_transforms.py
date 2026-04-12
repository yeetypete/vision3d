import math

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
from vision3d.transforms import (
    RandomFlip3D,
    RandomRotate3D,
    RandomScale3D,
    RandomTranslate3D,
)
from vision3d.transforms.functional import (
    flip_3d,
    flip_3d_bounding_boxes,
    flip_3d_point_cloud,
    rotate_3d,
    rotate_3d_bounding_boxes,
    rotate_3d_camera_extrinsics,
    rotate_3d_point_cloud,
    scale_3d,
    scale_3d_bounding_boxes,
    scale_3d_camera_extrinsics,
    scale_3d_point_cloud,
    translate_3d,
    translate_3d_bounding_boxes,
    translate_3d_camera_extrinsics,
    translate_3d_point_cloud,
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

    def test_passthrough_camera_images(self) -> None:
        imgs = make_camera_images(num_cameras=2, height=32, width=32)
        out = flip_3d(imgs, axis="x")
        assert out is imgs

    def test_passthrough_camera_extrinsics(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        out = flip_3d(ext, axis="x")
        assert out is ext

    def test_passthrough_camera_intrinsics(self) -> None:
        intr = make_camera_intrinsics(num_cameras=2)
        out = flip_3d(intr, axis="x")
        assert out is intr

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = flip_3d(labels, axis="x")
        assert out is labels


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
    def test_camera_data_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomFlip3D(axis="x", p=1.0)
        out = transform(sample)
        assert torch.equal(out["images"], sample["images"])
        assert torch.equal(out["extrinsics"], sample["extrinsics"])
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


# Reference implementations
def _reference_translate_point_cloud(
    points: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    out = points.clone()
    out[..., :3] += offset
    return out


def _reference_translate_bounding_boxes(
    boxes: torch.Tensor, format: BoundingBox3DFormat, offset: torch.Tensor
) -> torch.Tensor:
    out = boxes.clone()
    if format is BoundingBox3DFormat.XYZXYZ:
        out[..., :3] += offset
        out[..., 3:6] += offset
    else:
        out[..., :3] += offset
    return out


# Kernel tests
class TestTranslate3DPointCloudKernel:
    @pytest.mark.parametrize(
        "offset", [torch.tensor([1.0, 0, 0]), torch.tensor([0, -2.0, 3.0])]
    )
    def test_correctness(self, offset: torch.Tensor) -> None:
        points = torch.rand(50, 3) * 200 - 100
        actual = translate_3d_point_cloud(points, offset=offset)
        expected = _reference_translate_point_cloud(points, offset)
        torch.testing.assert_close(actual, expected)

    def test_preserves_features(self) -> None:
        points = torch.rand(10, 6)
        offset = torch.tensor([1.0, 2.0, 3.0])
        actual = translate_3d_point_cloud(points, offset=offset)
        torch.testing.assert_close(actual[:, 3:], points[:, 3:])

    def test_does_not_modify_input(self) -> None:
        points = torch.rand(10, 3)
        original = points.clone()
        translate_3d_point_cloud(points, offset=torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(points, original)

    def test_inverse(self) -> None:
        points = torch.rand(10, 3)
        offset = torch.tensor([5.0, -3.0, 1.0])
        roundtripped = translate_3d_point_cloud(
            translate_3d_point_cloud(points, offset=offset), offset=-offset
        )
        torch.testing.assert_close(roundtripped, points)


class TestTranslate3DBoundingBoxesKernel:
    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_correctness(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=5)
        raw = bbox.as_subclass(torch.Tensor)
        offset = torch.tensor([1.0, -2.0, 0.5])
        actual = translate_3d_bounding_boxes(raw, format=format, offset=offset)
        expected = _reference_translate_bounding_boxes(raw, format, offset)
        torch.testing.assert_close(actual, expected)

    def test_does_not_modify_input(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        original = bbox.clone()
        translate_3d_bounding_boxes(
            bbox.as_subclass(torch.Tensor),
            format=BoundingBox3DFormat.XYZLWHYPR,
            offset=torch.tensor([1.0, 2.0, 3.0]),
        )
        torch.testing.assert_close(bbox, original)

    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_inverse(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        raw = bbox.as_subclass(torch.Tensor)
        offset = torch.tensor([5.0, -3.0, 1.0])
        roundtripped = translate_3d_bounding_boxes(
            translate_3d_bounding_boxes(raw, format=format, offset=offset),
            format=format,
            offset=-offset,
        )
        torch.testing.assert_close(roundtripped, raw)

    def test_dimensions_unchanged(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR, num_boxes=3)
        raw = bbox.as_subclass(torch.Tensor)
        offset = torch.tensor([10.0, 20.0, 30.0])
        translated = translate_3d_bounding_boxes(
            raw, format=BoundingBox3DFormat.XYZLWHYPR, offset=offset
        )
        # Dimensions (columns 3-6) and angles (columns 6-9) should be unchanged
        torch.testing.assert_close(translated[:, 3:], raw[:, 3:])


class TestTranslate3DCameraExtrinsicsKernel:
    def test_correctness(self) -> None:
        ext = torch.eye(4).unsqueeze(0)
        offset = torch.tensor([1.0, 2.0, 3.0])
        actual = translate_3d_camera_extrinsics(ext, offset=offset)
        # With identity rotation: E'[:3,3] = E[:3,3] - I @ offset = -offset
        expected = torch.eye(4).unsqueeze(0)
        expected[0, :3, 3] = -offset
        torch.testing.assert_close(actual, expected)

    def test_inverse(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        raw = ext.as_subclass(torch.Tensor)
        offset = torch.tensor([5.0, -3.0, 1.0])
        roundtripped = translate_3d_camera_extrinsics(
            translate_3d_camera_extrinsics(raw, offset=offset), offset=-offset
        )
        torch.testing.assert_close(roundtripped, raw)

    def test_does_not_modify_input(self) -> None:
        ext = make_camera_extrinsics(num_cameras=2)
        original = ext.clone()
        translate_3d_camera_extrinsics(
            ext.as_subclass(torch.Tensor), offset=torch.tensor([1.0, 2.0, 3.0])
        )
        torch.testing.assert_close(ext, original)

    def test_projection_consistent(self) -> None:
        """Verify projection is unchanged after translate."""
        ext = make_camera_extrinsics(num_cameras=1)
        raw_ext = ext.as_subclass(torch.Tensor)[0]  # [4, 4]
        K = torch.tensor([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])

        point_lidar = torch.tensor([10.0, 5.0, 1.0, 1.0])
        p_cam = raw_ext @ point_lidar
        pixel_before = K @ p_cam[:3]
        pixel_before = pixel_before[:2] / pixel_before[2]

        offset = torch.tensor([2.0, -1.0, 0.5])
        ext_translated = translate_3d_camera_extrinsics(
            raw_ext.unsqueeze(0), offset=offset
        )[0]
        point_translated = point_lidar.clone()
        point_translated[:3] += offset
        p_cam_after = ext_translated @ point_translated
        pixel_after = K @ p_cam_after[:3]
        pixel_after = pixel_after[:2] / pixel_after[2]

        torch.testing.assert_close(pixel_before, pixel_after)


# Functional tests
class TestTranslate3DDispatch:
    def test_dispatches_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        offset = torch.tensor([1.0, 0, 0])
        out = translate_3d(pc, offset=offset)
        assert isinstance(out, PointCloud3D)

    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_dispatches_bounding_boxes(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        offset = torch.tensor([1.0, 0, 0])
        out = translate_3d(bbox, offset=offset)
        assert isinstance(out, BoundingBoxes3D)
        assert out.format == format

    def test_dispatches_camera_extrinsics(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        offset = torch.tensor([1.0, 0, 0])
        out = translate_3d(ext, offset=offset)
        assert isinstance(out, CameraExtrinsics)

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = translate_3d(labels, offset=torch.tensor([1.0, 0, 0]))
        assert out is labels


# Transform tests
class TestRandomTranslate3D:
    def test_p_one_always_translates(self) -> None:
        sample = _make_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert not torch.equal(out["points"], sample["points"])

    def test_p_zero_never_translates(self) -> None:
        sample = _make_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=0.0)
        out = transform(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_labels_passthrough(self) -> None:
        sample = _make_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert torch.equal(out["labels"], sample["labels"])

    def test_preserves_types(self) -> None:
        sample = _make_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)

    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_preserves_format(self, format: BoundingBox3DFormat) -> None:
        sample = _make_sample(format=format)
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert out["boxes"].format == format

    def test_per_axis_range(self) -> None:
        transform = RandomTranslate3D(translation_range=(1.0, 2.0, 3.0), p=1.0)
        assert transform.translation_range == (1.0, 2.0, 3.0)


class TestRandomTranslate3DFusion:
    def test_extrinsics_updated(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert not torch.equal(out["extrinsics"], sample["extrinsics"])

    def test_images_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert torch.equal(out["images"], sample["images"])

    def test_intrinsics_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert torch.equal(out["intrinsics"], sample["intrinsics"])

    def test_all_types_preserved(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomTranslate3D(translation_range=5.0, p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)
        assert isinstance(out["images"], CameraImages)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert isinstance(out["intrinsics"], CameraIntrinsics)


Z_AXIS = torch.tensor([0.0, 0.0, 1.0])
X_AXIS = torch.tensor([1.0, 0.0, 0.0])


def _make_z_rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


# Kernel tests
class TestRotate3DPointCloudKernel:
    def test_z_rotation_90deg(self) -> None:
        points = torch.tensor([[1.0, 0, 0]])
        R = _make_z_rotation(math.pi / 2)
        actual = rotate_3d_point_cloud(points, rotation_matrix=R)
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_preserves_features(self) -> None:
        points = torch.rand(10, 6)
        R = _make_z_rotation(0.5)
        actual = rotate_3d_point_cloud(points, rotation_matrix=R)
        torch.testing.assert_close(actual[:, 3:], points[:, 3:])

    def test_does_not_modify_input(self) -> None:
        points = torch.rand(10, 3)
        original = points.clone()
        R = _make_z_rotation(0.3)
        rotate_3d_point_cloud(points, rotation_matrix=R)
        torch.testing.assert_close(points, original)

    def test_inverse(self) -> None:
        points = torch.rand(10, 3)
        R = _make_z_rotation(0.7)
        roundtripped = rotate_3d_point_cloud(
            rotate_3d_point_cloud(points, rotation_matrix=R),
            rotation_matrix=R.T,
        )
        torch.testing.assert_close(roundtripped, points, atol=1e-6, rtol=1e-6)


class TestRotate3DBoundingBoxesKernel:
    @pytest.mark.parametrize(
        "format",
        [BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR],
    )
    def test_center_rotated(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=5)
        raw = bbox.as_subclass(torch.Tensor)
        R = _make_z_rotation(math.pi / 4)
        actual = rotate_3d_bounding_boxes(raw, format=format, rotation_matrix=R)
        expected_centers = (R @ raw[:, :3].unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(
            actual[:, :3], expected_centers, atol=1e-6, rtol=1e-6
        )

    @pytest.mark.parametrize(
        "format",
        [BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZLWH],
    )
    def test_axis_aligned_raises(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=2)
        R = _make_z_rotation(0.3)
        with pytest.raises(NotImplementedError, match="not supported"):
            rotate_3d_bounding_boxes(
                bbox.as_subclass(torch.Tensor), format=format, rotation_matrix=R
            )

    def test_xyzlwhy_non_z_rotation_raises(self) -> None:
        from vision3d.transforms.functional._geometry import _rotation_matrix

        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHY, num_boxes=2)
        R = _rotation_matrix(X_AXIS, 0.3)
        with pytest.raises(ValueError, match="Z-axis"):
            rotate_3d_bounding_boxes(
                bbox.as_subclass(torch.Tensor),
                format=BoundingBox3DFormat.XYZLWHY,
                rotation_matrix=R,
            )

    def test_yaw_updated_for_xyzlwhy(self) -> None:
        boxes = torch.tensor([[5.0, 10, 0, 4, 2, 1.5, 0.0]])
        angle = math.pi / 6
        R = _make_z_rotation(angle)
        actual = rotate_3d_bounding_boxes(
            boxes, format=BoundingBox3DFormat.XYZLWHY, rotation_matrix=R
        )
        torch.testing.assert_close(
            actual[0, 6], torch.tensor(angle), atol=1e-6, rtol=1e-6
        )

    def test_dimensions_unchanged(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR, num_boxes=3)
        raw = bbox.as_subclass(torch.Tensor)
        R = _make_z_rotation(0.5)
        rotated = rotate_3d_bounding_boxes(
            raw, format=BoundingBox3DFormat.XYZLWHYPR, rotation_matrix=R
        )
        torch.testing.assert_close(rotated[:, 3:6], raw[:, 3:6])

    def test_does_not_modify_input(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        original = bbox.clone()
        R = _make_z_rotation(0.3)
        rotate_3d_bounding_boxes(
            bbox.as_subclass(torch.Tensor),
            format=BoundingBox3DFormat.XYZLWHYPR,
            rotation_matrix=R,
        )
        torch.testing.assert_close(bbox, original)

    @pytest.mark.parametrize(
        "format",
        [BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR],
    )
    def test_inverse(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        raw = bbox.as_subclass(torch.Tensor)
        R = _make_z_rotation(0.7)
        roundtripped = rotate_3d_bounding_boxes(
            rotate_3d_bounding_boxes(raw, format=format, rotation_matrix=R),
            format=format,
            rotation_matrix=R.T,
        )
        torch.testing.assert_close(roundtripped, raw, atol=1e-5, rtol=1e-5)


class TestRotate3DCameraExtrinsicsKernel:
    def test_projection_consistent(self) -> None:
        ext = make_camera_extrinsics(num_cameras=1)
        raw_ext = ext.as_subclass(torch.Tensor)[0]
        K = torch.tensor([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])

        point_lidar = torch.tensor([10.0, 5.0, 1.0, 1.0])
        p_cam = raw_ext @ point_lidar
        pixel_before = K @ p_cam[:3]
        pixel_before = pixel_before[:2] / pixel_before[2]

        R = _make_z_rotation(0.3)
        ext_rotated = rotate_3d_camera_extrinsics(
            raw_ext.unsqueeze(0), rotation_matrix=R
        )[0]
        point_rotated = torch.tensor([*(R @ point_lidar[:3]).tolist(), 1.0])
        p_cam_after = ext_rotated @ point_rotated
        pixel_after = K @ p_cam_after[:3]
        pixel_after = pixel_after[:2] / pixel_after[2]

        torch.testing.assert_close(pixel_before, pixel_after, atol=1e-4, rtol=1e-4)

    def test_inverse(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        raw = ext.as_subclass(torch.Tensor)
        R = _make_z_rotation(0.7)
        roundtripped = rotate_3d_camera_extrinsics(
            rotate_3d_camera_extrinsics(raw, rotation_matrix=R),
            rotation_matrix=R.T,
        )
        torch.testing.assert_close(roundtripped, raw, atol=1e-5, rtol=1e-5)

    def test_does_not_modify_input(self) -> None:
        ext = make_camera_extrinsics(num_cameras=2)
        original = ext.clone()
        R = _make_z_rotation(0.3)
        rotate_3d_camera_extrinsics(ext.as_subclass(torch.Tensor), rotation_matrix=R)
        torch.testing.assert_close(ext, original)


# Functional tests
class TestRotate3DDispatch:
    def test_dispatches_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        R = _make_z_rotation(0.5)
        out = rotate_3d(pc, rotation_matrix=R)
        assert isinstance(out, PointCloud3D)

    @pytest.mark.parametrize(
        "format", [BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR]
    )
    def test_dispatches_bounding_boxes(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        R = _make_z_rotation(0.5)
        out = rotate_3d(bbox, rotation_matrix=R)
        assert isinstance(out, BoundingBoxes3D)
        assert out.format == format

    def test_dispatches_camera_extrinsics(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        R = _make_z_rotation(0.5)
        out = rotate_3d(ext, rotation_matrix=R)
        assert isinstance(out, CameraExtrinsics)

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        R = _make_z_rotation(0.5)
        out = rotate_3d(labels, rotation_matrix=R)
        assert out is labels


# Transform tests
class TestRandomRotate3D:
    def test_p_one_always_rotates(self) -> None:
        sample = _make_sample()
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert not torch.equal(out["points"], sample["points"])

    def test_p_zero_never_rotates(self) -> None:
        sample = _make_sample()
        transform = RandomRotate3D(angle_range=0.5, p=0.0)
        out = transform(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_labels_passthrough(self) -> None:
        sample = _make_sample()
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert torch.equal(out["labels"], sample["labels"])

    def test_preserves_types(self) -> None:
        sample = _make_sample()
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)

    @pytest.mark.parametrize(
        "format", [BoundingBox3DFormat.XYZLWHY, BoundingBox3DFormat.XYZLWHYPR]
    )
    def test_preserves_format(self, format: BoundingBox3DFormat) -> None:
        sample = _make_sample(format=format)
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert out["boxes"].format == format

    def test_custom_axis(self) -> None:
        transform = RandomRotate3D(angle_range=0.5, axis=(1.0, 0.0, 0.0), p=1.0)
        assert torch.equal(transform.axis, X_AXIS)


class TestRandomRotate3DFusion:
    def test_extrinsics_updated(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert not torch.equal(out["extrinsics"], sample["extrinsics"])

    def test_images_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert torch.equal(out["images"], sample["images"])

    def test_intrinsics_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert torch.equal(out["intrinsics"], sample["intrinsics"])

    def test_all_types_preserved(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomRotate3D(angle_range=0.5, p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)
        assert isinstance(out["images"], CameraImages)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert isinstance(out["intrinsics"], CameraIntrinsics)


# Kernel tests
class TestScale3DPointCloudKernel:
    def test_correctness(self) -> None:
        points = torch.rand(50, 3) * 200 - 100
        actual = scale_3d_point_cloud(points, factor=2.0)
        expected = points.clone()
        expected[..., :3] *= 2.0
        torch.testing.assert_close(actual, expected)

    def test_preserves_features(self) -> None:
        points = torch.rand(10, 6)
        actual = scale_3d_point_cloud(points, factor=3.0)
        torch.testing.assert_close(actual[:, 3:], points[:, 3:])

    def test_does_not_modify_input(self) -> None:
        points = torch.rand(10, 3)
        original = points.clone()
        scale_3d_point_cloud(points, factor=2.0)
        torch.testing.assert_close(points, original)

    def test_inverse(self) -> None:
        points = torch.rand(10, 3)
        roundtripped = scale_3d_point_cloud(
            scale_3d_point_cloud(points, factor=2.0), factor=0.5
        )
        torch.testing.assert_close(roundtripped, points)


class TestScale3DBoundingBoxesKernel:
    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_correctness(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=5)
        raw = bbox.as_subclass(torch.Tensor)
        actual = scale_3d_bounding_boxes(raw, format=format, factor=2.0)
        expected = raw.clone()
        expected[..., :6] *= 2.0
        torch.testing.assert_close(actual, expected)

    def test_angles_unchanged(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR, num_boxes=3)
        raw = bbox.as_subclass(torch.Tensor)
        scaled = scale_3d_bounding_boxes(
            raw, format=BoundingBox3DFormat.XYZLWHYPR, factor=2.0
        )
        torch.testing.assert_close(scaled[:, 6:], raw[:, 6:])

    def test_does_not_modify_input(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        original = bbox.clone()
        scale_3d_bounding_boxes(
            bbox.as_subclass(torch.Tensor),
            format=BoundingBox3DFormat.XYZLWHYPR,
            factor=2.0,
        )
        torch.testing.assert_close(bbox, original)

    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_inverse(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        raw = bbox.as_subclass(torch.Tensor)
        roundtripped = scale_3d_bounding_boxes(
            scale_3d_bounding_boxes(raw, format=format, factor=2.0),
            format=format,
            factor=0.5,
        )
        torch.testing.assert_close(roundtripped, raw)


class TestScale3DCameraExtrinsicsKernel:
    def test_correctness(self) -> None:
        ext = torch.eye(4).unsqueeze(0)
        ext[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        actual = scale_3d_camera_extrinsics(ext, factor=2.0)
        expected = torch.eye(4).unsqueeze(0)
        expected[0, :3, 3] = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(actual, expected)

    def test_rotation_unchanged(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        raw = ext.as_subclass(torch.Tensor)
        scaled = scale_3d_camera_extrinsics(raw, factor=2.5)
        torch.testing.assert_close(scaled[..., :3, :3], raw[..., :3, :3])

    def test_inverse(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        raw = ext.as_subclass(torch.Tensor)
        roundtripped = scale_3d_camera_extrinsics(
            scale_3d_camera_extrinsics(raw, factor=2.0), factor=0.5
        )
        torch.testing.assert_close(roundtripped, raw)

    def test_does_not_modify_input(self) -> None:
        ext = make_camera_extrinsics(num_cameras=2)
        original = ext.clone()
        scale_3d_camera_extrinsics(ext.as_subclass(torch.Tensor), factor=2.0)
        torch.testing.assert_close(ext, original)

    def test_projection_consistent(self) -> None:
        """Verify projection is unchanged after scaling lidar frame.

        Scaling the world uniformly should not change pixel coordinates:
        the camera point and its depth scale together, and ``K @ p / depth``
        is invariant under that uniform scale.
        """
        ext = make_camera_extrinsics(num_cameras=1)
        raw_ext = ext.as_subclass(torch.Tensor)[0]  # [4, 4]
        K = torch.tensor([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])

        point_lidar = torch.tensor([10.0, 5.0, 1.0, 1.0])
        p_cam = raw_ext @ point_lidar
        pixel_before = K @ p_cam[:3]
        pixel_before = pixel_before[:2] / pixel_before[2]

        factor = 3.0
        ext_scaled = scale_3d_camera_extrinsics(raw_ext.unsqueeze(0), factor=factor)[0]
        point_scaled = point_lidar.clone()
        point_scaled[:3] *= factor
        p_cam_after = ext_scaled @ point_scaled
        pixel_after = K @ p_cam_after[:3]
        pixel_after = pixel_after[:2] / pixel_after[2]

        torch.testing.assert_close(pixel_before, pixel_after)


# Functional tests
class TestScale3DDispatch:
    def test_dispatches_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        out = scale_3d(pc, factor=2.0)
        assert isinstance(out, PointCloud3D)

    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_dispatches_bounding_boxes(self, format: BoundingBox3DFormat) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=3)
        out = scale_3d(bbox, factor=2.0)
        assert isinstance(out, BoundingBoxes3D)
        assert out.format == format

    def test_dispatches_camera_extrinsics(self) -> None:
        ext = make_camera_extrinsics(num_cameras=4)
        out = scale_3d(ext, factor=2.0)
        assert isinstance(out, CameraExtrinsics)

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = scale_3d(labels, factor=2.0)
        assert out is labels


# Transform tests
class TestRandomScale3D:
    def test_p_one_always_scales(self) -> None:
        sample = _make_sample()
        transform = RandomScale3D(scale_range=(0.5, 0.9), p=1.0)
        out = transform(sample)
        assert not torch.equal(out["points"], sample["points"])

    def test_p_zero_never_scales(self) -> None:
        sample = _make_sample()
        transform = RandomScale3D(scale_range=(0.5, 1.5), p=0.0)
        out = transform(sample)
        assert torch.equal(out["points"], sample["points"])

    def test_labels_passthrough(self) -> None:
        sample = _make_sample()
        transform = RandomScale3D(scale_range=(0.8, 1.2), p=1.0)
        out = transform(sample)
        assert torch.equal(out["labels"], sample["labels"])

    def test_preserves_types(self) -> None:
        sample = _make_sample()
        transform = RandomScale3D(scale_range=(0.8, 1.2), p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)

    @pytest.mark.parametrize("format", ALL_FORMATS)
    def test_preserves_format(self, format: BoundingBox3DFormat) -> None:
        sample = _make_sample(format=format)
        transform = RandomScale3D(scale_range=(0.8, 1.2), p=1.0)
        out = transform(sample)
        assert out["boxes"].format == format

    def test_negative_range_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RandomScale3D(scale_range=(-1.0, 1.0))


class TestRandomScale3DFusion:
    def test_extrinsics_updated(self) -> None:
        sample = _make_fusion_sample()
        raw = sample["extrinsics"].as_subclass(torch.Tensor).clone()
        raw[..., :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        original_extrinsics = CameraExtrinsics(raw)
        sample["extrinsics"] = original_extrinsics

        transform = RandomScale3D(scale_range=(0.5, 0.9), p=1.0)
        out = transform(sample)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert not torch.equal(out["extrinsics"], original_extrinsics)

    def test_images_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomScale3D(scale_range=(0.8, 1.2), p=1.0)
        out = transform(sample)
        assert torch.equal(out["images"], sample["images"])

    def test_intrinsics_passthrough(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomScale3D(scale_range=(0.8, 1.2), p=1.0)
        out = transform(sample)
        assert torch.equal(out["intrinsics"], sample["intrinsics"])

    def test_all_types_preserved(self) -> None:
        sample = _make_fusion_sample()
        transform = RandomScale3D(scale_range=(0.8, 1.2), p=1.0)
        out = transform(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)
        assert isinstance(out["images"], CameraImages)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert isinstance(out["intrinsics"], CameraIntrinsics)


class TestPointsAndBoxesStayConsistent:
    """A random transform must apply the same random decision to every
    TVTensor in the sample. If the flip or offset were sampled
    independently per TVTensor, points and boxes would end up in
    different coordinate frames."""

    def test_flip_negates_both_point_x_and_box_center_x(self) -> None:
        sample = _make_sample()
        points_before = sample["points"].clone()
        boxes_before = sample["boxes"].clone()

        out = RandomFlip3D(axis="x", p=1.0)(sample)

        assert torch.allclose(out["points"][:, 0], -points_before[:, 0])
        assert torch.allclose(out["boxes"][:, 0], -boxes_before[:, 0])

    def test_translate_shifts_points_and_boxes_by_same_offset(self) -> None:
        torch.manual_seed(42)
        sample = _make_sample()
        points_before = sample["points"].clone()
        boxes_before = sample["boxes"].clone()

        out = RandomTranslate3D(translation_range=5.0, p=1.0)(sample)

        point_delta = out["points"][0, :3] - points_before[0, :3]
        box_delta = out["boxes"][0, :3] - boxes_before[0, :3]
        assert torch.allclose(point_delta, box_delta, atol=1e-5)
