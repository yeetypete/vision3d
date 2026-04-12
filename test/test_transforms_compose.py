"""Pipeline composition tests for vision3d transforms."""

import math
from typing import Any

import torch
from torch import Tensor
from torchvision.transforms import v2

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


def _make_targets() -> dict[str, Any]:
    """Build a targets dict with one box + one label.

    Returns:
        ``{"boxes": BoundingBoxes3D, "labels": Tensor}``.
    """
    return {
        "boxes": BoundingBoxes3D(
            torch.tensor([[1.0, 2, 0, 2, 2, 2, 0.3, 0.1, -0.05]]),
            format=BoundingBox3DFormat.XYZLWHYPR,
        ),
        "labels": torch.tensor([0], dtype=torch.long),
    }


def _make_lidar_inputs() -> dict[str, Any]:
    """Build a lidar-only inputs dict.

    Returns:
        ``{"points": PointCloud3D}``, no camera fields.
    """
    return {"points": PointCloud3D(torch.randn(100, 4))}


def _make_camera_inputs() -> dict[str, Any]:
    """Build a camera-only inputs dict.

    Returns:
        ``{"images": CameraImages, "intrinsics": CameraIntrinsics,
        "extrinsics": CameraExtrinsics}``, no points.
    """
    return {
        "images": CameraImages(torch.rand(2, 3, 16, 24)),
        "intrinsics": CameraIntrinsics(
            torch.eye(3).unsqueeze(0).expand(2, -1, -1).clone(),
            image_size=(16, 24),
        ),
        "extrinsics": CameraExtrinsics(
            torch.eye(4).unsqueeze(0).expand(2, -1, -1).clone()
        ),
    }


def _make_sample() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a fusion sample (every modality present).

    Returns:
        ``(inputs, targets)`` dicts mirroring ``NuScenes3D``'s output.
    """
    inputs = _make_lidar_inputs() | _make_camera_inputs()
    return inputs, _make_targets()


def _standard_chain() -> list[Any]:
    """A representative 4-transform scene pipeline.

    Returns:
        Flip + rotate + scale + translate, deterministic with ``p=1``.
    """
    return [
        RandomFlip3D(axis="x", p=1.0),
        RandomRotate3D(angle_range=math.pi / 8, p=1.0),
        RandomScale3D(scale_range=(0.9, 1.1), p=1.0),
        RandomTranslate3D(translation_range=1.0, p=1.0),
    ]


def _apply_sequential(
    transforms: list[Any],
    inputs: dict[str, Any],
    targets: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for t in transforms:
        inputs, targets = t(inputs, targets)
    return inputs, targets


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        chain = _standard_chain()

        torch.manual_seed(123)
        inputs_a, targets_a = _make_sample()
        inputs_a, targets_a = _apply_sequential(chain, inputs_a, targets_a)

        torch.manual_seed(123)
        inputs_b, targets_b = _make_sample()
        inputs_b, targets_b = _apply_sequential(chain, inputs_b, targets_b)

        assert torch.allclose(
            inputs_a["points"].as_subclass(Tensor),
            inputs_b["points"].as_subclass(Tensor),
        )
        assert torch.allclose(
            targets_a["boxes"].as_subclass(Tensor),
            targets_b["boxes"].as_subclass(Tensor),
        )


class TestTorchvisionComposeCompat:
    """Our ``Transform`` base must work inside :class:`v2.Compose`."""

    def test_v2_compose_with_single_vision3d_transform(self) -> None:
        inputs, targets = _make_sample()
        compose = v2.Compose([RandomFlip3D(axis="z", p=1.0)])

        result = compose(inputs, targets)

        assert isinstance(result, tuple)
        out_inputs, out_targets = result
        assert isinstance(out_inputs["points"], PointCloud3D)
        assert isinstance(out_targets["boxes"], BoundingBoxes3D)

    def test_v2_compose_equals_manual_sequential_under_same_seed(self) -> None:
        chain = _standard_chain()

        torch.manual_seed(7)
        manual_inputs, manual_targets = _make_sample()
        manual_inputs, manual_targets = _apply_sequential(
            chain, manual_inputs, manual_targets
        )

        torch.manual_seed(7)
        compose_inputs, compose_targets = _make_sample()
        compose_inputs, compose_targets = v2.Compose(chain)(
            compose_inputs, compose_targets
        )

        assert torch.allclose(
            manual_inputs["points"].as_subclass(Tensor),
            compose_inputs["points"].as_subclass(Tensor),
        )
        assert torch.allclose(
            manual_targets["boxes"].as_subclass(Tensor),
            compose_targets["boxes"].as_subclass(Tensor),
        )


class TestMixedCompose:
    def test_v2_transform_inside_compose_leaves_targets_intact(self) -> None:
        inputs, targets = _make_sample()
        boxes_before = targets["boxes"].as_subclass(Tensor).clone()

        compose = v2.Compose([v2.ColorJitter(brightness=0.5)])
        _, out_targets = compose(inputs, targets)

        assert torch.allclose(out_targets["boxes"].as_subclass(Tensor), boxes_before)


class TestModalityCompose:
    """Per-modality compose tests — lidar-only, camera-only, fusion.

    Each test builds a sample matching one of the ``SampleInputs``
    modality profiles (:class:`LidarInputs`, :class:`CameraInputs`,
    :class:`FusionInputs`) and verifies the corresponding realistic
    training-style compose runs end-to-end with the expected
    dispatch behavior.
    """

    def test_lidar_only_compose_preserves_points_and_boxes(self) -> None:
        inputs = _make_lidar_inputs()
        targets = _make_targets()
        points_before = inputs["points"].as_subclass(Tensor).clone()

        compose = v2.Compose(
            [
                RandomFlip3D(axis="x", p=1.0),
                RandomRotate3D(angle_range=math.pi / 8, p=1.0),
                RandomScale3D(scale_range=(0.9, 1.1), p=1.0),
                RandomTranslate3D(translation_range=1.0, p=1.0),
            ]
        )

        out_inputs, out_targets = compose(inputs, targets)

        assert isinstance(out_inputs["points"], PointCloud3D)
        assert isinstance(out_targets["boxes"], BoundingBoxes3D)
        assert "images" not in out_inputs
        assert "intrinsics" not in out_inputs
        assert "extrinsics" not in out_inputs
        # Points actually moved through the chain (flip at minimum
        # negates the x axis).
        assert not torch.allclose(
            out_inputs["points"].as_subclass(Tensor), points_before
        )

    def test_camera_only_compose_preserves_images_and_intrinsics(self) -> None:
        inputs = _make_camera_inputs()
        targets = _make_targets()

        compose = v2.Compose(
            [
                RandomFlip3D(axis="x", p=1.0),
                RandomRotate3D(angle_range=math.pi / 8, p=1.0),
                v2.Resize(size=[8, 12]),
                v2.CenterCrop(size=[6, 10]),
                v2.ColorJitter(brightness=0.3),
            ]
        )

        out_inputs, out_targets = compose(inputs, targets)

        assert isinstance(out_inputs["images"], CameraImages)
        assert isinstance(out_inputs["intrinsics"], CameraIntrinsics)
        assert isinstance(out_inputs["extrinsics"], CameraExtrinsics)
        assert isinstance(out_targets["boxes"], BoundingBoxes3D)
        assert "points" not in out_inputs
        # Geometric v2 transforms updated both the image tensor and
        # the paired CameraIntrinsics image_size.
        assert out_inputs["images"].shape[-2:] == (6, 10)
        assert out_inputs["intrinsics"].image_size == (6, 10)

    def test_fusion_compose_preserves_every_tvtensor(self) -> None:
        inputs, targets = _make_sample()

        compose = v2.Compose(
            [
                RandomFlip3D(axis="x", p=1.0),
                RandomRotate3D(angle_range=math.pi / 8, p=1.0),
                RandomScale3D(scale_range=(0.9, 1.1), p=1.0),
                RandomTranslate3D(translation_range=1.0, p=1.0),
                v2.Resize(size=[8, 12]),
                v2.ColorJitter(brightness=0.3),
            ]
        )

        out_inputs, out_targets = compose(inputs, targets)

        assert isinstance(out_inputs["points"], PointCloud3D)
        assert isinstance(out_inputs["images"], CameraImages)
        assert isinstance(out_inputs["intrinsics"], CameraIntrinsics)
        assert isinstance(out_inputs["extrinsics"], CameraExtrinsics)
        assert isinstance(out_targets["boxes"], BoundingBoxes3D)
        assert out_inputs["images"].shape[-2:] == (8, 12)
        assert out_inputs["intrinsics"].image_size == (8, 12)
