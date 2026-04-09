"""Tests for CameraImages kernel registrations on torchvision v2 transforms."""

import pytest
import torch
from torchvision.transforms import v2

from vision3d.tensors import CameraImages


@pytest.fixture
def camera_images() -> CameraImages:
    return CameraImages(torch.rand(6, 3, 64, 80))


@pytest.fixture
def camera_images_uint8() -> CameraImages:
    return CameraImages(torch.randint(0, 256, (6, 3, 64, 80), dtype=torch.uint8))


class TestColorTransforms:
    @pytest.mark.parametrize(
        "transform",
        [
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            v2.RandomAutocontrast(p=1.0),
            v2.RandomInvert(p=1.0),
            v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
            v2.RandomSolarize(threshold=0.5, p=1.0),
            v2.GaussianBlur(kernel_size=3),
            v2.GaussianNoise(sigma=0.1),
        ],
    )
    def test_preserves_type(
        self, transform: v2.Transform, camera_images: CameraImages
    ) -> None:
        output = transform(camera_images)
        assert isinstance(output, CameraImages)

    @pytest.mark.parametrize(
        "transform",
        [
            v2.ColorJitter(brightness=0.5),
            v2.GaussianBlur(kernel_size=3),
            v2.GaussianNoise(sigma=0.1),
            v2.RandomAutocontrast(p=1.0),
            v2.RandomInvert(p=1.0),
        ],
    )
    def test_preserves_shape(
        self, transform: v2.Transform, camera_images: CameraImages
    ) -> None:
        output = transform(camera_images)
        assert output.shape == camera_images.shape

    @pytest.mark.parametrize(
        "transform",
        [
            v2.RandomEqualize(p=1.0),
            v2.RandomPosterize(bits=4, p=1.0),
        ],
    )
    def test_uint8_transforms(
        self, transform: v2.Transform, camera_images_uint8: CameraImages
    ) -> None:
        output = transform(camera_images_uint8)
        assert isinstance(output, CameraImages)
        assert output.shape == camera_images_uint8.shape


class TestNormalize:
    def test_preserves_type(self, camera_images: CameraImages) -> None:
        t = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        output = t(camera_images)
        assert isinstance(output, CameraImages)

    def test_preserves_shape(self, camera_images: CameraImages) -> None:
        t = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        output = t(camera_images)
        assert output.shape == camera_images.shape

    def test_values_normalized(self, camera_images: CameraImages) -> None:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        t = v2.Normalize(mean=mean, std=std)
        output = t(camera_images)
        # (x - 0.5) / 0.5 maps [0,1] -> [-1,1]
        assert output.min() >= -1.0 - 1e-5
        assert output.max() <= 1.0 + 1e-5


class TestToDtype:
    def test_float_to_uint8(self, camera_images: CameraImages) -> None:
        from torchvision.transforms.v2.functional import to_dtype

        output = to_dtype(camera_images, torch.uint8, scale=True)
        assert isinstance(output, CameraImages)
        assert output.dtype == torch.uint8

    def test_uint8_to_float(self, camera_images_uint8: CameraImages) -> None:
        from torchvision.transforms.v2.functional import to_dtype

        output = to_dtype(camera_images_uint8, torch.float32, scale=True)
        assert isinstance(output, CameraImages)
        assert output.dtype == torch.float32
        assert output.max() <= 1.0


class TestGrayscale:
    def test_rgb_to_grayscale(self, camera_images: CameraImages) -> None:
        from torchvision.transforms.v2.functional import rgb_to_grayscale

        output = rgb_to_grayscale(camera_images, num_output_channels=1)
        assert isinstance(output, CameraImages)
        assert output.shape == (6, 1, 64, 80)


class TestSameParamsAllCameras:
    def test_color_jitter_same_across_cameras(self) -> None:
        # All cameras have the same pixel values — after jitter they should
        # still all be identical (same params applied to all)
        uniform = CameraImages(torch.full((3, 3, 32, 32), 0.5))
        t = v2.ColorJitter(brightness=0.5)
        torch.manual_seed(42)
        output = t(uniform)
        # Each camera should have the same values
        assert torch.equal(output[0], output[1])
        assert torch.equal(output[1], output[2])
