"""Tests for CameraImages/CameraIntrinsics kernel registrations on torchvision v2 transforms."""

import pytest
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from vision3d.tensors import CameraImages, CameraIntrinsics


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
        uniform = CameraImages(torch.full((3, 3, 32, 32), 0.5))
        t = v2.ColorJitter(brightness=0.5)
        torch.manual_seed(42)
        output = t(uniform)
        assert torch.equal(output[0], output[1])
        assert torch.equal(output[1], output[2])


# Geometric transforms
@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    K = torch.eye(3).unsqueeze(0).expand(2, -1, -1).clone()
    K[:, 0, 0] = 500.0  # fx
    K[:, 1, 1] = 500.0  # fy
    K[:, 0, 2] = 320.0  # cx
    K[:, 1, 2] = 240.0  # cy
    return CameraIntrinsics(K, image_size=(480, 640))


class TestResizeIntrinsics:
    def test_preserves_type(self, intrinsics: CameraIntrinsics) -> None:
        output = F.resize(intrinsics, size=[240, 320])
        assert isinstance(output, CameraIntrinsics)

    def test_half_size(self, intrinsics: CameraIntrinsics) -> None:
        output = F.resize(intrinsics, size=[240, 320])
        # fx, cx scale by 320/640 = 0.5
        assert output[0, 0, 0].isclose(torch.tensor(250.0))  # fx
        assert output[0, 0, 2].isclose(torch.tensor(160.0))  # cx
        # fy, cy scale by 240/480 = 0.5
        assert output[0, 1, 1].isclose(torch.tensor(250.0))  # fy
        assert output[0, 1, 2].isclose(torch.tensor(120.0))  # cy

    def test_updates_image_size(self, intrinsics: CameraIntrinsics) -> None:
        output = F.resize(intrinsics, size=[240, 320])
        assert isinstance(output, CameraIntrinsics)
        assert output.image_size == (240, 320)

    def test_double_size(self, intrinsics: CameraIntrinsics) -> None:
        output = F.resize(intrinsics, size=[960, 1280])
        assert output[0, 0, 0].isclose(torch.tensor(1000.0))  # fx * 2
        assert output[0, 1, 1].isclose(torch.tensor(1000.0))  # fy * 2

    def test_resize_images_preserves_type(self, camera_images: CameraImages) -> None:
        output = F.resize(camera_images, size=[32, 40])
        assert isinstance(output, CameraImages)
        assert output.shape == (6, 3, 32, 40)

    def test_resize_shorter_edge_int(self, intrinsics: CameraIntrinsics) -> None:
        # 480x640 with shorter edge -> 240: scale = 0.5, longer = 320.
        output = F.resize(intrinsics, size=240)
        assert output.image_size == (240, 320)
        assert output[0, 0, 0].isclose(torch.tensor(250.0))  # fx * 0.5
        assert output[0, 1, 1].isclose(torch.tensor(250.0))  # fy * 0.5

    def test_resize_max_size_only(self, intrinsics: CameraIntrinsics) -> None:
        # size=None + max_size=320: scale longer (640) -> 320, scale = 0.5.
        output = F.resize(intrinsics, size=None, max_size=320)
        assert output.image_size == (240, 320)
        assert output[0, 0, 0].isclose(torch.tensor(250.0))


class TestCropIntrinsics:
    def test_preserves_type(self, intrinsics: CameraIntrinsics) -> None:
        output = F.crop(intrinsics, top=10, left=20, height=100, width=200)
        assert isinstance(output, CameraIntrinsics)

    def test_shifts_principal_point(self, intrinsics: CameraIntrinsics) -> None:
        output = F.crop(intrinsics, top=40, left=20, height=400, width=600)
        assert output[0, 0, 2].isclose(torch.tensor(300.0))  # cx - 20
        assert output[0, 1, 2].isclose(torch.tensor(200.0))  # cy - 40
        # fx, fy unchanged
        assert output[0, 0, 0].isclose(torch.tensor(500.0))
        assert output[0, 1, 1].isclose(torch.tensor(500.0))

    def test_updates_image_size(self, intrinsics: CameraIntrinsics) -> None:
        output = F.crop(intrinsics, top=10, left=20, height=100, width=200)
        assert isinstance(output, CameraIntrinsics)
        assert output.image_size == (100, 200)


class TestPadIntrinsics:
    def test_preserves_type(self, intrinsics: CameraIntrinsics) -> None:
        output = F.pad(intrinsics, padding=[10, 20, 10, 20])
        assert isinstance(output, CameraIntrinsics)

    def test_shifts_principal_point(self, intrinsics: CameraIntrinsics) -> None:
        output = F.pad(intrinsics, padding=[10, 20, 10, 20])
        assert output[0, 0, 2].isclose(torch.tensor(330.0))  # cx + 10
        assert output[0, 1, 2].isclose(torch.tensor(260.0))  # cy + 20
        # fx, fy unchanged
        assert output[0, 0, 0].isclose(torch.tensor(500.0))
        assert output[0, 1, 1].isclose(torch.tensor(500.0))

    def test_updates_image_size(self, intrinsics: CameraIntrinsics) -> None:
        output = F.pad(intrinsics, padding=[10, 20, 10, 20])
        assert isinstance(output, CameraIntrinsics)
        assert output.image_size == (520, 660)  # 480+20+20, 640+10+10

    def test_two_element_padding(self, intrinsics: CameraIntrinsics) -> None:
        output = F.pad(intrinsics, padding=[10, 20])
        assert output[0, 0, 2].isclose(torch.tensor(330.0))  # cx + 10
        assert output[0, 1, 2].isclose(torch.tensor(260.0))  # cy + 20

    def test_int_padding(self, intrinsics: CameraIntrinsics) -> None:
        output = F.pad(intrinsics, padding=15)
        assert output[0, 0, 2].isclose(torch.tensor(335.0))  # cx + 15
        assert output[0, 1, 2].isclose(torch.tensor(255.0))  # cy + 15
        assert output.image_size == (510, 670)  # 480+30, 640+30

    def test_one_element_padding(self, intrinsics: CameraIntrinsics) -> None:
        output = F.pad(intrinsics, padding=[15])
        assert output[0, 0, 2].isclose(torch.tensor(335.0))  # cx + 15
        assert output[0, 1, 2].isclose(torch.tensor(255.0))  # cy + 15
        assert output.image_size == (510, 670)

    def test_invalid_padding_length_raises(self, intrinsics: CameraIntrinsics) -> None:
        with pytest.raises(ValueError, match="1, 2, or 4"):
            F.pad(intrinsics, padding=[1, 2, 3])


class TestCenterCropIntrinsics:
    def test_preserves_type(self, intrinsics: CameraIntrinsics) -> None:
        output = F.center_crop(intrinsics, output_size=[200, 300])
        assert isinstance(output, CameraIntrinsics)

    def test_symmetric_crop(self, intrinsics: CameraIntrinsics) -> None:
        output = F.center_crop(intrinsics, output_size=[240, 320])
        # Crop removes (480-240)/2=120 from top, (640-320)/2=160 from left
        assert output[0, 0, 2].isclose(torch.tensor(160.0))  # cx - 160
        assert output[0, 1, 2].isclose(torch.tensor(120.0))  # cy - 120

    def test_updates_image_size(self, intrinsics: CameraIntrinsics) -> None:
        output = F.center_crop(intrinsics, output_size=[200, 300])
        assert isinstance(output, CameraIntrinsics)
        assert output.image_size == (200, 300)


class TestResizedCropIntrinsics:
    def test_preserves_type(self, intrinsics: CameraIntrinsics) -> None:
        output = F.resized_crop(
            intrinsics, top=0, left=0, height=240, width=320, size=[480, 640]
        )
        assert isinstance(output, CameraIntrinsics)

    def test_crop_then_resize(self, intrinsics: CameraIntrinsics) -> None:
        # Crop top-left 240x320, then resize back to 480x640
        output = F.resized_crop(
            intrinsics, top=0, left=0, height=240, width=320, size=[480, 640]
        )
        # Crop: cx stays 320, cy stays 240 (top=0, left=0)
        # Resize: scale by 640/320=2 and 480/240=2
        assert output[0, 0, 0].isclose(torch.tensor(1000.0))  # fx * 2
        assert output[0, 0, 2].isclose(torch.tensor(640.0))  # cx * 2
        assert output[0, 1, 1].isclose(torch.tensor(1000.0))  # fy * 2
        assert output[0, 1, 2].isclose(torch.tensor(480.0))  # cy * 2

    def test_updates_image_size(self, intrinsics: CameraIntrinsics) -> None:
        output = F.resized_crop(
            intrinsics, top=0, left=0, height=240, width=320, size=[100, 200]
        )
        assert isinstance(output, CameraIntrinsics)
        assert output.image_size == (100, 200)


class TestJointDispatch:
    """End-to-end checks via v2.Transform on a sample containing both
    CameraImages and CameraIntrinsics. These verify that the image kernel
    and the intrinsics kernel stay shape-consistent — if torchvision ever
    changes resize semantics in a way that drifts from our intrinsics
    update, these fail loudly.
    """

    @pytest.fixture
    def sample(self) -> dict[str, CameraImages | CameraIntrinsics]:
        K = torch.eye(3).unsqueeze(0).expand(2, -1, -1).clone()
        K[:, 0, 0] = 500.0
        K[:, 1, 1] = 500.0
        K[:, 0, 2] = 320.0
        K[:, 1, 2] = 240.0
        return {
            "images": CameraImages(torch.rand(2, 3, 480, 640)),
            "intrinsics": CameraIntrinsics(K, image_size=(480, 640)),
        }

    def test_resize(self, sample: dict[str, CameraImages | CameraIntrinsics]) -> None:
        out = v2.Resize(size=[240, 320])(sample)
        assert out["images"].shape[-2:] == out["intrinsics"].image_size

    def test_center_crop(
        self, sample: dict[str, CameraImages | CameraIntrinsics]
    ) -> None:
        out = v2.CenterCrop(size=[200, 300])(sample)
        assert out["images"].shape[-2:] == out["intrinsics"].image_size

    def test_pad(self, sample: dict[str, CameraImages | CameraIntrinsics]) -> None:
        out = v2.Pad(padding=[10, 20, 30, 40])(sample)
        assert out["images"].shape[-2:] == out["intrinsics"].image_size
