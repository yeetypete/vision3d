"""Tests for the safety-aware torchvision v2 mirror."""

import pytest
import torch
from common_utils import make_camera_images, make_camera_intrinsics, make_fusion_sample
from torch import nn
from torchvision.transforms import v2 as tv_v2

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    PointCloud3D,
)
from vision3d.transforms import v2 as v3d_v2


class TestImagePhotometricSafeAlongside3D:
    """Photometric transforms touch pixels only and must accept every
    vision3d TVTensor type in the sample."""

    def test_color_jitter_on_fusion(self) -> None:
        sample = make_fusion_sample()
        v3d_v2.ColorJitter(brightness=0.3)(sample)

    def test_gaussian_blur_on_fusion(self) -> None:
        sample = make_fusion_sample()
        v3d_v2.GaussianBlur(kernel_size=3)(sample)

    def test_normalize_on_fusion(self) -> None:
        sample = make_fusion_sample()
        sample["images"] = CameraImages(sample["images"].float())
        v3d_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(sample)


class TestImageGeometricSafeAlongside3D:
    """Image-geometric transforms with an existing CameraIntrinsics
    kernel keep image+intrinsics consistent and must accept all
    3D-aware TVTensors."""

    def test_resize_on_fusion_updates_intrinsics(self) -> None:
        sample = make_fusion_sample()
        out = v3d_v2.Resize(size=[16, 16])(sample)
        assert out["images"].shape[-2:] == (16, 16)
        assert out["intrinsics"].image_size == (16, 16)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)
        assert isinstance(out["extrinsics"], CameraExtrinsics)

    def test_center_crop_on_fusion(self) -> None:
        sample = make_fusion_sample()
        out = v3d_v2.CenterCrop(size=[16, 24])(sample)
        assert out["images"].shape[-2:] == (16, 24)
        assert out["intrinsics"].image_size == (16, 24)

    def test_pad_on_fusion(self) -> None:
        sample = make_fusion_sample()
        out = v3d_v2.Pad(padding=2)(sample)
        assert out["images"].shape[-2:] == (36, 36)
        assert out["intrinsics"].image_size == (36, 36)


class TestImageGeometricUnsafeAlongside3D:
    """Flip/rotation transforms have no matching 3D update, so they must
    refuse any 3D-aware TVTensor in the sample."""

    @pytest.mark.parametrize(
        "transform",
        [
            v3d_v2.RandomHorizontalFlip(p=1.0),
            v3d_v2.RandomVerticalFlip(p=1.0),
        ],
    )
    def test_raises_on_fusion(self, transform: nn.Module) -> None:
        sample = make_fusion_sample()
        with pytest.raises(TypeError):
            transform(sample)

    def test_raises_when_only_intrinsics_present(self) -> None:
        # Intrinsics alone are enough to make the flip unsafe because
        # cx wouldn't be updated.
        sample = {
            "images": make_camera_images(num_cameras=2, height=8, width=8),
            "intrinsics": make_camera_intrinsics(num_cameras=2),
        }
        with pytest.raises(TypeError, match="CameraIntrinsics"):
            v3d_v2.RandomHorizontalFlip(p=1.0)(sample)

    def test_passes_on_plain_image_only(self) -> None:
        # No 3D-aware types in the sample, so flipping is acceptable.
        from torchvision.tv_tensors import Image

        sample = {"img": Image(torch.rand(3, 8, 8))}
        out = v3d_v2.RandomHorizontalFlip(p=1.0)(sample)
        assert isinstance(out["img"], Image)

    @pytest.mark.parametrize(
        "transform",
        [
            v3d_v2.RandomHorizontalFlip(p=1.0),
            v3d_v2.RandomVerticalFlip(p=1.0),
        ],
    )
    def test_refuses_camera_images_alone(self, transform: nn.Module) -> None:
        # CameraImages is a vision3d-aware TVTensor, so it must be rejected
        # even when no other 3D tensors are present.
        sample = {"images": make_camera_images(num_cameras=2, height=8, width=8)}
        with pytest.raises(TypeError, match="CameraImages"):
            transform(sample)


class TestBehaviourMatchesTorchvision:
    """Wrapped transforms must produce the same output as their
    torchvision counterpart on a sample they both accept."""

    @staticmethod
    def _image_only_sample() -> tuple[CameraImages, CameraImages]:
        # Same underlying tensor, two CameraImages wrappers, so both
        # branches transform identical input.
        data = torch.rand(2, 3, 16, 16)
        return CameraImages(data.clone()), CameraImages(data.clone())

    def test_color_jitter_identity_matches(self) -> None:
        img_a, img_b = self._image_only_sample()
        # brightness=0.0 is the identity jitter, so both branches must
        # return the input untouched regardless of RNG draws.
        out_a = v3d_v2.ColorJitter(brightness=0.0)({"img": img_a})["img"]
        out_b = tv_v2.ColorJitter(brightness=0.0)({"img": img_b})["img"]
        torch.testing.assert_close(out_a, out_b)

    def test_resize_parity(self) -> None:
        img_a, img_b = self._image_only_sample()
        out_a = v3d_v2.Resize(size=[8, 8])({"img": img_a})["img"]
        out_b = tv_v2.Resize(size=[8, 8])({"img": img_b})["img"]
        torch.testing.assert_close(out_a, out_b)

    def test_subclass_relationship(self) -> None:
        # The wrapped flip classes must remain a subclass of the
        # torchvision original, so framework features (e.g. isinstance
        # checks in downstream code, torch.compile, serialisation) keep
        # working.
        assert issubclass(v3d_v2.RandomHorizontalFlip, tv_v2.RandomHorizontalFlip)
        assert issubclass(v3d_v2.RandomVerticalFlip, tv_v2.RandomVerticalFlip)


class TestComposeInterop:
    def test_v3d_mirror_works_in_torchvision_compose(self) -> None:
        sample = make_fusion_sample()
        chain = tv_v2.Compose(
            [
                v3d_v2.Resize(size=[16, 16]),
                v3d_v2.ColorJitter(brightness=0.1),
            ]
        )
        out = chain(sample)
        assert out["images"].shape[-2:] == (16, 16)

    def test_unsafe_transform_in_compose_raises_at_runtime(self) -> None:
        sample = make_fusion_sample()
        chain = tv_v2.Compose(
            [
                v3d_v2.Resize(size=[16, 16]),
                v3d_v2.RandomHorizontalFlip(p=1.0),
            ]
        )
        with pytest.raises(TypeError):
            chain(sample)
