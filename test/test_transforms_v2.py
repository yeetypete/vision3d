"""Tests for the safety-aware torchvision v2 mirror."""

from collections.abc import Callable

import pytest
import torch
from common_utils import make_camera_images, make_camera_intrinsics, make_fusion_sample
from torch import nn
from torchvision.transforms import v2 as tv_v2
from torchvision.tv_tensors import Image

from vision3d.tensors import (
    CameraImages,
)
from vision3d.transforms import v2 as v3d_v2
from vision3d.transforms.v2 import _REFUSED

#: Default-arg factory for every refused transform. Drives the
#: parametrized refusal tests below; covers transforms that need
#: positional args (``size``, ``degrees``, ``num_classes``) since their
#: defaults are not zero-arg.
_REFUSED_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "AugMix": v3d_v2.AugMix,
    "AutoAugment": v3d_v2.AutoAugment,
    "CutMix": lambda: v3d_v2.CutMix(num_classes=10),
    "ElasticTransform": v3d_v2.ElasticTransform,
    "FiveCrop": lambda: v3d_v2.FiveCrop(size=4),
    "MixUp": lambda: v3d_v2.MixUp(num_classes=10),
    "RandAugment": v3d_v2.RandAugment,
    "RandomAffine": lambda: v3d_v2.RandomAffine(degrees=(-15.0, 15.0)),
    "RandomHorizontalFlip": lambda: v3d_v2.RandomHorizontalFlip(p=1.0),
    "RandomIoUCrop": v3d_v2.RandomIoUCrop,
    "RandomPerspective": lambda: v3d_v2.RandomPerspective(p=1.0),
    "RandomRotation": lambda: v3d_v2.RandomRotation(degrees=(-15.0, 15.0)),
    "RandomVerticalFlip": lambda: v3d_v2.RandomVerticalFlip(p=1.0),
    "TenCrop": lambda: v3d_v2.TenCrop(size=4),
    "TrivialAugmentWide": v3d_v2.TrivialAugmentWide,
}


class TestRefusedSetCoverage:
    def test_factories_cover_refused_set(self) -> None:
        missing = _REFUSED - _REFUSED_FACTORIES.keys()
        extra = _REFUSED_FACTORIES.keys() - _REFUSED
        assert not missing, f"Missing factory for refused transforms: {sorted(missing)}"
        assert not extra, f"Factory for non-refused names: {sorted(extra)}"

    def test_refused_names_exist_on_torchvision(self) -> None:
        # If torchvision ever renames or removes one of these the
        # __getattr__ forwarder would raise AttributeError lazily;
        # surface the mismatch here instead.
        for name in _REFUSED:
            assert hasattr(tv_v2, name), f"torchvision.transforms.v2 lacks {name}"


class TestRefusedAlongside3D:
    """Every transform in ``v3d_v2._REFUSED`` must refuse a sample that
    contains any vision3d-aware TVTensor."""

    @pytest.mark.parametrize("name", sorted(_REFUSED_FACTORIES))
    def test_raises_on_fusion(self, name: str) -> None:
        transform = _REFUSED_FACTORIES[name]()
        sample = make_fusion_sample()
        with pytest.raises(TypeError, match=name):
            transform(sample)

    @pytest.mark.parametrize("name", sorted(_REFUSED_FACTORIES))
    def test_raises_on_camera_images_alone(self, name: str) -> None:
        # CameraImages is a vision3d-aware TVTensor; refusal must fire
        # even when no other 3D tensors are present.
        transform = _REFUSED_FACTORIES[name]()
        sample = {"images": make_camera_images(num_cameras=2, height=8, width=8)}
        with pytest.raises(TypeError, match="CameraImages"):
            transform(sample)

    def test_raises_when_only_intrinsics_present(self) -> None:
        # Intrinsics alone is enough to make flipping unsafe.
        sample = {
            "images": make_camera_images(num_cameras=2, height=8, width=8),
            "intrinsics": make_camera_intrinsics(num_cameras=2),
        }
        with pytest.raises(TypeError, match="CameraIntrinsics"):
            v3d_v2.RandomHorizontalFlip(p=1.0)(sample)

    def test_flip_passes_on_plain_image_only(self) -> None:
        # Without any vision3d TVTensor, the wrapped flip behaves as
        # torchvision's.
        sample = {"img": Image(torch.rand(3, 8, 8))}
        out = v3d_v2.RandomHorizontalFlip(p=1.0)(sample)
        assert isinstance(out["img"], Image)


class TestSafeAlongside3D:
    """Spot-checks that the bare re-exports work on fusion samples."""

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

    def test_resize_on_fusion_updates_intrinsics(self) -> None:
        sample = make_fusion_sample()
        out = v3d_v2.Resize(size=[16, 16])(sample)
        assert out["images"].shape[-2:] == (16, 16)
        assert out["intrinsics"].image_size == (16, 16)

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


class TestBehaviourMatchesTorchvision:
    """Bare re-exports must be the same object as torchvision's; wrapped
    transforms must remain a subclass of the torchvision original so
    framework features (``isinstance``, pickling, ``torch.compile``)
    keep working."""

    @pytest.mark.parametrize(
        "name",
        ["ColorJitter", "GaussianBlur", "Normalize", "Resize", "CenterCrop", "Pad"],
    )
    def test_safe_reexport_is_identity(self, name: str) -> None:
        assert getattr(v3d_v2, name) is getattr(tv_v2, name)

    @pytest.mark.parametrize("name", sorted(_REFUSED))
    def test_wrapped_subclasses_torchvision_original(self, name: str) -> None:
        assert issubclass(getattr(v3d_v2, name), getattr(tv_v2, name))

    @staticmethod
    def _image_only_sample() -> tuple[CameraImages, CameraImages]:
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
