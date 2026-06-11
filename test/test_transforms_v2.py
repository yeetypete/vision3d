"""Tests for the safety-aware torchvision v2 mirror."""

from collections.abc import Callable

import pytest
import torch
from common_utils import make_camera_images, make_fusion_sample
from torch import nn
from torchvision.transforms import v2 as tv_v2
from torchvision.tv_tensors import Image

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)
from vision3d.transforms import v2 as v3d_v2
from vision3d.transforms.v2 import _REFUSED

#: Default-arg factory for every refused transform. Drives the
#: parametrized refusal tests below.
_REFUSED_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "AugMix": v3d_v2.AugMix,
    "AutoAugment": v3d_v2.AutoAugment,
    "CutMix": lambda: v3d_v2.CutMix(num_classes=10),
    "ElasticTransform": v3d_v2.ElasticTransform,
    "FiveCrop": lambda: v3d_v2.FiveCrop(size=4),
    "MixUp": lambda: v3d_v2.MixUp(num_classes=10),
    "RandAugment": v3d_v2.RandAugment,
    "RandomAffine": lambda: v3d_v2.RandomAffine(degrees=(-15.0, 15.0)),
    "RandomIoUCrop": v3d_v2.RandomIoUCrop,
    "RandomPerspective": lambda: v3d_v2.RandomPerspective(p=1.0),
    "RandomRotation": lambda: v3d_v2.RandomRotation(degrees=(-15.0, 15.0)),
    "TenCrop": lambda: v3d_v2.TenCrop(size=4),
    "TrivialAugmentWide": v3d_v2.TrivialAugmentWide,
}


class TestRefusedSetCoverage:
    """Guards against drift between :data:`_REFUSED`, the local factory
    table, and torchvision's public surface.
    """

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

    def test_plain_image_still_works(self) -> None:
        # Without any vision3d TVTensor, every refused transform forwards
        # to torchvision unchanged on plain Image inputs.
        sample = {"img": Image(torch.rand(3, 64, 64))}
        v3d_v2.ElasticTransform()(sample)


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


class TestFlipDispatchOnFusion:
    """``v2.RandomHorizontalFlip`` (= world Y) and ``v2.RandomVerticalFlip``
    (= world Z) must flip every modality of a fusion sample through the
    registered kernels."""

    def test_horizontal_flip_negates_point_cloud_y(self) -> None:
        sample = make_fusion_sample()
        original = sample["points"].as_subclass(torch.Tensor).clone()
        out = v3d_v2.RandomHorizontalFlip(p=1.0)(sample)
        expected = original.clone()
        expected[..., 1] = -expected[..., 1]
        torch.testing.assert_close(out["points"].as_subclass(torch.Tensor), expected)

    def test_vertical_flip_negates_point_cloud_z(self) -> None:
        sample = make_fusion_sample()
        original = sample["points"].as_subclass(torch.Tensor).clone()
        out = v3d_v2.RandomVerticalFlip(p=1.0)(sample)
        expected = original.clone()
        expected[..., 2] = -expected[..., 2]
        torch.testing.assert_close(out["points"].as_subclass(torch.Tensor), expected)

    def test_horizontal_flip_updates_intrinsics_cx(self) -> None:
        sample = make_fusion_sample(image_size=(48, 60))
        original_cx = sample["intrinsics"][..., 0, 2].clone()
        out = v3d_v2.RandomHorizontalFlip(p=1.0)(sample)
        assert out["intrinsics"].image_size == (48, 60)
        torch.testing.assert_close(out["intrinsics"][..., 0, 2], 60 - original_cx)

    def test_vertical_flip_updates_intrinsics_cy(self) -> None:
        sample = make_fusion_sample(image_size=(48, 60))
        original_cy = sample["intrinsics"][..., 1, 2].clone()
        out = v3d_v2.RandomVerticalFlip(p=1.0)(sample)
        assert out["intrinsics"].image_size == (48, 60)
        torch.testing.assert_close(out["intrinsics"][..., 1, 2], 48 - original_cy)

    def test_horizontal_flip_extrinsics_remain_rigid(self) -> None:
        sample = make_fusion_sample()
        out = v3d_v2.RandomHorizontalFlip(p=1.0)(sample)
        R = out["extrinsics"][..., :3, :3].as_subclass(torch.Tensor)
        eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        torch.testing.assert_close(R @ R.transpose(-1, -2), eye, atol=1e-5, rtol=1e-5)
        det = torch.linalg.det(R)
        torch.testing.assert_close(det, torch.ones_like(det), atol=1e-5, rtol=1e-5)

    def test_horizontal_flip_images_horizontally(self) -> None:
        sample = make_fusion_sample()
        original = sample["images"].as_subclass(torch.Tensor).clone()
        out = v3d_v2.RandomHorizontalFlip(p=1.0)(sample)
        torch.testing.assert_close(
            out["images"].as_subclass(torch.Tensor),
            torch.flip(original, dims=[-1]),
        )

    def test_vertical_flip_images_vertically(self) -> None:
        sample = make_fusion_sample()
        original = sample["images"].as_subclass(torch.Tensor).clone()
        out = v3d_v2.RandomVerticalFlip(p=1.0)(sample)
        torch.testing.assert_close(
            out["images"].as_subclass(torch.Tensor),
            torch.flip(original, dims=[-2]),
        )

    def test_types_preserved_for_horizontal_flip(self) -> None:
        sample = make_fusion_sample()
        out = v3d_v2.RandomHorizontalFlip(p=1.0)(sample)
        assert isinstance(out["points"], PointCloud3D)
        assert isinstance(out["boxes"], BoundingBoxes3D)
        assert isinstance(out["images"], CameraImages)
        assert isinstance(out["extrinsics"], CameraExtrinsics)
        assert isinstance(out["intrinsics"], CameraIntrinsics)

    @pytest.mark.parametrize(
        ("transform_factory", "world_axis"),
        [
            (lambda: v3d_v2.RandomHorizontalFlip(p=1.0), 1),  # world Y
            (lambda: v3d_v2.RandomVerticalFlip(p=1.0), 2),  # world Z
        ],
    )
    def test_projection_consistency_full_cloud(
        self,
        transform_factory: Callable[[], nn.Module],
        world_axis: int,
    ) -> None:
        """Project every lidar point through every camera, before and
        after the flip. After H-flip, every pixel mirrors as
        ``u --> W - u``; after V-flip, ``v --> H - v``.
        """
        from vision3d.ops import project_to_image

        torch.manual_seed(0)
        num_cameras = 4
        h, w = 48, 64
        sample = make_fusion_sample(
            num_cameras=num_cameras, image_size=(h, w), num_points=200
        )
        # Replace identity extrinsics with random non-trivial rotations.
        ext = torch.eye(4).expand(num_cameras, -1, -1).clone()
        for c in range(num_cameras):
            angles = torch.rand(3) * 3.14
            cx_a, sx_a = float(angles[0].cos()), float(angles[0].sin())
            cy_a, sy_a = float(angles[1].cos()), float(angles[1].sin())
            cz_a, sz_a = float(angles[2].cos()), float(angles[2].sin())
            Rx = torch.tensor([[1, 0, 0], [0, cx_a, -sx_a], [0, sx_a, cx_a]])
            Ry = torch.tensor([[cy_a, 0, sy_a], [0, 1, 0], [-sy_a, 0, cy_a]])
            Rz = torch.tensor([[cz_a, -sz_a, 0], [sz_a, cz_a, 0], [0, 0, 1]])
            ext[c, :3, :3] = Rz @ Ry @ Rx
            ext[c, :3, 3] = (torch.rand(3) - 0.5) * 4.0
        sample["extrinsics"] = CameraExtrinsics(ext)

        orig_pts = sample["points"].as_subclass(torch.Tensor)[:, :3].clone()
        orig_ext = sample["extrinsics"].as_subclass(torch.Tensor).clone()
        orig_intr = sample["intrinsics"].as_subclass(torch.Tensor).clone()

        out = transform_factory()(sample)

        new_pts = out["points"].as_subclass(torch.Tensor)[:, :3]
        new_ext = out["extrinsics"].as_subclass(torch.Tensor)
        new_intr = out["intrinsics"].as_subclass(torch.Tensor)

        # Sanity check that the point cloud was flipped on the right axis.
        expected_pts = orig_pts.clone()
        expected_pts[..., world_axis] = -expected_pts[..., world_axis]
        torch.testing.assert_close(new_pts, expected_pts)

        for c in range(num_cameras):
            uv_orig, depth_orig = project_to_image(orig_pts, orig_ext[c], orig_intr[c])
            uv_new, depth_new = project_to_image(new_pts, new_ext[c], new_intr[c])
            in_front = (depth_orig > 0) & (depth_new > 0)
            assert in_front.any(), f"camera {c}: no points in front of camera"
            torch.testing.assert_close(
                depth_new[in_front], depth_orig[in_front], rtol=1e-5, atol=1e-4
            )
            if world_axis == 1:  # H-flip --> u mirrors, v unchanged
                expected_u = w - uv_orig[in_front, 0]
                expected_v = uv_orig[in_front, 1]
            else:  # V-flip --> u unchanged, v mirrors
                expected_u = uv_orig[in_front, 0]
                expected_v = h - uv_orig[in_front, 1]
            torch.testing.assert_close(
                uv_new[in_front, 0], expected_u, rtol=1e-4, atol=1e-3
            )
            torch.testing.assert_close(
                uv_new[in_front, 1], expected_v, rtol=1e-4, atol=1e-3
            )


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
                v3d_v2.ElasticTransform(),
            ]
        )
        with pytest.raises(TypeError):
            chain(sample)
