"""Tests for grid transforms."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torchvision
from common_utils import (
    make_fusion_sample,
    make_grid_mask_3d,
    make_lidar_sample,
    make_point_cloud_3d,
)
from packaging.version import Version
from torchvision.transforms import v2 as tv_v2
from torchvision.transforms.v2 import InterpolationMode

from vision3d.transforms import (
    CopyPaste3D,
    RandomFlip3D,
    RandomRotate3D,
    RandomScale3D,
    RandomTranslate3D,
    ResampleGrid,
)
from vision3d.transforms.functional import resample_grid

_TV_ACCEPTS_INTERPOLATION_STRINGS = Version(torchvision.__version__) >= Version("0.28")


class TestResampleGridDispatch:
    def test_dispatches_grid(self) -> None:
        with pytest.raises(NotImplementedError, match="grids"):
            resample_grid(make_grid_mask_3d(), affine=torch.eye(4), size=(8, 8))

    def test_rejects_unsupported_interpolation(self) -> None:
        with pytest.raises(ValueError, match="only support interpolation"):
            resample_grid(
                make_grid_mask_3d(),
                affine=torch.eye(4),
                size=(8, 8),
                interpolation=InterpolationMode.BICUBIC,
            )

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.bool])
    def test_rejects_bilinear_non_float(self, dtype: torch.dtype) -> None:
        with pytest.raises(ValueError, match="floating point grid"):
            resample_grid(
                make_grid_mask_3d(dtype=dtype, num_classes=2),
                affine=torch.eye(4),
                size=(8, 8),
                interpolation=InterpolationMode.BILINEAR,
            )

    def test_accepts_bilinear_float(self) -> None:
        with pytest.raises(NotImplementedError, match="grids"):
            resample_grid(
                make_grid_mask_3d(dtype=torch.float32),
                affine=torch.eye(4),
                size=(8, 8),
                interpolation=InterpolationMode.BILINEAR,
            )

    def test_rejects_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        with pytest.raises(TypeError, match="supports inputs of type"):
            resample_grid(pc, affine=torch.eye(4), size=(8, 8))

    def test_rejects_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        with pytest.raises(TypeError, match="supports inputs of type"):
            resample_grid(labels, affine=torch.eye(4), size=(8, 8))


class TestResampleGrid:
    def test_not_implemented(self) -> None:
        transform = ResampleGrid(affine=torch.eye(4), size=(8, 8))
        with pytest.raises(NotImplementedError, match="grids"):
            transform({"grid": make_grid_mask_3d()})

    @pytest.mark.parametrize("size", [(8,), (1, 2, 3, 4)])
    def test_rejects_size(self, size: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="2 or 3 elements"):
            ResampleGrid(affine=torch.eye(4), size=size)

    @pytest.mark.parametrize("size", [(0, 8), (-1, 8), (8.5, 8)])
    def test_rejects_non_positive_int_size(self, size: tuple[Any, ...]) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            ResampleGrid(affine=torch.eye(4), size=size)

    def test_rejects_singular_affine(self) -> None:
        affine = torch.eye(4)
        affine[0, 0] = 0.0
        with pytest.raises(ValueError, match="invertible affine transform"):
            ResampleGrid(affine=affine, size=(8, 8))

    def test_passthrough_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        out = ResampleGrid(affine=torch.eye(4), size=(8, 8))({"points": pc})
        assert out["points"] is pc

    @pytest.mark.parametrize(
        ("interpolation", "expected"),
        [
            (InterpolationMode.BILINEAR, InterpolationMode.BILINEAR),
            pytest.param(
                "bilinear",
                InterpolationMode.BILINEAR,
                marks=pytest.mark.skipif(
                    not _TV_ACCEPTS_INTERPOLATION_STRINGS,
                    reason="requires torchvision>=0.28",
                ),
            ),
            (0, InterpolationMode.NEAREST),
        ],
    )
    def test_normalizes_interpolation(
        self, interpolation: Any, expected: InterpolationMode
    ) -> None:
        transform = ResampleGrid(
            affine=torch.eye(4), size=(8, 8), interpolation=interpolation
        )
        assert transform.interpolation == expected

    def test_rejects_invalid_interpolation(self) -> None:
        with pytest.raises(ValueError, match="interpolation"):
            ResampleGrid(affine=torch.eye(4), size=(8, 8), interpolation="linear")

    @pytest.mark.parametrize(
        "interpolation",
        [
            InterpolationMode.NEAREST_EXACT,
            InterpolationMode.BICUBIC,
            InterpolationMode.BOX,
            InterpolationMode.HAMMING,
            InterpolationMode.LANCZOS,
        ],
    )
    def test_rejects_unsupported_interpolation(
        self, interpolation: InterpolationMode
    ) -> None:
        with pytest.raises(ValueError, match="only support interpolation"):
            ResampleGrid(affine=torch.eye(4), size=(8, 8), interpolation=interpolation)

    def test_rejects_invalid_fill(self) -> None:
        fill: Any = "255"
        with pytest.raises(TypeError, match="fill"):
            ResampleGrid(affine=torch.eye(4), size=(8, 8), fill=fill)


# Grid kernels raise until implemented instead of passing grids through.
GEOMETRIC_TRANSFORMS: list[Callable[[], Any]] = [
    lambda: RandomFlip3D(axis="x", p=1.0),
    lambda: RandomTranslate3D(translation_range=1.0, p=1.0),
    lambda: RandomRotate3D(p=1.0),
    lambda: RandomScale3D(scale_range=(0.9, 1.1), p=1.0),
    lambda: tv_v2.RandomHorizontalFlip(p=1.0),
    lambda: tv_v2.RandomVerticalFlip(p=1.0),
]


class TestTransformDispatch:
    @pytest.mark.parametrize("make_transform", GEOMETRIC_TRANSFORMS)
    def test_geometric_transforms_not_implemented(
        self, make_transform: Callable[[], Any]
    ) -> None:
        sample = {"points": make_point_cloud_3d(), "grid": make_grid_mask_3d()}
        with pytest.raises(NotImplementedError, match="grids"):
            make_transform()(sample)

    @pytest.mark.parametrize(
        "make_transform",
        [
            lambda: tv_v2.Resize(size=[16, 16]),
            lambda: tv_v2.CenterCrop(size=[16, 16]),
            lambda: tv_v2.Pad(padding=2),
            lambda: tv_v2.ColorJitter(brightness=0.5),
        ],
    )
    def test_camera_image_transforms_pass_grid_through(
        self, make_transform: Callable[[], Any]
    ) -> None:
        # These act on the camera image plane, and grids live in the ego frame.
        sample = make_fusion_sample()
        mask = make_grid_mask_3d()
        sample["grid"] = mask
        out = make_transform()(sample)
        assert out["grid"] is mask

    def test_copy_paste_rejects_grids(self) -> None:
        sample = make_lidar_sample()
        sample["grid"] = make_grid_mask_3d()
        transform = CopyPaste3D(target_counts={0: 1})
        with pytest.raises(TypeError, match="grids"):
            transform((sample,), ({},))
