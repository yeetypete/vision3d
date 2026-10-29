"""Tests for grid transforms."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from common_utils import (
    make_fusion_sample,
    make_grid_mask_3d,
    make_lidar_sample,
    make_point_cloud_3d,
)
from torchvision.transforms import v2 as tv_v2

from vision3d.transforms import (
    CopyPaste3D,
    RandomFlip3D,
    RandomRotate3D,
    RandomScale3D,
    RandomTranslate3D,
    ResampleGrid,
)
from vision3d.transforms.functional import resample_grid


class TestResampleGridDispatch:
    def test_dispatches_grid(self) -> None:
        with pytest.raises(NotImplementedError, match="grids"):
            resample_grid(make_grid_mask_3d(), affine=torch.eye(4), size=(8, 8))

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

    def test_passthrough_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        out = ResampleGrid(affine=torch.eye(4), size=(8, 8))({"points": pc})
        assert out["points"] is pc

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
