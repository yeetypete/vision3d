"""Tests for grid transforms."""

from typing import Any

import pytest
import torch
from common_utils import make_grid_mask_3d, make_point_cloud_3d

from vision3d.transforms import ResampleGrid
from vision3d.transforms.functional import resample_grid


class TestResampleGridDispatch:
    def test_passthrough_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        out = resample_grid(pc, affine=torch.eye(4), size=(8, 8))
        assert out is pc

    def test_passthrough_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        out = resample_grid(labels, affine=torch.eye(4), size=(8, 8))
        assert out is labels


class TestResampleGrid:
    def test_not_implemented(self) -> None:
        transform = ResampleGrid(affine=torch.eye(4), size=(8, 8))
        with pytest.raises(NotImplementedError, match="grids"):
            transform({"grid": make_grid_mask_3d()})

    @pytest.mark.parametrize("size", [(8,), (1, 2, 3, 4)])
    def test_rejects_size(self, size: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="2 or 3 elements"):
            ResampleGrid(affine=torch.eye(4), size=size)

    def test_rejects_invalid_fill(self) -> None:
        fill: Any = "255"
        with pytest.raises(TypeError, match="fill"):
            ResampleGrid(affine=torch.eye(4), size=(8, 8), fill=fill)
