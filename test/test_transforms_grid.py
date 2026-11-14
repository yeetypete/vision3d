"""Tests for grid transforms."""

import itertools
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torchvision
from common_utils import (
    make_fusion_sample,
    make_grid_affine,
    make_grid_mask_3d,
    make_lidar_sample,
    make_point_cloud_3d,
)
from packaging.version import Version
from torch import Tensor
from torchvision.transforms import v2 as tv_v2
from torchvision.transforms.v2 import InterpolationMode

from vision3d.tensors import GridMask3D
from vision3d.transforms import (
    CopyPaste3D,
    RandomFlip3D,
    RandomRotate3D,
    RandomScale3D,
    RandomTranslate3D,
    ResampleGrid,
)
from vision3d.transforms.functional import resample_grid, resample_grid_mask_3d

_TV_ACCEPTS_INTERPOLATION_STRINGS = Version(torchvision.__version__) >= Version("0.28")


def _shift(affine: Tensor, d: float = 0.0, h: float = 0.0, w: float = 0.0) -> Tensor:
    """Move a grid layout by whole or fractional cells along its own axes.

    Returns:
        The shifted ``[4, 4]`` affine transform, on the device of ``affine``.
    """
    shift = torch.eye(4, dtype=torch.float64, device=affine.device)
    shift[:3, 3] = torch.tensor([d, h, w], dtype=torch.float64, device=affine.device)
    return affine @ shift


def _resample(
    mask: GridMask3D, affine: Tensor, size: tuple[int, ...], **kwargs: Any
) -> Tensor:
    return resample_grid_mask_3d(
        mask.as_subclass(Tensor),
        input_affine=mask.affine,
        affine=affine,
        size=size,
        **kwargs,
    )


class TestResampleGridDispatch:
    def test_dispatches_grid(self) -> None:
        mask = make_grid_mask_3d()
        affine = _shift(mask.affine, h=1)
        out = resample_grid(mask, affine=affine, size=(4, 5))
        assert isinstance(out, GridMask3D)
        assert out.shape == (1, 4, 5)
        torch.testing.assert_close(out.affine, affine)

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
        out = resample_grid(
            make_grid_mask_3d(dtype=torch.float32),
            affine=torch.eye(4),
            size=(8, 8),
            interpolation=InterpolationMode.BILINEAR,
        )
        assert out.dtype == torch.float32

    def test_rejects_point_cloud(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        with pytest.raises(TypeError, match="supports inputs of type"):
            resample_grid(pc, affine=torch.eye(4), size=(8, 8))

    def test_rejects_plain_tensor(self) -> None:
        labels = torch.tensor([0, 1, 2])
        with pytest.raises(TypeError, match="supports inputs of type"):
            resample_grid(labels, affine=torch.eye(4), size=(8, 8))


class TestResampleGridMask3D:
    @pytest.mark.parametrize("spatial_shape", [(6, 7), (4, 6, 7)])
    @pytest.mark.parametrize(
        ("dtype", "interpolation"),
        [
            (torch.bool, InterpolationMode.NEAREST),
            (torch.uint8, InterpolationMode.NEAREST),
            (torch.int64, InterpolationMode.NEAREST),
            (torch.float32, InterpolationMode.NEAREST),
            (torch.float16, InterpolationMode.BILINEAR),
            (torch.float32, InterpolationMode.BILINEAR),
            (torch.float64, InterpolationMode.BILINEAR),
        ],
    )
    def test_identity(
        self,
        spatial_shape: tuple[int, ...],
        dtype: torch.dtype,
        interpolation: InterpolationMode,
    ) -> None:
        mask = make_grid_mask_3d(
            channels=2, spatial_shape=spatial_shape, num_classes=2, dtype=dtype
        )
        out = _resample(mask, mask.affine, spatial_shape, interpolation=interpolation)
        assert out.dtype == dtype
        assert out.device == mask.device
        torch.testing.assert_close(out, mask.as_subclass(Tensor))

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("cells", [1, -2])
    @pytest.mark.parametrize(
        ("dtype", "interpolation"),
        [
            (torch.int64, InterpolationMode.NEAREST),
            (torch.float32, InterpolationMode.BILINEAR),
        ],
    )
    def test_translate_whole_cells(
        self,
        axis: int,
        cells: int,
        dtype: torch.dtype,
        interpolation: InterpolationMode,
    ) -> None:
        # Target cell i reads input cell i + cells.
        mask = make_grid_mask_3d(channels=2, spatial_shape=(4, 6, 7), dtype=dtype)
        offset = [0.0, 0.0, 0.0]
        offset[axis] = cells
        out = _resample(
            mask,
            _shift(mask.affine, *offset),
            (4, 6, 7),
            interpolation=interpolation,
            fill=[100, 200],
        )

        data = mask.as_subclass(Tensor)
        fill = torch.tensor([100, 200], dtype=dtype).view(2, 1, 1, 1)
        expected = fill.expand_as(data).clone()
        n = data.shape[axis + 1]
        target = [slice(None)] * 4
        source = [slice(None)] * 4
        if cells > 0:
            target[axis + 1], source[axis + 1] = slice(0, n - cells), slice(cells, n)
        else:
            target[axis + 1], source[axis + 1] = slice(-cells, n), slice(0, n + cells)
        expected[tuple(target)] = data[tuple(source)]
        torch.testing.assert_close(out, expected)

    def test_rotate_quarter_turn(self) -> None:
        mask = make_grid_mask_3d(spatial_shape=(2, 4, 4))
        quarter_turn = mask.affine.new_tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 3], [0, 0, 0, 1]]
        )
        out = _resample(mask, mask.affine @ quarter_turn, (2, 4, 4))
        torch.testing.assert_close(
            out, torch.rot90(mask.as_subclass(Tensor), 1, dims=(-2, -1))
        )

    def test_rotate_eighth_turn(self) -> None:
        # 45 degrees about the center cell moves the outer ring by one step.
        mask = GridMask3D(torch.arange(9).view(1, 3, 3), affine=make_grid_affine())
        c = math.cos(math.pi / 4)
        rotate = mask.affine.new_tensor(
            [[1, 0, 0, 0], [0, c, -c, 1], [0, c, c, 1 - 2 * c], [0, 0, 0, 1]]
        )
        out = _resample(mask, mask.affine @ rotate, (3, 3))
        assert out[0].tolist() == [[3, 0, 1], [6, 4, 2], [7, 8, 5]]

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_reflect_matches_flip(self, axis: int) -> None:
        mask = make_grid_mask_3d(spatial_shape=(4, 6, 7))
        n = mask.shape[axis + 1]
        reflect = torch.eye(4, dtype=torch.float64, device=mask.affine.device)
        reflect[axis, axis], reflect[axis, 3] = -1.0, n - 1.0
        out = _resample(mask, mask.affine @ reflect, (4, 6, 7))
        torch.testing.assert_close(out, mask.as_subclass(Tensor).flip(axis + 1))

    def test_upsample_repeats_cells(self) -> None:
        # Half-size cells with the same outer edges as the input.
        mask = make_grid_mask_3d(spatial_shape=(3, 4, 5))
        half = torch.diag(mask.affine.new_tensor([0.5, 0.5, 0.5, 1.0]))
        affine = _shift(mask.affine @ half, -0.5, -0.5, -0.5)
        out = _resample(mask, affine, (6, 8, 10))
        data = mask.as_subclass(Tensor)
        expected = data.repeat_interleave(2, 1).repeat_interleave(2, 2)
        torch.testing.assert_close(out, expected.repeat_interleave(2, 3))

    def test_nearest_cells_are_half_open(self) -> None:
        # Cell k covers [k - 0.5, k + 0.5).
        mask = GridMask3D(torch.arange(4).view(1, 1, 4), affine=make_grid_affine())
        out = _resample(mask, _shift(mask.affine, w=0.5), (1, 4), fill=255)
        assert out.flatten().tolist() == [1, 2, 3, 255]
        out = _resample(mask, _shift(mask.affine, w=-0.5), (1, 4), fill=255)
        assert out.flatten().tolist() == [0, 1, 2, 3]

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("cell_size", [0.4, 0.3, 0.15])
    def test_nearest_ties_read_upper_cell(self, axis: int, cell_size: float) -> None:
        # 2x downsampling puts every target center on an input cell boundary.
        # These cell sizes aren't exact in binary.
        n = 200
        shape = [2, 2, 2]
        shape[axis] = n
        steps = torch.arange(n).view([n if a == axis else 1 for a in range(3)])
        data = steps.expand(shape).unsqueeze(0)
        origin = torch.full((3,), -40.0, dtype=torch.float64)
        input_affine = torch.eye(4, dtype=torch.float64)
        input_affine[:3, :3] *= cell_size
        input_affine[:3, 3] = origin + cell_size / 2
        affine = torch.eye(4, dtype=torch.float64)
        affine[:3, :3] *= 2 * cell_size
        affine[:3, 3] = origin + cell_size
        size = [1, 1, 1]
        size[axis] = n // 2
        out = resample_grid_mask_3d(
            data, input_affine=input_affine, affine=affine, size=size
        )
        assert out.flatten().tolist() == list(range(1, n, 2))

    def test_bilinear_blends_neighbors(self) -> None:
        mask = GridMask3D(torch.arange(4.0).view(1, 1, 4), affine=make_grid_affine())
        out = _resample(
            mask,
            _shift(mask.affine, w=0.5),
            (1, 4),
            interpolation=InterpolationMode.BILINEAR,
            fill=255,
        )
        assert out.flatten().tolist() == [0.5, 1.5, 2.5, 255.0]

    def test_bilinear_trilinear_volume(self) -> None:
        mask = make_grid_mask_3d(spatial_shape=(2, 2, 2), dtype=torch.float64)
        out = _resample(
            mask,
            _shift(mask.affine, 0.5, 0.5, 0.5),
            (1, 1, 1),
            interpolation=InterpolationMode.BILINEAR,
        )
        torch.testing.assert_close(
            out.flatten(), mask.as_subclass(Tensor).mean().view(1)
        )

    def test_bilinear_does_not_blend_fill(self) -> None:
        # Target cell 0 is inside input cell 0 but before its center.
        mask = GridMask3D(torch.arange(4.0).view(1, 1, 4), affine=make_grid_affine())
        out = _resample(
            mask,
            _shift(mask.affine, w=-0.25),
            (1, 4),
            interpolation=InterpolationMode.BILINEAR,
            fill=255,
        )
        assert out.flatten().tolist() == [0.0, 0.75, 1.75, 2.75]

    @pytest.mark.parametrize(
        ("offset", "inside"), [(0.4, True), (-0.5, True), (0.5, False), (-0.6, False)]
    )
    @pytest.mark.parametrize(
        ("dtype", "interpolation"),
        [
            (torch.int64, InterpolationMode.NEAREST),
            (torch.float32, InterpolationMode.BILINEAR),
        ],
    )
    def test_planar_layer_thickness(
        self,
        offset: float,
        inside: bool,
        dtype: torch.dtype,
        interpolation: InterpolationMode,
    ) -> None:
        # The layer is one cell thick.
        mask = make_grid_mask_3d(spatial_shape=(6, 7), dtype=dtype)
        out = _resample(
            mask,
            _shift(mask.affine, d=offset),
            (6, 7),
            interpolation=interpolation,
            fill=255,
        )
        data = mask.as_subclass(Tensor)
        torch.testing.assert_close(out, data if inside else torch.full_like(data, 255))

    @pytest.mark.parametrize(
        ("spatial_shape", "axis"),
        [
            # Mostly about d, so the planar layer only tilts a little.
            ((5, 6), (5.0, 1.0, 1.0)),
            ((4, 5, 6), (1.0, 2.0, 3.0)),
        ],
    )
    def test_oblique_affine_matches_reference(
        self, spatial_shape: tuple[int, ...], axis: tuple[float, float, float]
    ) -> None:
        # Rotate, shear, scale and reflect, then compare with a per-cell lookup.
        mask = make_grid_mask_3d(channels=2, spatial_shape=spatial_shape)
        x, y, z = torch.nn.functional.normalize(torch.tensor(axis), dim=0).tolist()
        cross = torch.tensor(
            [[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float64, device="cpu"
        )
        rotation = torch.linalg.matrix_exp(math.radians(50) * cross)
        shear = torch.eye(3, dtype=torch.float64, device="cpu")
        shear[1, 2] = 0.3
        scale = torch.diag(cross.new_tensor([0.8, 1.1, -1.3]))
        index_map = torch.eye(4, dtype=torch.float64, device="cpu")
        index_map[:3, :3] = rotation @ shear @ scale
        # Keep the grid centers aligned so most cells stay inside.
        center = (cross.new_tensor((1, *spatial_shape)[-3:]) - 1) / 2
        index_map[:3, 3] = center - index_map[:3, :3] @ center
        out = _resample(mask, mask.affine @ index_map, spatial_shape, fill=-1)

        data = mask.as_subclass(Tensor)
        volume = data.unsqueeze(1) if data.ndim == 3 else data
        expected = torch.full_like(volume, -1)
        for target in itertools.product(*map(range, volume.shape[1:])):
            point = index_map @ index_map.new_tensor([*target, 1.0])
            cell = [math.floor(x + 0.5) for x in point[:3].tolist()]
            if all(0 <= c < n for c, n in zip(cell, volume.shape[1:], strict=True)):
                expected[(slice(None), *target)] = volume[(slice(None), *cell)]
        expected = expected.view(out.shape)
        assert 0 < (expected == -1).sum() < expected.numel() / 2
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize(
        ("dtype", "fill", "expected"),
        [
            (torch.uint8, -1, 255),
            (torch.uint8, [-1.0], 255),
            (torch.int64, 0.7, 1),
            (torch.bool, 0.7, True),
        ],
    )
    def test_fill_is_rounded_for_integer_grids(
        self, dtype: torch.dtype, fill: Any, expected: Any
    ) -> None:
        # ResampleGrid passes per-channel fills as floats.
        mask = GridMask3D(torch.zeros(1, 1, 4, dtype=dtype), affine=make_grid_affine())
        out = _resample(mask, _shift(mask.affine, w=100), (1, 4), fill=fill)
        assert out.flatten().tolist() == [expected] * 4

    def test_fill_defaults_to_zero(self) -> None:
        mask = GridMask3D(torch.ones(1, 8, 8), affine=make_grid_affine())
        out = _resample(mask, _shift(mask.affine, h=100), (8, 8))
        assert (out == 0).all()

    def test_rejects_fill_length(self) -> None:
        mask = make_grid_mask_3d(channels=3)
        with pytest.raises(ValueError, match="1 or 3 values"):
            _resample(mask, mask.affine, (8, 8), fill=[1.0, 2.0])

    @pytest.mark.parametrize(
        ("spatial_shape", "size"), [((8, 8), (1, 8, 8)), ((2, 8, 8), (8, 8))]
    )
    def test_rejects_size_mismatch(
        self, spatial_shape: tuple[int, ...], size: tuple[int, ...]
    ) -> None:
        mask = make_grid_mask_3d(spatial_shape=spatial_shape)
        with pytest.raises(ValueError, match="Expected size"):
            _resample(mask, mask.affine, size)

    @pytest.mark.parametrize("shape", [(8, 8), (1, 1, 8, 8, 8)])
    def test_rejects_non_grid_shape(self, shape: tuple[int, ...]) -> None:
        affine = make_grid_affine()
        with pytest.raises(ValueError, match="Expected a grid"):
            resample_grid_mask_3d(
                torch.zeros(shape), input_affine=affine, affine=affine, size=shape[1:]
            )


class TestResampleGrid:
    def test_resamples_grid_in_sample(self) -> None:
        sample = {"points": make_point_cloud_3d(), "grid": make_grid_mask_3d()}
        affine = _shift(sample["grid"].affine, w=2)
        transform = ResampleGrid(affine=affine, size=(8, 8), fill={GridMask3D: 255})
        out = transform(sample)
        assert out["points"] is sample["points"]
        assert isinstance(out["grid"], GridMask3D)
        torch.testing.assert_close(out["grid"].affine, affine)
        grid = out["grid"].as_subclass(Tensor)
        data = sample["grid"].as_subclass(Tensor)
        torch.testing.assert_close(grid[..., :-2], data[..., 2:])
        assert (grid[..., -2:] == 255).all()

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
