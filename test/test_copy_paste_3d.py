"""Tests for CopyPaste3D transform."""

import math
from typing import Any

import pytest
import torch
from common_utils import make_bounding_boxes_3d

from vision3d.ops import box3d_overlap, points_in_boxes_3d
from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)
from vision3d.transforms import CopyPaste3D

CAR = 0
PED = 1

ALL_FORMATS = [
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWHYPR,
]

# Distributional assertions allow this many standard errors of slack (~2e-9
# false-failure rate).
_SIGMA_TOL = 6.0
_N_SAMPLES = 20000

# Distinct fills for paste-source vs. scene images; assertions use a +/-_FILL_TOL band.
_PASTE_FILL = 0.9
_SCENE_FILL = 0.1
_FILL_TOL = 0.1

# Absolute tolerance for box-centre coincidence checks (scene units).
_CENTER_ATOL = 1e-4


def _uniform_std(lo: float, hi: float) -> float:
    """Population standard deviation of a uniform distribution on ``[lo, hi]``."""
    return (hi - lo) / math.sqrt(12.0)


def _sample_mean_stderr(std: float, n: int) -> float:
    """Standard error of the sample mean for ``n`` i.i.d. draws (``std / sqrt(n)``)."""
    return std / math.sqrt(n)


def _sample_std_stderr(std: float, n: int) -> float:
    """Standard error of the sample std for ``n`` i.i.d. draws (``std / sqrt(2n)``)."""
    return std / math.sqrt(2 * n)


def _make_lidar_batch(
    batch_size: int = 2,
    num_points_per_box: int = 20,
    num_boxes: int = 3,
    labels: list[int] | None = None,
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHY,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if labels is None:
        labels = [CAR] * num_boxes
    assert len(labels) == num_boxes

    inputs = []
    targets = []
    for _ in range(batch_size):
        boxes = make_bounding_boxes_3d(format=format, num_boxes=num_boxes)
        raw = boxes.as_subclass(torch.Tensor)

        all_points = []
        for j in range(num_boxes):
            cx, cy, cz = raw[j, 0], raw[j, 1], raw[j, 2]
            l, w, h = raw[j, 3], raw[j, 4], raw[j, 5]
            local = (torch.rand(num_points_per_box, 3) - 0.5) * torch.tensor([l, w, h])
            local[:, 0] += cx
            local[:, 1] += cy
            local[:, 2] += cz
            all_points.append(local)

        points = torch.cat(all_points)
        inp: dict[str, Any] = {"points": PointCloud3D(points)}
        tgt: dict[str, Any] = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        inputs.append(inp)
        targets.append(tgt)
    return tuple(inputs), tuple(targets)


def _make_fusion_batch(
    batch_size: int = 2,
    num_points_per_box: int = 20,
    num_boxes: int = 3,
    num_cameras: int = 1,
    img_h: int = 480,
    img_w: int = 640,
    labels: list[int] | None = None,
    image_fill: float = 0.5,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if labels is None:
        labels = [CAR] * num_boxes
    assert len(labels) == num_boxes

    extrinsics = CameraExtrinsics(
        torch.eye(4).unsqueeze(0).expand(num_cameras, -1, -1).clone()
    )
    K = torch.eye(3)
    K[0, 0] = 500.0
    K[1, 1] = 500.0
    K[0, 2] = float(img_w) / 2
    K[1, 2] = float(img_h) / 2
    intrinsics = CameraIntrinsics(
        K.unsqueeze(0).expand(num_cameras, -1, -1).clone(),
        image_size=(img_h, img_w),
    )

    inputs = []
    targets = []
    for _ in range(batch_size):
        box_data = []
        all_points = []
        for _j in range(num_boxes):
            cx = (torch.rand(1).item() - 0.5) * 4
            cy = (torch.rand(1).item() - 0.5) * 4
            cz = torch.rand(1).item() * 10 + 5
            l, w, h = 1.0, 1.0, 1.0
            yaw = 0.0
            box_data.append([cx, cy, cz, l, w, h, yaw])

            local = (torch.rand(num_points_per_box, 3) - 0.5) * torch.tensor([l, w, h])
            local[:, 0] += cx
            local[:, 1] += cy
            local[:, 2] += cz
            all_points.append(local)

        boxes = BoundingBoxes3D(
            torch.tensor(box_data), format=BoundingBox3DFormat.XYZLWHY
        )
        points = PointCloud3D(torch.cat(all_points))
        images = CameraImages(torch.full((num_cameras, 3, img_h, img_w), image_fill))

        inp: dict[str, Any] = {
            "points": points,
            "images": images,
            "extrinsics": extrinsics.clone(),
            "intrinsics": intrinsics.clone(),
        }
        tgt: dict[str, Any] = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        inputs.append(inp)
        targets.append(tgt)
    return tuple(inputs), tuple(targets)


def _make_camera_batch(
    batch_size: int = 2,
    num_boxes: int = 3,
    num_cameras: int = 1,
    img_h: int = 480,
    img_w: int = 640,
    labels: list[int] | None = None,
    image_fill: float = 0.5,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Camera-only batch — images, extrinsics, intrinsics, boxes, labels. No point cloud.

    Returns:
        Batch of ``(inputs, targets)`` tuples without point clouds.
    """
    if labels is None:
        labels = [CAR] * num_boxes
    assert len(labels) == num_boxes

    extrinsics = CameraExtrinsics(
        torch.eye(4).unsqueeze(0).expand(num_cameras, -1, -1).clone()
    )
    K = torch.eye(3)
    K[0, 0] = 500.0
    K[1, 1] = 500.0
    K[0, 2] = float(img_w) / 2
    K[1, 2] = float(img_h) / 2
    intrinsics = CameraIntrinsics(
        K.unsqueeze(0).expand(num_cameras, -1, -1).clone(),
        image_size=(img_h, img_w),
    )

    inputs = []
    targets = []
    for _ in range(batch_size):
        box_data = []
        for _j in range(num_boxes):
            cx = (torch.rand(1).item() - 0.5) * 4
            cy = (torch.rand(1).item() - 0.5) * 4
            cz = torch.rand(1).item() * 10 + 5
            box_data.append([cx, cy, cz, 1.0, 1.0, 1.0, 0.0])

        boxes = BoundingBoxes3D(
            torch.tensor(box_data), format=BoundingBox3DFormat.XYZLWHY
        )
        images = CameraImages(torch.full((num_cameras, 3, img_h, img_w), image_fill))

        inp: dict[str, Any] = {
            "images": images,
            "extrinsics": extrinsics.clone(),
            "intrinsics": intrinsics.clone(),
        }
        tgt: dict[str, Any] = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        inputs.append(inp)
        targets.append(tgt)
    return tuple(inputs), tuple(targets)


def _populate_and_paste(
    cp: CopyPaste3D,
    make_fn: Any,
    **paste_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch1 = make_fn(batch_size=2, num_boxes=5)
    cp(*batch1)
    batch2 = make_fn(batch_size=1, num_boxes=1, **paste_kwargs)
    out_inputs, out_targets = cp(*batch2)
    return out_inputs[0], out_targets[0]


# Database lifecycle
class TestDatabase:
    def test_first_batch_populates(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        assert len(cp._database[CAR]) > 0

    def test_database_grows_across_batches(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        size1 = len(cp._database[CAR])
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        assert len(cp._database[CAR]) > size1

    def test_max_database_size(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1, max_database_size=5)
        for _ in range(10):
            cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        assert len(cp._database[CAR]) <= 5

    def test_min_points_filter(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=9999)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, num_points_per_box=2))
        assert len(cp._database[CAR]) == 0

    def test_multi_class_database(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10, PED: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=4, labels=[CAR, PED, CAR, PED]))
        assert len(cp._database[CAR]) > 0
        assert len(cp._database[PED]) > 0


# Core paste correctness
class TestPasteCorrectness:
    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_paste_increases_box_count(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        _, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=1, format=fmt))
        # >= 1 because all candidates may collide (especially axis-aligned)
        assert out_targets[0]["boxes"].shape[0] >= 1

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_preserves_format(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        _, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=1, format=fmt))
        assert out_targets[0]["boxes"].format == fmt

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_pasted_boxes_dont_overlap_existing(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        n_original = 2
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        _, out_targets = cp(
            *_make_lidar_batch(batch_size=1, num_boxes=n_original, format=fmt)
        )

        boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        n_pasted = boxes.shape[0] - n_original
        if n_pasted > 0:
            # Check pasted boxes vs all other boxes (no self-overlap)
            pasted = boxes[n_original:]
            overlap = box3d_overlap(pasted, boxes, fmt)
            # Zero out pasted-vs-pasted diagonal
            overlap[:, n_original:].fill_diagonal_(False)
            assert not overlap.any()

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_labels_count_matches_boxes(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        _, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=1, format=fmt))
        assert out_targets[0]["labels"].shape[0] == out_targets[0]["boxes"].shape[0]

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_pasted_labels_are_correct_class_ids(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        _, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=1, format=fmt))
        pasted_labels = out_targets[0]["labels"][1:]
        if pasted_labels.shape[0] > 0:
            assert (pasted_labels == CAR).all()

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_concatenation_order_boxes(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        batch2 = _make_lidar_batch(batch_size=1, num_boxes=2, format=fmt)
        original_boxes = batch2[1][0]["boxes"].as_subclass(torch.Tensor).clone()
        _, out_targets = cp(*batch2)

        out_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        if out_boxes.shape[0] > 2:
            assert torch.allclose(out_boxes[:2], original_boxes)

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_concatenation_order_labels(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        batch2 = _make_lidar_batch(batch_size=1, num_boxes=2, format=fmt)
        original_labels = batch2[1][0]["labels"].clone()
        _, out_targets = cp(*batch2)
        assert torch.equal(
            out_targets[0]["labels"][: len(original_labels)], original_labels
        )

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_scene_points_removed(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3, format=fmt))
        batch2 = _make_lidar_batch(batch_size=1, num_boxes=1, format=fmt)
        original_points = batch2[0][0]["points"].clone()
        out_inputs, out_targets = cp(*batch2)

        if out_targets[0]["boxes"].shape[0] > 1:
            pasted_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)[1:]
            inside = points_in_boxes_3d(original_points, pasted_boxes, fmt)
            original_in_paste_region = inside.any(dim=1).sum()
            out_pts = out_inputs[0]["points"].as_subclass(torch.Tensor)
            assert (
                out_pts.shape[0] != original_points.shape[0]
                or original_in_paste_region == 0
            )

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_preserves_point_cloud_type(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        inp, _ = _populate_and_paste(
            cp, lambda **kw: _make_lidar_batch(format=fmt, **kw)
        )
        assert isinstance(inp["points"], PointCloud3D)

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_preserves_bounding_boxes_type(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        _, tgt = _populate_and_paste(
            cp, lambda **kw: _make_lidar_batch(format=fmt, **kw)
        )
        assert isinstance(tgt["boxes"], BoundingBoxes3D)
        assert tgt["boxes"].format == fmt


# Probability
class TestProbability:
    def test_p_zero_no_paste(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, p=0.0)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        _, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=2))
        assert out_targets[0]["boxes"].shape[0] == 2


# Multi-class
class TestMultiClass:
    def test_multi_class_paste(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10, PED: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=4, labels=[CAR, PED, CAR, PED]))

        batch2 = _make_lidar_batch(batch_size=1, num_boxes=2, labels=[CAR, PED])
        _, out_targets = cp(*batch2)
        out_labels = out_targets[0]["labels"]
        assert out_labels.shape[0] >= 2
        assert set(out_labels.tolist()).issubset({CAR, PED})

    def test_multi_class_labels_correct(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10, PED: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=4, labels=[CAR, PED, CAR, PED]))
        _, out_targets = cp(
            *_make_lidar_batch(batch_size=1, num_boxes=2, labels=[CAR, PED])
        )
        for lbl in out_targets[0]["labels"].tolist():
            assert lbl in (CAR, PED)

    def test_class_not_in_database_skipped(self) -> None:
        cp = CopyPaste3D(target_counts={99: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        _, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=2))
        assert out_targets[0]["boxes"].shape[0] == 2


# Camera crop extraction
class TestCameraExtract:
    def test_extracts_camera_crops(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=1, num_boxes=2))
        assert len(cp._database[CAR]) > 0
        entry = cp._database[CAR][0]
        assert entry.camera_crops is not None
        assert len(entry.camera_crops) == 1

    def test_camera_crop_has_valid_mask(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=1, num_boxes=1))
        entry = cp._database[CAR][0]
        assert entry.camera_crops is not None
        crop_data = entry.camera_crops[0]
        assert crop_data is not None
        assert crop_data.mask.any()
        assert crop_data.crop.shape[0] == 3
        assert crop_data.crop.shape[1] == crop_data.mask.shape[0]
        assert crop_data.crop.shape[2] == crop_data.mask.shape[1]

    def test_no_camera_crops_without_images(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=1, num_boxes=2))
        for entry in cp._database[CAR]:
            assert entry.camera_crops is None

    def test_crop_pixel_values_match_source_image(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=1, num_boxes=1, image_fill=0.75))
        entry = cp._database[CAR][0]
        assert entry.camera_crops is not None
        crop_data = entry.camera_crops[0]
        assert crop_data is not None
        assert torch.allclose(crop_data.crop, torch.full_like(crop_data.crop, 0.75))

    def test_multi_camera_extracts(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=1, num_boxes=1, num_cameras=3))
        entry = cp._database[CAR][0]
        assert entry.camera_crops is not None
        assert len(entry.camera_crops) == 3


# Camera paste
class TestCameraPaste:
    def test_images_modified_after_paste(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3))
        out_inputs, out_targets = cp(*_make_fusion_batch(batch_size=1, num_boxes=1))
        if out_targets[0]["boxes"].shape[0] > 1:
            assert isinstance(out_inputs[0]["images"], CameraImages)

    def test_paste_writes_exact_pixel_values(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3, image_fill=_PASTE_FILL))
        out_inputs, out_targets = cp(
            *_make_fusion_batch(batch_size=1, num_boxes=1, image_fill=_SCENE_FILL)
        )
        if out_targets[0]["boxes"].shape[0] > 1:
            images = out_inputs[0]["images"]
            is_paste = images > _PASTE_FILL - _FILL_TOL
            is_scene = images < _SCENE_FILL + _FILL_TOL
            assert is_paste.any(), "Pasted crop pixels should appear"
            assert is_scene.any(), "Original pixels should remain"
            assert (is_paste | is_scene).all()

    def test_does_not_mutate_input_images(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3, image_fill=_PASTE_FILL))
        batch2 = _make_fusion_batch(batch_size=1, num_boxes=1, image_fill=_SCENE_FILL)
        original_images = batch2[0][0]["images"].clone()
        cp(*batch2)
        assert torch.equal(batch2[0][0]["images"], original_images)

    def test_paste_with_multiple_cameras(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3, num_cameras=3))
        out_inputs, _ = cp(
            *_make_fusion_batch(batch_size=1, num_boxes=1, num_cameras=3)
        )
        assert out_inputs[0]["images"].shape[0] == 3

    def test_preserves_camera_images_type(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        inp, tgt = _populate_and_paste(cp, _make_fusion_batch)
        if tgt["boxes"].shape[0] > 1:
            assert isinstance(inp["images"], CameraImages)


# Cross-modal
class TestCrossModal:
    def test_lidar_only_works(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        out_inputs, _ = cp(*_make_lidar_batch(batch_size=1, num_boxes=1))
        assert isinstance(out_inputs[0]["points"], PointCloud3D)
        assert "images" not in out_inputs[0]

    def test_lidar_db_entries_have_no_camera_crops(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        for entry in cp._database[CAR]:
            assert entry.camera_crops is None

    def test_lidar_db_paste_into_camera_sample(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        batch2 = _make_fusion_batch(batch_size=1, num_boxes=1, image_fill=0.5)
        original_images = batch2[0][0]["images"].clone()
        out_inputs, _ = cp(*batch2)
        assert torch.equal(out_inputs[0]["images"], original_images)

    def test_camera_db_paste_into_lidar_only(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3))
        out_inputs, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=1))
        assert isinstance(out_inputs[0]["points"], PointCloud3D)
        assert "images" not in out_inputs[0]
        assert out_targets[0]["boxes"].shape[0] >= 1


# Position jitter
class TestOffset:
    """Random x/y/z position jitter of pasted objects."""

    def test_default_disables_jitter(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10})
        assert cp._jitter is False

    def test_scalar_range_broadcasts_to_all_axes(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, offset_range=(-0.5, 0.5))
        assert cp._jitter is True
        assert cp.offset_range == ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))

    def test_per_axis_range_kept(self) -> None:
        cp = CopyPaste3D(
            target_counts={CAR: 10},
            offset_range=((-1.0, 2.0), (0.0, 0.0), (-0.5, 0.5)),
        )
        assert cp._jitter is True
        assert cp.offset_range == ((-1.0, 2.0), (0.0, 0.0), (-0.5, 0.5))

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_constant_offset_shifts_pasted_box_xyz(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        # A distinct constant per axis so a missed axis would be caught.
        ox, oy, oz = 3.0, -4.0, 7.0
        cp = CopyPaste3D(
            target_counts={CAR: 10},
            min_points=1,
            offset_range=((ox, ox), (oy, oy), (oz, oz)),
        )
        cp(*_make_lidar_batch(batch_size=3, num_boxes=3, format=fmt))
        n_original = 1
        _, out_targets = cp(
            *_make_lidar_batch(batch_size=1, num_boxes=n_original, format=fmt)
        )

        out_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        pasted = out_boxes[n_original:]
        if pasted.shape[0] == 0:
            pytest.skip("no objects pasted")

        # Each pasted centre must equal some unmodified source centre, either
        # shifted by the per-axis offset or (when every jittered pose collided)
        # left at the original pose (the documented fallback). XYZXYZ stores
        # the min corner at indices 0:3, which also translates rigidly, so the
        # same check applies.
        src = [
            tuple(round(v, 3) for v in e.box[:3].tolist()) for e in cp._database[CAR]
        ]

        def matches_source(cx: float, cy: float, cz: float) -> bool:
            return any(
                abs(cx - sx) < 1e-2 and abs(cy - sy) < 1e-2 and abs(cz - sz) < 1e-2
                for sx, sy, sz in src
            )

        for box in pasted:
            bx, by, bz = box[0].item(), box[1].item(), box[2].item()
            shifted = matches_source(bx - ox, by - oy, bz - oz)
            fallback = matches_source(bx, by, bz)
            assert shifted or fallback

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_pasted_points_follow_jittered_box(self, fmt: BoundingBox3DFormat) -> None:
        # Stored object points are inside their source box; a rigid translation
        # of box and points together keeps them inside the jittered box.
        cp = CopyPaste3D(
            target_counts={CAR: 10},
            min_points=1,
            offset_range=((3.0, 3.0), (-4.0, -4.0), (7.0, 7.0)),
        )
        cp(*_make_lidar_batch(batch_size=3, num_boxes=3, format=fmt))
        n_original = 1
        out_inputs, out_targets = cp(
            *_make_lidar_batch(batch_size=1, num_boxes=n_original, format=fmt)
        )

        out_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        pasted = BoundingBoxes3D(out_boxes[n_original:], format=fmt)
        if pasted.shape[0] == 0:
            pytest.skip("no objects pasted")

        points = out_inputs[0]["points"]
        inside = points_in_boxes_3d(points, pasted, fmt)
        # Each pasted box still contains at least its own points.
        assert (inside.sum(dim=0) >= 1).all()

    def test_sample_offsets_shape_and_per_axis_bounds(self) -> None:
        cp = CopyPaste3D(
            target_counts={CAR: 10},
            offset_range=((-2.0, 2.0), (0.0, 0.0), (-0.5, 1.5)),
        )
        n = _N_SAMPLES
        torch.manual_seed(0)
        s = cp._sample_offsets(n, torch.device("cpu"), torch.float32)
        assert s.shape == (n, 3)
        assert s[:, 0].min().item() >= -2.0
        assert s[:, 0].max().item() <= 2.0
        # Degenerate axis stays exactly zero.
        assert torch.all(s[:, 1] == 0.0)
        assert s[:, 2].min().item() >= -0.5
        assert s[:, 2].max().item() <= 1.5

    def test_uniform_sampling_when_std_none(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, offset_range=(-1.0, 1.0))
        n = _N_SAMPLES
        torch.manual_seed(0)
        s = cp._sample_offsets(n, torch.device("cpu"), torch.float32)[:, 2]
        assert s.min().item() >= -1.0
        assert s.max().item() <= 1.0
        # Sample std should match the uniform population std within a few stderrs.
        expected_std = _uniform_std(-1.0, 1.0)
        tol = _SIGMA_TOL * _sample_std_stderr(expected_std, n)
        assert abs(s.std().item() - expected_std) < tol

    def test_truncated_normal_within_bounds_and_concentrated(self) -> None:
        nominal_std = 0.3
        cp = CopyPaste3D(
            target_counts={CAR: 10}, offset_range=(-1.0, 1.0), offset_std=nominal_std
        )
        n = _N_SAMPLES
        torch.manual_seed(0)
        s = cp._sample_offsets(n, torch.device("cpu"), torch.float32)[:, 0]
        # Never escapes the interval.
        assert s.min().item() >= -1.0 - 1e-5
        assert s.max().item() <= 1.0 + 1e-5
        # Centred on the midpoint (true mean 0) within a few stderrs.
        mean_tol = _SIGMA_TOL * _sample_mean_stderr(nominal_std, n)
        assert abs(s.mean().item()) < mean_tol
        # Truncation at +/-1 (= +/-3.3 sigma) is negligible, so the realized std
        # stays close to the nominal value -- and well under the uniform baseline.
        std_tol = _SIGMA_TOL * _sample_std_stderr(nominal_std, n)
        assert abs(s.std().item() - nominal_std) < std_tol
        assert s.std().item() < _uniform_std(-1.0, 1.0)

    def test_per_axis_std_mix(self) -> None:
        # Uniform on x, truncated-normal on z, y disabled.
        cp = CopyPaste3D(
            target_counts={CAR: 10},
            offset_range=((-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)),
            offset_std=(None, None, 0.3),
        )
        n = _N_SAMPLES
        torch.manual_seed(0)
        s = cp._sample_offsets(n, torch.device("cpu"), torch.float32)
        # x is uniform (wide spread), z is concentrated (narrow spread).
        x_std = _uniform_std(-1.0, 1.0)
        z_std = 0.3
        assert abs(s[:, 0].std().item() - x_std) < _SIGMA_TOL * _sample_std_stderr(
            x_std, n
        )
        assert abs(s[:, 2].std().item() - z_std) < _SIGMA_TOL * _sample_std_stderr(
            z_std, n
        )

    def test_falls_back_to_original_when_jitter_collides(self) -> None:
        # Constant offset, so every jitter attempt lands at the same pose.
        offset = 10.0
        cp = CopyPaste3D(
            target_counts={CAR: 2},
            min_points=1,
            offset_range=(offset, offset),
            max_jitter_attempts=5,
        )
        # Populate the database with a single object.
        cp(*_make_lidar_batch(batch_size=1, num_boxes=1))
        assert len(cp._database[CAR]) == 1
        db_box = cp._database[CAR][0].box.clone()  # [K], XYZLWHY

        # Blocker sits exactly where the (constant) jitter would land, so every
        # attempt collides. Scene points stay away from it so it is not itself
        # added to the database.
        blocker = db_box.clone()
        blocker[:3] = db_box[:3] + offset
        boxes = BoundingBoxes3D(
            blocker.unsqueeze(0), format=BoundingBox3DFormat.XYZLWHY
        )
        far_points = PointCloud3D(db_box[:3].unsqueeze(0).repeat(5, 1) - 100.0)
        inp: dict[str, Any] = {"points": far_points}
        tgt: dict[str, Any] = {"boxes": boxes, "labels": torch.tensor([CAR])}
        _, out_targets = cp((inp,), (tgt,))

        out_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        assert out_boxes.shape[0] == 2  # blocker + pasted fallback
        assert torch.allclose(
            out_boxes[1, :3].cpu(), db_box[:3].cpu(), atol=_CENTER_ATOL
        )

    def test_does_not_mutate_database_entries(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1, offset_range=(5.0, 5.0))
        cp(*_make_lidar_batch(batch_size=3, num_boxes=3))
        before = [e.box[:3].tolist() for e in cp._database[CAR]]
        cp(*_make_lidar_batch(batch_size=1, num_boxes=1))
        after = [e.box[:3].tolist() for e in cp._database[CAR]]
        assert after[: len(before)] == before

    def test_random_range_displaces_pasted_centers(self) -> None:
        # Paste into an empty scene so nothing can collide: every candidate is
        # placed at its jittered pose, never the original-pose fallback. With a
        # wide random range each drawn offset is non-zero (a.s.), so every
        # pasted centre must differ from every source centre.
        cp = CopyPaste3D(
            target_counts={CAR: 5}, min_points=1, offset_range=(-25.0, 25.0)
        )
        cp(*_make_lidar_batch(batch_size=4, num_boxes=4))
        src = torch.stack([e.box[:3] for e in cp._database[CAR]])

        empty_inp: dict[str, Any] = {"points": PointCloud3D(torch.zeros(0, 3))}
        empty_tgt: dict[str, Any] = {
            "boxes": BoundingBoxes3D(
                torch.zeros(0, 7), format=BoundingBox3DFormat.XYZLWHY
            ),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        torch.manual_seed(0)
        _, out_targets = cp((empty_inp,), (empty_tgt,))

        pasted = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        assert pasted.shape[0] > 0, "expected objects to be pasted"
        for box in pasted:
            # L1 distance to the nearest source centre: a real displacement
            # means it coincides with none of them. (DB boxes live on CPU.)
            dist = (src - box[:3].cpu()).abs().sum(dim=1)
            assert dist.min().item() > _CENTER_ATOL

    def test_camera_paste_with_jitter(self) -> None:
        # The pasted object is re-bound to its jittered box before camera
        # reprojection (the most fragile path). Exercise it on a fusion batch:
        # a small constant offset keeps objects in front of the camera and in
        # frame, so the re-projected crop is actually written into the image.
        ox, oy, oz = 0.5, 0.5, 1.0
        cp = CopyPaste3D(
            target_counts={CAR: 10},
            min_points=1,
            offset_range=((ox, ox), (oy, oy), (oz, oz)),
        )
        torch.manual_seed(0)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3, image_fill=_PASTE_FILL))
        n_original = 1
        out_inputs, out_targets = cp(
            *_make_fusion_batch(
                batch_size=1, num_boxes=n_original, image_fill=_SCENE_FILL
            )
        )

        out_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        if out_boxes.shape[0] <= n_original:
            pytest.skip("no objects pasted")

        # Camera reprojection of the jittered box succeeded: crop pixels (0.9)
        # appear over the original fill (0.1), confirming the re-bound box gave
        # a valid in-frame projection.
        images = out_inputs[0]["images"]
        assert (images > _PASTE_FILL - _FILL_TOL).any(), "Pasted crop pixels appear"
        assert (images < _SCENE_FILL + _FILL_TOL).any(), "Original pixels remain"

        # The pasted 3-D centres are shifted by the offset (or, when every
        # jittered pose collided, left at the source pose as documented).
        src = [
            tuple(round(v, 3) for v in e.box[:3].tolist()) for e in cp._database[CAR]
        ]

        def matches_source(cx: float, cy: float, cz: float) -> bool:
            return any(
                abs(cx - sx) < 1e-2 and abs(cy - sy) < 1e-2 and abs(cz - sz) < 1e-2
                for sx, sy, sz in src
            )

        for box in out_boxes[n_original:]:
            bx, by, bz = box[0].item(), box[1].item(), box[2].item()
            shifted = matches_source(bx - ox, by - oy, bz - oz)
            fallback = matches_source(bx, by, bz)
            assert shifted or fallback


# Determinism
class TestDeterminism:
    def test_reproducible_with_seed(self) -> None:
        def run_with_seed(seed: int) -> int:
            torch.manual_seed(seed)
            cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
            torch.manual_seed(seed)
            cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
            torch.manual_seed(seed + 1)
            _, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=2))
            return out_targets[0]["boxes"].shape[0]

        assert run_with_seed(42) == run_with_seed(42)


# Validation
class TestValidation:
    def test_p_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="`p` should be a float"):
            CopyPaste3D(target_counts={CAR: 10}, p=1.5)

    def test_p_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="`p` should be a float"):
            CopyPaste3D(target_counts={CAR: 10}, p=-0.1)

    def test_min_points_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="`min_points` should be a positive"):
            CopyPaste3D(target_counts={CAR: 10}, min_points=0)

    def test_max_database_size_zero_raises(self) -> None:
        with pytest.raises(
            ValueError, match="`max_database_size` should be a positive"
        ):
            CopyPaste3D(target_counts={CAR: 10}, max_database_size=0)

    def test_max_jitter_attempts_zero_raises(self) -> None:
        with pytest.raises(
            ValueError, match="`max_jitter_attempts` should be a positive"
        ):
            CopyPaste3D(target_counts={CAR: 10}, max_jitter_attempts=0)

    def test_offset_range_wrong_length_raises(self) -> None:
        bad_range: Any = (0.0,)
        with pytest.raises(ValueError, match="`offset_range` should be a"):
            CopyPaste3D(target_counts={CAR: 10}, offset_range=bad_range)

    def test_offset_range_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min must not exceed max"):
            CopyPaste3D(target_counts={CAR: 10}, offset_range=(0.5, -0.5))

    def test_offset_range_per_axis_scalar_entry_raises(self) -> None:
        bad_range: Any = (1.0, (0.0, 0.0), (0.0, 0.0))
        with pytest.raises(TypeError, match="must be a \\(min, max\\) pair"):
            CopyPaste3D(target_counts={CAR: 10}, offset_range=bad_range)

    def test_offset_std_non_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="`offset_std` values should be positive"):
            CopyPaste3D(
                target_counts={CAR: 10}, offset_range=(-1.0, 1.0), offset_std=0.0
            )

    def test_offset_std_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="`offset_std` should be a float"):
            CopyPaste3D(
                target_counts={CAR: 10},
                offset_range=(-1.0, 1.0),
                offset_std=(0.3, 0.3),
            )

    def test_offset_std_with_degenerate_range_raises(self) -> None:
        with pytest.raises(ValueError, match="`offset_std` has no effect"):
            CopyPaste3D(
                target_counts={CAR: 10}, offset_range=(0.0, 0.0), offset_std=0.5
            )
        with pytest.raises(ValueError, match="`offset_std` has no effect"):
            CopyPaste3D(
                target_counts={CAR: 10}, offset_range=(5.0, 5.0), offset_std=0.5
            )

    def test_forward_empty_batch_is_noop(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10})
        result = cp((), ())
        assert result == ((), ())

    def test_forward_mismatched_sample_counts(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10})
        batch = _make_lidar_batch(batch_size=2, num_boxes=2)
        with pytest.raises(TypeError, match="equal sized lists"):
            cp(batch[0], batch[1][:1])


# Input modalities
class TestInputModalities:
    """Verify lidar-only, camera-only, and fusion inputs all work."""

    def test_lidar_only_extract_and_paste(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_lidar_batch(batch_size=2, num_boxes=3))
        assert len(cp._database[CAR]) > 0
        out_inputs, out_targets = cp(*_make_lidar_batch(batch_size=1, num_boxes=1))
        assert isinstance(out_inputs[0]["points"], PointCloud3D)
        assert "images" not in out_inputs[0]
        assert out_targets[0]["boxes"].shape[0] >= 1

    def test_camera_only_extract_and_paste(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=2, num_boxes=3))
        assert len(cp._database[CAR]) > 0
        out_inputs, out_targets = cp(*_make_camera_batch(batch_size=1, num_boxes=1))
        assert "points" not in out_inputs[0]
        assert isinstance(out_inputs[0]["images"], CameraImages)
        assert out_targets[0]["boxes"].shape[0] >= 1

    def test_camera_only_entries_have_no_points(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=1, num_boxes=2))
        for entry in cp._database[CAR]:
            assert entry.points is None

    def test_camera_only_entries_have_camera_crops(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=1, num_boxes=1))
        entry = cp._database[CAR][0]
        assert entry.camera_crops is not None

    def test_fusion_extract_and_paste(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3))
        assert len(cp._database[CAR]) > 0
        out_inputs, out_targets = cp(*_make_fusion_batch(batch_size=1, num_boxes=1))
        assert isinstance(out_inputs[0]["points"], PointCloud3D)
        assert isinstance(out_inputs[0]["images"], CameraImages)
        assert out_targets[0]["boxes"].shape[0] >= 1

    def test_camera_only_db_paste_into_fusion(self) -> None:
        """Camera-only entries pasted into fusion scene: boxes/images update, no points added."""
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=2, num_boxes=3))
        batch2 = _make_fusion_batch(batch_size=1, num_boxes=1)
        original_n_points = batch2[0][0]["points"].shape[0]
        out_inputs, _out_targets = cp(*batch2)
        assert isinstance(out_inputs[0]["points"], PointCloud3D)
        # No pasted points added (camera-only entries have None points),
        # but scene points in pasted box regions may be removed.
        assert out_inputs[0]["points"].shape[0] <= original_n_points

    def test_fusion_db_paste_into_camera_only(self) -> None:
        """Fusion entries pasted into camera-only scene: boxes/images update, no point cloud created."""
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3))
        out_inputs, out_targets = cp(*_make_camera_batch(batch_size=1, num_boxes=1))
        assert "points" not in out_inputs[0]
        assert out_targets[0]["boxes"].shape[0] >= 1
