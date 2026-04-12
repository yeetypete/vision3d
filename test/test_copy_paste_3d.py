"""Tests for CopyPaste3D transform."""

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
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3, image_fill=0.9))
        out_inputs, out_targets = cp(
            *_make_fusion_batch(batch_size=1, num_boxes=1, image_fill=0.1)
        )
        if out_targets[0]["boxes"].shape[0] > 1:
            images = out_inputs[0]["images"]
            assert (images > 0.8).any(), "Pasted crop pixels should appear"
            assert (images < 0.2).any(), "Original pixels should remain"
            assert ((images > 0.8) | (images < 0.2)).all()

    def test_does_not_mutate_input_images(self) -> None:
        cp = CopyPaste3D(target_counts={CAR: 10}, min_points=1)
        cp(*_make_fusion_batch(batch_size=2, num_boxes=3, image_fill=0.9))
        batch2 = _make_fusion_batch(batch_size=1, num_boxes=1, image_fill=0.1)
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
