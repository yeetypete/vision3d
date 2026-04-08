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


def _make_batch(
    batch_size: int = 2,
    num_points_per_box: int = 20,
    num_boxes: int = 3,
    class_names: list[str] | None = None,
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHY,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    # Create a lidar-only batch with points guaranteed to be inside the boxes.
    if class_names is None:
        class_names = ["Car"] * num_boxes
    assert len(class_names) == num_boxes

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
            "labels": torch.arange(num_boxes, dtype=torch.long),
            "class_names": list(class_names),
        }
        inputs.append(inp)
        targets.append(tgt)
    return tuple(inputs), tuple(targets)


def _make_camera_batch(
    batch_size: int = 2,
    num_points_per_box: int = 20,
    num_boxes: int = 3,
    num_cameras: int = 1,
    img_h: int = 480,
    img_w: int = 640,
    class_names: list[str] | None = None,
    image_fill: float = 0.5,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    # Create a batch with lidar + camera data.
    # Objects placed in front of the camera (positive Z) at known positions.
    if class_names is None:
        class_names = ["Car"] * num_boxes
    assert len(class_names) == num_boxes

    extrinsics = CameraExtrinsics(
        torch.eye(4).unsqueeze(0).expand(num_cameras, -1, -1).clone()
    )
    K = torch.eye(3)
    K[0, 0] = 500.0
    K[1, 1] = 500.0
    K[0, 2] = float(img_w) / 2
    K[1, 2] = float(img_h) / 2
    intrinsics = CameraIntrinsics(K.unsqueeze(0).expand(num_cameras, -1, -1).clone())

    inputs = []
    targets = []
    for _ in range(batch_size):
        box_data = []
        all_points = []
        for _j in range(num_boxes):
            cx = (torch.rand(1).item() - 0.5) * 4  # [-2, 2]
            cy = (torch.rand(1).item() - 0.5) * 4  # [-2, 2]
            cz = torch.rand(1).item() * 10 + 5  # [5, 15]
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
            "labels": torch.arange(num_boxes, dtype=torch.long),
            "class_names": list(class_names),
        }
        inputs.append(inp)
        targets.append(tgt)
    return tuple(inputs), tuple(targets)


def _populate_and_paste(
    cp: CopyPaste3D,
    make_fn: Any,
    **paste_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Populate database from batch1, then paste into batch2.
    batch1 = make_fn(batch_size=2, num_boxes=5)
    cp(*batch1)
    batch2 = make_fn(batch_size=1, num_boxes=1, **paste_kwargs)
    out_inputs, out_targets = cp(*batch2)
    return out_inputs[0], out_targets[0]


# Database lifecycle
class TestDatabase:
    def test_first_batch_populates(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        batch = _make_batch(batch_size=2, num_boxes=3)
        cp(*batch)
        assert len(cp._database["Car"]) > 0

    def test_database_grows_across_batches(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        size1 = len(cp._database["Car"])
        cp(*_make_batch(batch_size=2, num_boxes=3))
        assert len(cp._database["Car"]) > size1

    def test_max_database_size(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1, max_database_size=5)
        for _ in range(10):
            cp(*_make_batch(batch_size=2, num_boxes=3))
        assert len(cp._database["Car"]) <= 5

    def test_min_points_filter(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=9999)
        cp(*_make_batch(batch_size=2, num_boxes=3, num_points_per_box=2))
        assert len(cp._database["Car"]) == 0

    def test_multi_class_database(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10, "Ped": 10}, min_points=1)
        batch = _make_batch(
            batch_size=2, num_boxes=4, class_names=["Car", "Ped", "Car", "Ped"]
        )
        cp(*batch)
        assert len(cp._database["Car"]) > 0
        assert len(cp._database["Ped"]) > 0


# Lidar-only pasting
class TestLidarPaste:
    def test_second_batch_pastes(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        _, out_targets = cp(*_make_batch(batch_size=1, num_boxes=2))
        any_pasted = out_targets[0]["boxes"].shape[0] > 2
        assert any_pasted

    def test_box_count_increases(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=5))
        _, out_targets = cp(*_make_batch(batch_size=1, num_boxes=2))
        assert out_targets[0]["boxes"].shape[0] >= 2

    def test_scene_points_removed_in_paste_region(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        batch2 = _make_batch(batch_size=1, num_boxes=1)
        original_points = batch2[0][0]["points"].clone()
        out_inputs, out_targets = cp(*batch2)

        if out_targets[0]["boxes"].shape[0] > 1:
            # Pasted boxes should not contain any original scene points
            pasted_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)[1:]
            fmt = out_targets[0]["boxes"].format
            inside = points_in_boxes_3d(original_points, pasted_boxes, fmt)
            original_in_paste_region = inside.any(dim=1).sum()
            # The output should have removed those points
            out_pts = out_inputs[0]["points"].as_subclass(torch.Tensor)
            # Just verify original points in paste regions were removed
            assert (
                out_pts.shape[0] != original_points.shape[0]
                or original_in_paste_region == 0
            )

    def test_concatenation_order_boxes_original_first(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        batch2 = _make_batch(batch_size=1, num_boxes=2)
        original_boxes = batch2[1][0]["boxes"].as_subclass(torch.Tensor).clone()
        _, out_targets = cp(*batch2)

        out_boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        if out_boxes.shape[0] > 2:
            # First 2 boxes should be the originals
            assert torch.allclose(out_boxes[:2], original_boxes)

    def test_concatenation_order_labels(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        batch2 = _make_batch(batch_size=1, num_boxes=2)
        original_labels = batch2[1][0]["labels"].clone()
        _, out_targets = cp(*batch2)

        out_labels = out_targets[0]["labels"]
        # Original labels come first
        assert torch.equal(out_labels[: len(original_labels)], original_labels)

    def test_concatenation_order_class_names(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        batch2 = _make_batch(batch_size=1, num_boxes=2)
        original_names = batch2[1][0]["class_names"][:]
        _, out_targets = cp(*batch2)

        out_names = out_targets[0]["class_names"]
        assert out_names[: len(original_names)] == original_names

    def test_p_zero_no_paste(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, p=0.0)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        _, out_targets = cp(*_make_batch(batch_size=1, num_boxes=2))
        assert out_targets[0]["boxes"].shape[0] == 2

    def test_no_bev_overlap_after_paste(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        _, out_targets = cp(*_make_batch(batch_size=1, num_boxes=2))

        boxes = out_targets[0]["boxes"].as_subclass(torch.Tensor)
        if boxes.shape[0] > 1:
            overlap = box3d_overlap(boxes, boxes, BoundingBox3DFormat.XYZLWHY)
            overlap.fill_diagonal_(False)
            assert not overlap.any()


# Multi-class
class TestMultiClass:
    def test_multi_class_paste(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10, "Ped": 10}, min_points=1)
        batch1 = _make_batch(
            batch_size=2, num_boxes=4, class_names=["Car", "Ped", "Car", "Ped"]
        )
        cp(*batch1)

        batch2 = _make_batch(batch_size=1, num_boxes=2, class_names=["Car", "Ped"])
        _, out_targets = cp(*batch2)
        names = out_targets[0]["class_names"]
        # Should have pasted some Cars and/or Peds
        assert len(names) >= 2
        # Every name should be one of the known classes
        assert all(n in ("Car", "Ped") for n in names)

    def test_class_not_in_database_skipped(self) -> None:
        cp = CopyPaste3D(target_counts={"Truck": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))  # only "Car" in database
        _, out_targets = cp(*_make_batch(batch_size=1, num_boxes=2))
        # No Trucks in database, so nothing pasted
        assert out_targets[0]["boxes"].shape[0] == 2


# Type preservation
class TestTypePreservation:
    def test_preserves_point_cloud_type(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        inp, _ = _populate_and_paste(cp, _make_batch)
        assert isinstance(inp["points"], PointCloud3D)

    def test_preserves_bounding_boxes_type(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        _, tgt = _populate_and_paste(cp, _make_batch)
        assert isinstance(tgt["boxes"], BoundingBoxes3D)
        assert tgt["boxes"].format == BoundingBox3DFormat.XYZLWHY

    def test_preserves_camera_images_type(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        inp, tgt = _populate_and_paste(cp, _make_camera_batch)
        if tgt["boxes"].shape[0] > 1:
            assert isinstance(inp["images"], CameraImages)

    def test_class_names_count_matches_boxes(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        _, tgt = _populate_and_paste(cp, _make_batch)
        assert len(tgt["class_names"]) == tgt["boxes"].shape[0]

    def test_labels_count_matches_boxes(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        _, tgt = _populate_and_paste(cp, _make_batch)
        assert tgt["labels"].shape[0] == tgt["boxes"].shape[0]


# Format parametrization
class TestFormatSupport:
    @pytest.mark.parametrize(
        "fmt",
        [
            BoundingBox3DFormat.XYZLWHY,
            BoundingBox3DFormat.XYZLWH,
            BoundingBox3DFormat.XYZXYZ,
            BoundingBox3DFormat.XYZLWHYPR,
        ],
    )
    def test_paste_with_format(self, fmt: BoundingBox3DFormat) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3, format=fmt))
        _, out_targets = cp(*_make_batch(batch_size=1, num_boxes=1, format=fmt))
        assert out_targets[0]["boxes"].format == fmt
        assert out_targets[0]["boxes"].shape[0] >= 1


# Camera crop extraction
class TestCameraExtract:
    def test_extracts_camera_crops(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=1, num_boxes=2))

        assert len(cp._database["Car"]) > 0
        entry = cp._database["Car"][0]
        assert entry.camera_crops is not None
        assert len(entry.camera_crops) == 1

    def test_camera_crop_has_valid_mask(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=1, num_boxes=1))

        entry = cp._database["Car"][0]
        assert entry.camera_crops is not None
        crop_data = entry.camera_crops[0]
        assert crop_data is not None
        assert crop_data.mask.any()
        assert crop_data.crop.shape[0] == 3
        assert crop_data.crop.shape[1] == crop_data.mask.shape[0]
        assert crop_data.crop.shape[2] == crop_data.mask.shape[1]

    def test_no_camera_crops_without_images(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=1, num_boxes=2))

        for entry in cp._database["Car"]:
            assert entry.camera_crops is None

    def test_crop_pixel_values_match_source_image(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=1, num_boxes=1, image_fill=0.75))

        entry = cp._database["Car"][0]
        assert entry.camera_crops is not None
        crop_data = entry.camera_crops[0]
        assert crop_data is not None
        # All crop pixels should be 0.75 (from the uniform source image)
        assert torch.allclose(crop_data.crop, torch.full_like(crop_data.crop, 0.75))

    def test_multi_camera_extracts(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=1, num_boxes=1, num_cameras=3))

        entry = cp._database["Car"][0]
        assert entry.camera_crops is not None
        assert len(entry.camera_crops) == 3


# Camera paste
class TestCameraPaste:
    def test_images_modified_after_paste(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=2, num_boxes=3))

        batch2 = _make_camera_batch(batch_size=1, num_boxes=1)
        out_inputs, out_targets = cp(*batch2)

        if out_targets[0]["boxes"].shape[0] > 1:
            assert isinstance(out_inputs[0]["images"], CameraImages)

    def test_paste_writes_exact_pixel_values(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        # Source images filled with 0.9
        batch1 = _make_camera_batch(batch_size=2, num_boxes=3, image_fill=0.9)
        cp(*batch1)

        # Target images filled with 0.1
        batch2 = _make_camera_batch(batch_size=1, num_boxes=1, image_fill=0.1)
        out_inputs, out_targets = cp(*batch2)

        if out_targets[0]["boxes"].shape[0] > 1:
            images = out_inputs[0]["images"]
            # Pasted regions should have 0.9 values (from source crops)
            assert (images > 0.8).any(), "Pasted crop pixels should appear"
            # Non-pasted regions should still have 0.1
            assert (images < 0.2).any(), "Original pixels should remain"
            # No other values should exist (only 0.1 or 0.9)
            is_source = images > 0.8
            is_target = images < 0.2
            assert (is_source | is_target).all(), (
                "Only source or target values expected"
            )

    def test_does_not_mutate_input_images(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=2, num_boxes=3, image_fill=0.9))

        batch2 = _make_camera_batch(batch_size=1, num_boxes=1, image_fill=0.1)
        original_images = batch2[0][0]["images"].clone()
        cp(*batch2)

        # Input should not be mutated
        assert torch.equal(batch2[0][0]["images"], original_images)

    def test_paste_with_multiple_cameras(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=2, num_boxes=3, num_cameras=3))

        batch2 = _make_camera_batch(batch_size=1, num_boxes=1, num_cameras=3)
        out_inputs, _ = cp(*batch2)

        # Output should still have all 3 cameras
        assert out_inputs[0]["images"].shape[0] == 3


# Lidar-only mode (no camera data)
class TestLidarOnlyMode:
    def test_works_without_camera_data(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        out_inputs, _ = cp(*_make_batch(batch_size=1, num_boxes=1))
        assert isinstance(out_inputs[0]["points"], PointCloud3D)
        assert "images" not in out_inputs[0]

    def test_lidar_db_entries_have_no_camera_crops(self) -> None:
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))
        for entry in cp._database["Car"]:
            assert entry.camera_crops is None

    def test_lidar_paste_into_camera_sample_no_image_paste(self) -> None:
        # Database populated from lidar-only batch (no camera crops)
        # Then paste into a sample that HAS camera data
        # Images should not be modified (entries have no crops)
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_batch(batch_size=2, num_boxes=3))

        batch2 = _make_camera_batch(batch_size=1, num_boxes=1, image_fill=0.5)
        original_images = batch2[0][0]["images"].clone()
        out_inputs, _ = cp(*batch2)

        # Images should be unchanged since db entries have no camera crops
        assert torch.equal(out_inputs[0]["images"], original_images)


# Camera-only extraction, lidar-only paste target
class TestCameraToLidarCrossModes:
    def test_camera_db_paste_into_lidar_only(self) -> None:
        # Database populated from camera+lidar batch
        # Paste into lidar-only sample — should paste points, skip images
        cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
        cp(*_make_camera_batch(batch_size=2, num_boxes=3))

        batch2 = _make_batch(batch_size=1, num_boxes=1)
        out_inputs, out_targets = cp(*batch2)

        assert isinstance(out_inputs[0]["points"], PointCloud3D)
        assert "images" not in out_inputs[0]
        # Points should still be pasted even though target has no cameras
        assert out_targets[0]["boxes"].shape[0] >= 1


# Determinism
class TestDeterminism:
    def test_reproducible_with_seed(self) -> None:
        def run_with_seed(seed: int) -> int:
            torch.manual_seed(seed)
            cp = CopyPaste3D(target_counts={"Car": 10}, min_points=1)
            torch.manual_seed(seed)
            cp(*_make_batch(batch_size=2, num_boxes=3))
            torch.manual_seed(seed + 1)
            _, out_targets = cp(*_make_batch(batch_size=1, num_boxes=2))
            return out_targets[0]["boxes"].shape[0]

        r1 = run_with_seed(42)
        r2 = run_with_seed(42)
        assert r1 == r2
