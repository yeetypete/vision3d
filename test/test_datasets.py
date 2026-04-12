from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from vision3d.datasets import Kitti3D
from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)


def _write_velodyne(path: Path, num_points: int = 50) -> np.ndarray:
    points = (
        np.random.default_rng(42).uniform(-20, 20, (num_points, 4)).astype(np.float32)
    )
    points[:, 3] = np.clip(points[:, 3], 0, 1)
    points.tofile(path)
    return points


# Real KITTI frame 000000 label
_KITTI_LABEL_000000 = "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n"

# Real KITTI frame 000000 calibration
_KITTI_CALIB_000000 = r"""P0: 7.070493e+02 0.000000e+00 6.040814e+02 0.000000e+00 0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.070493e+02 0.000000e+00 6.040814e+02 -3.797842e+02 0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.070493e+02 0.000000e+00 6.040814e+02 4.575831e+01 0.000000e+00 7.070493e+02 1.805066e+02 -3.454157e-01 0.000000e+00 0.000000e+00 1.000000e+00 4.981016e-03
P3: 7.070493e+02 0.000000e+00 6.040814e+02 -3.341081e+02 0.000000e+00 7.070493e+02 1.805066e+02 2.330660e+00 0.000000e+00 0.000000e+00 1.000000e+00 3.201153e-03
R0_rect: 9.999128e-01 1.009263e-02 -8.511932e-03 -1.012729e-02 9.999406e-01 -4.037671e-03 8.470675e-03 4.123522e-03 9.999556e-01
Tr_velo_to_cam: 6.927964e-03 -9.999722e-01 -2.757829e-03 -2.457729e-02 -1.162982e-03 2.749836e-03 -9.999955e-01 -6.127237e-02 9.999753e-01 6.931141e-03 -1.143899e-03 -3.321029e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01
"""


def _write_label(path: Path) -> None:
    path.write_text(_KITTI_LABEL_000000)


def _write_calib(path: Path) -> None:
    path.write_text(_KITTI_CALIB_000000)


def _write_image(path: Path, width: int = 800, height: int = 400) -> None:
    img = Image.fromarray(
        np.random.default_rng(42).integers(0, 255, (height, width, 3), dtype=np.uint8)
    )
    img.save(path)


@pytest.fixture
def kitti_root(tmp_path: Path) -> Path:
    # Matches Kitti3D._raw_folder: <root>/Kitti3D/raw/<split>/...
    raw = tmp_path / "Kitti3D" / "raw"
    for split in ["training", "testing"]:
        for subdir in ["velodyne", "calib", "image_2"]:
            (raw / split / subdir).mkdir(parents=True)

    (raw / "training" / "label_2").mkdir()

    for i in range(3):
        frame_id = f"{i:06d}"
        _write_velodyne(raw / "training" / "velodyne" / f"{frame_id}.bin")
        _write_label(raw / "training" / "label_2" / f"{frame_id}.txt")
        _write_calib(raw / "training" / "calib" / f"{frame_id}.txt")
        _write_image(raw / "training" / "image_2" / f"{frame_id}.png")

    for i in range(2):
        frame_id = f"{i:06d}"
        _write_velodyne(raw / "testing" / "velodyne" / f"{frame_id}.bin")
        _write_calib(raw / "testing" / "calib" / f"{frame_id}.txt")
        _write_image(raw / "testing" / "image_2" / f"{frame_id}.png")

    return tmp_path


class TestKitti3DConstruction:
    def test_train_split(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=True)
        assert len(ds) == 3

    def test_test_split(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=False)
        assert len(ds) == 2

    def test_missing_data_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            Kitti3D(tmp_path, train=True)

    def test_str_repr(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=True)
        assert isinstance(str(ds), str)
        assert isinstance(repr(ds), str)

    @pytest.mark.parametrize("train", [True, False])
    def test_num_examples(self, kitti_root: Path, train: bool) -> None:
        ds = Kitti3D(kitti_root, train=train)
        expected = 3 if train else 2
        assert len(ds) == expected
        assert all(ds[i] is not None for i in range(expected))


class TestKitti3DGetItem:
    @pytest.mark.parametrize("train", [True, False])
    def test_input_types(self, kitti_root: Path, train: bool) -> None:
        ds = Kitti3D(kitti_root, train=train)
        inputs, _ = ds[0]

        assert isinstance(inputs["points"], PointCloud3D)
        assert isinstance(inputs["images"], CameraImages)
        assert isinstance(inputs["extrinsics"], CameraExtrinsics)
        assert isinstance(inputs["intrinsics"], CameraIntrinsics)

    def test_target_types_train(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=True)
        _, targets = ds[0]

        assert targets is not None
        assert isinstance(targets["boxes"], BoundingBoxes3D)
        assert isinstance(targets["labels"], torch.Tensor)

    def test_target_none_test(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=False)
        _, targets = ds[0]
        assert targets is None

    @pytest.mark.parametrize("train", [True, False])
    def test_input_shapes(self, kitti_root: Path, train: bool) -> None:
        ds = Kitti3D(kitti_root, train=train)
        inputs, _ = ds[0]

        assert inputs["points"].ndim == 2
        assert inputs["points"].shape[1] == 4
        assert inputs["images"].shape[:2] == (1, 3)
        assert inputs["extrinsics"].shape == (1, 4, 4)
        assert inputs["intrinsics"].shape == (1, 3, 3)

    def test_boxes_format(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=True)
        _, targets = ds[0]
        assert targets is not None
        assert targets["boxes"].format == BoundingBox3DFormat.XYZLWHY

    def test_boxes_count(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=True)
        _, targets = ds[0]
        assert targets is not None
        assert targets["boxes"].shape[0] == 1  # 1 Pedestrian in frame 000000

    def test_boxes_are_in_lidar_frame(self, kitti_root: Path) -> None:
        ds = Kitti3D(kitti_root, train=True)
        _, targets = ds[0]
        assert targets is not None
        boxes = targets["boxes"]
        # Real frame 000000: Pedestrian at camera (1.84, 1.47, 8.41).
        # After conversion, lidar X should be roughly camera Z (~8.4).
        assert boxes[0, 0].abs() > 5.0

    def test_dontcare_filtered(self, kitti_root: Path) -> None:
        label_path = (
            kitti_root / "Kitti3D" / "raw" / "training" / "label_2" / "000000.txt"
        )
        content = label_path.read_text()
        content += "DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10\n"
        label_path.write_text(content)

        ds = Kitti3D(kitti_root, train=True)
        _, targets = ds[0]
        assert targets is not None
        assert targets["boxes"].shape[0] == 1  # DontCare not counted

    def test_empty_label(self, kitti_root: Path) -> None:
        label_path = (
            kitti_root / "Kitti3D" / "raw" / "training" / "label_2" / "000000.txt"
        )
        label_path.write_text("")

        ds = Kitti3D(kitti_root, train=True)
        _, targets = ds[0]
        assert targets is not None
        assert targets["boxes"].shape == (0, 7)
        assert targets["labels"].shape == (0,)


class TestKitti3DTransforms:
    def test_transforms_called(self, kitti_root: Path) -> None:
        called = False

        def my_transform(
            inputs: dict[str, Any], targets: dict[str, Any] | None
        ) -> tuple[dict[str, Any], dict[str, Any] | None]:
            nonlocal called
            called = True
            return inputs, targets

        ds = Kitti3D(kitti_root, train=True, transforms=my_transform)
        ds[0]
        assert called
