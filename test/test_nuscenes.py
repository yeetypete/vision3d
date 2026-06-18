"""Tests for the nuScenes dataset loader.

The parser must match the results produced by the official ``nuscenes-devkit``.

Tests that depend on the actual dataset payload are skipped unless the mini
split is available on disk.They may be pointed at a custom location with the
``NUSCENES_MINI_ROOT`` environment variable.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from nuscenes.eval.detection.constants import DETECTION_NAMES as DEVKIT_DETECTION_NAMES
from nuscenes.eval.detection.utils import (
    category_to_detection_name as devkit_category_to_detection_name,
)
from nuscenes.nuscenes import NuScenes as devkit_NuScenes
from nuscenes.utils import splits as devkit_splits
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from pyquaternion import Quaternion

from vision3d.datasets import NuScenes3D
from vision3d.datasets.nuscenes import (
    _CATEGORY_TO_DETECTION,
    _MINI_TRAIN,
    _MINI_VAL,
    _TEST,
    _TRAIN,
    _VAL,
    _category_to_detection_name,
    _NuScenesDB,
    _quaternion_to_rotation_matrix,
)

# Datasets return CPU tensors by convention.
pytestmark = pytest.mark.skip_device("cuda")


def test_detection_names_match_devkit() -> None:
    # Order matters: ``class_to_idx`` is order-dependent.
    assert NuScenes3D.classes == tuple(DEVKIT_DETECTION_NAMES)


def test_category_mapping_matches_devkit() -> None:
    # Every key/value in our inlined mapping must agree with the devkit.
    for cat, det in _CATEGORY_TO_DETECTION.items():
        assert devkit_category_to_detection_name(cat) == det


def test_category_mapping_unknown_returns_none() -> None:
    # Categories the devkit ignores (no detection class) must also map to
    # None in our wrapper.
    for cat in ("human.pedestrian.personal_mobility", "vehicle.emergency.ambulance"):
        assert devkit_category_to_detection_name(cat) is None
        assert _category_to_detection_name(cat) is None


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (_MINI_TRAIN, devkit_splits.mini_train),
        (_MINI_VAL, devkit_splits.mini_val),
        (_TRAIN, devkit_splits.train),
        (_VAL, devkit_splits.val),
        (_TEST, devkit_splits.test),
    ],
    ids=["mini_train", "mini_val", "train", "val", "test"],
)
def test_splits_match_devkit(ours: tuple[str, ...], theirs: list[str]) -> None:
    assert list(ours) == theirs


@given(
    components=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    )
)
def test_quaternion_to_rotation_matrix_matches_pyquaternion(
    components: list[float],
) -> None:
    q = np.array(components)
    norm = float(np.linalg.norm(q))
    # A near-zero vector has no well-defined rotation; skip it.
    assume(norm > 1e-6)
    q = q / norm  # unit quaternion in wxyz order
    ours = _quaternion_to_rotation_matrix(q.tolist())
    ref = Quaternion(q.tolist()).rotation_matrix
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def _default_mini_root() -> Path:
    env = os.environ.get("NUSCENES_MINI_ROOT")
    if env:
        return Path(env)
    return Path("~/.cache/vision3d/nuscenes-mini").expanduser()


@pytest.fixture(scope="module")
def mini_root() -> Path:
    root = _default_mini_root()
    if not (root / "v1.0-mini").is_dir():
        pytest.skip(
            "nuScenes mini split not found. Set NUSCENES_MINI_ROOT to a "
            "directory containing v1.0-mini/, or download it via "
            "NuScenes3D(root, download=True)."
        )
    return root


@pytest.fixture(scope="module")
def devkit_db(mini_root: Path) -> devkit_NuScenes:
    return devkit_NuScenes(version="v1.0-mini", dataroot=str(mini_root), verbose=False)


@pytest.fixture(scope="module")
def our_db(mini_root: Path) -> _NuScenesDB:
    return _NuScenesDB(dataroot=mini_root, version="v1.0-mini")


@pytest.fixture(scope="module")
def datasets(mini_root: Path) -> dict[str, NuScenes3D]:
    return {
        split: NuScenes3D(mini_root, version="v1.0-mini", split=split)
        for split in ("train", "val")
    }


@pytest.mark.parametrize(
    "table",
    [
        "category",
        "instance",
        "sensor",
        "calibrated_sensor",
        "ego_pose",
        "scene",
        "sample",
        "sample_data",
        "sample_annotation",
    ],
)
def test_db_tables_match(
    table: str, our_db: _NuScenesDB, devkit_db: devkit_NuScenes
) -> None:
    """Tables we load are identical lists of records to the devkit's."""
    ours = getattr(our_db, table)
    theirs = getattr(devkit_db, table)
    assert len(ours) == len(theirs)
    for rec, ref in zip(ours, theirs, strict=True):
        assert rec == ref, f"{table}[{rec['token']}]: {rec!r} != {ref!r}"


def test_db_sample_decorations_match(
    our_db: _NuScenesDB, devkit_db: devkit_NuScenes
) -> None:
    """Per-sample ``data``/``anns`` shortcuts mirror the devkit exactly."""
    for ours in our_db.sample:
        ref = devkit_db.get("sample", ours["token"])
        assert ours["data"] == ref["data"]
        assert ours["anns"] == ref["anns"]


def test_db_annotation_category_names_match(
    our_db: _NuScenesDB, devkit_db: devkit_NuScenes
) -> None:
    """Every annotation's joined ``category_name`` agrees with the devkit."""
    for ours in our_db.sample_annotation:
        ref = devkit_db.get("sample_annotation", ours["token"])
        assert ours["category_name"] == ref["category_name"]


def _devkit_make_transform(
    translation: list[float], rotation_wxyz: list[float]
) -> torch.Tensor:
    quaternion = Quaternion(rotation_wxyz)
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = torch.tensor(quaternion.rotation_matrix, dtype=torch.float32)
    T[:3, 3] = torch.tensor(translation, dtype=torch.float32)
    return T


def _devkit_sample_outputs(
    nusc: devkit_NuScenes, sample_token: str, root: Path
) -> dict[str, torch.Tensor]:
    # Reproduce ``NuScenes3D.__getitem__`` using the devkit directly so we
    # can compare every tensor field 1:1.

    class_to_idx = {n: i for i, n in enumerate(DEVKIT_DETECTION_NAMES)}
    sample = nusc.get("sample", sample_token)
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    points = torch.from_numpy(
        np.fromfile(root / lidar_data["filename"], dtype=np.float32).reshape(-1, 5)
    )

    lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    lidar_calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    lidar_to_global = _devkit_make_transform(
        lidar_ego_pose["translation"], lidar_ego_pose["rotation"]
    ) @ _devkit_make_transform(lidar_calib["translation"], lidar_calib["rotation"])

    images_list: list[torch.Tensor] = []
    intrinsics_list: list[torch.Tensor] = []
    extrinsics_list: list[torch.Tensor] = []
    for cam_name in NuScenes3D.camera_names:
        cam_data = nusc.get("sample_data", sample["data"][cam_name])
        cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        cam_ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
        img = np.array(Image.open(root / cam_data["filename"]).convert("RGB"))
        images_list.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        K = torch.tensor(cam_calib["camera_intrinsic"], dtype=torch.float32)
        intrinsics_list.append(K)
        cam_to_global = _devkit_make_transform(
            cam_ego_pose["translation"], cam_ego_pose["rotation"]
        ) @ _devkit_make_transform(cam_calib["translation"], cam_calib["rotation"])
        lidar_to_cam = torch.linalg.inv(cam_to_global) @ lidar_to_global
        extrinsics_list.append(lidar_to_cam)

    global_to_lidar = torch.linalg.inv(lidar_to_global)
    boxes: list[list[float]] = []
    label_ids: list[int] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = devkit_category_to_detection_name(ann["category_name"])
        if det_name is None:
            continue
        label_ids.append(class_to_idx[det_name])
        center_global = torch.tensor([*ann["translation"], 1.0], dtype=torch.float32)
        center_lidar = (global_to_lidar @ center_global)[:3]
        w, l, h = ann["size"]
        quaternion = Quaternion(ann["rotation"])
        forward_global = quaternion.rotate(np.array([1.0, 0.0, 0.0]))
        forward_lidar = (
            global_to_lidar[:3, :3] @ torch.tensor(forward_global, dtype=torch.float32)
        ).numpy()
        yaw = float(np.arctan2(forward_lidar[1], forward_lidar[0]))
        boxes.append(
            [
                center_lidar[0].item(),
                center_lidar[1].item(),
                center_lidar[2].item(),
                l,
                w,
                h,
                yaw,
            ]
        )

    return {
        "points": points,
        "images": torch.stack(images_list),
        "extrinsics": torch.stack(extrinsics_list),
        "intrinsics": torch.stack(intrinsics_list),
        "boxes": (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 7)
        ),
        "labels": torch.tensor(label_ids, dtype=torch.int64),
    }


@pytest.mark.parametrize("split", ["train", "val"])
@settings(
    max_examples=50,
    deadline=None,
)
@given(data=st.data())
def test_nuscenes3d_outputs_match_devkit(
    split: str,
    mini_root: Path,
    devkit_db: devkit_NuScenes,
    datasets: dict[str, NuScenes3D],
    data: st.DataObject,
) -> None:
    """``NuScenes3D.__getitem__`` must match a devkit-driven reference."""
    ds = datasets[split]
    index = data.draw(st.integers(min_value=0, max_value=len(ds) - 1))
    inputs, targets = ds[index]
    ref = _devkit_sample_outputs(devkit_db, ds._sample_tokens[index], mini_root)
    assert torch.equal(inputs["points"], ref["points"])
    assert torch.equal(inputs["images"], ref["images"])
    torch.testing.assert_close(
        inputs["extrinsics"], ref["extrinsics"], atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        inputs["intrinsics"], ref["intrinsics"], atol=1e-6, rtol=0
    )
    assert torch.equal(targets["labels"], ref["labels"])
    torch.testing.assert_close(targets["boxes"], ref["boxes"], atol=1e-5, rtol=0)


def _devkit_multisweep(
    nusc: devkit_NuScenes, sample_token: str, num_sweeps: int
) -> torch.Tensor:
    # Accumulate sweeps with the nuscenes-devkit. ``min_distance=0`` disables
    # its close-point removal (which our loader does not do) so the point clouds
    # match. The devkit keeps (x, y, z, intensity) and a separate time vector.
    # it drops the ring column.
    # TODO: provide close-point removal as a reusable distance-filter transform
    # rather than baking it into NuScenes3D, so ego self-returns can be filtered
    # like the devkit.
    sample = nusc.get("sample", sample_token)
    pc, times = LidarPointCloud.from_file_multisweep(
        nusc,
        sample,
        chan="LIDAR_TOP",
        ref_chan="LIDAR_TOP",
        nsweeps=num_sweeps,
        min_distance=0.0,
    )
    return torch.cat(
        [torch.from_numpy(pc.points.T), torch.from_numpy(times.T)], dim=1
    ).float()


def _drop_ring(points: torch.Tensor) -> torch.Tensor:
    # Our cloud is (x, y, z, intensity, ring, time). Drop ring to match the
    # devkit's (x, y, z, intensity, time) layout.
    return torch.cat([points[:, :4], points[:, 5:]], dim=1)


@pytest.mark.parametrize("num_sweeps", [3, 10])
def test_nuscenes3d_sweeps_match_devkit(
    mini_root: Path, devkit_db: devkit_NuScenes, num_sweeps: int
) -> None:
    """Aggregated multi-sweep clouds match the devkit ``from_file_multisweep``."""
    ds = NuScenes3D(
        mini_root, version="v1.0-mini", split="train", num_sweeps=num_sweeps
    )
    single = NuScenes3D(mini_root, version="v1.0-mini", split="train")
    for index in (0, len(ds) // 2, len(ds) - 1):
        points = ds[index][0]["points"]
        ref = _devkit_multisweep(devkit_db, ds._sample_tokens[index], num_sweeps)
        assert points.shape[1] == 6
        # Aggregating sweeps only adds points to the key-frame.
        assert points.shape[0] >= single[index][0]["points"].shape[0]
        torch.testing.assert_close(_drop_ring(points), ref, atol=1e-3, rtol=0)
        # Key-frame points come first and carry a zero time offset.
        assert points[: single[index][0]["points"].shape[0], -1].abs().max() == 0.0


def test_nuscenes3d_sweeps_scene_start(
    mini_root: Path, devkit_db: devkit_NuScenes
) -> None:
    """Requesting more sweeps than a scene provides falls back to the sweeps
    that are available."""
    ds = NuScenes3D(mini_root, version="v1.0-mini", split="train", num_sweeps=10)
    single = NuScenes3D(mini_root, version="v1.0-mini", split="train")
    # ``_sample_tokens`` is built scene by scene, so index 0 is a scene start.
    token = ds._sample_tokens[0]
    lidar_data = devkit_db.get(
        "sample_data", devkit_db.get("sample", token)["data"]["LIDAR_TOP"]
    )
    assert lidar_data["prev"] == "", "expected index 0 to be a scene start"

    points = ds[0][0]["points"]
    # With no previous sweeps, the result is exactly the single-frame key-frame.
    assert points.shape[0] == single[0][0]["points"].shape[0]
    assert points[:, -1].abs().max() == 0.0
    torch.testing.assert_close(
        _drop_ring(points), _devkit_multisweep(devkit_db, token, 10), atol=1e-3, rtol=0
    )
