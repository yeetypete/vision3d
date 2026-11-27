"""Tests for the nuScenes dataset loader.

The parser must match the results produced by the official ``nuscenes-devkit``.

Tests that depend on the actual dataset payload are skipped unless the mini
split is available on disk.They may be pointed at a custom location with the
``NUSCENES_MINI_ROOT`` environment variable.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from nuscenes.eval.detection.constants import DETECTION_NAMES as DEVKIT_DETECTION_NAMES
from nuscenes.eval.detection.utils import (
    category_to_detection_name as devkit_category_to_detection_name,
)
from nuscenes.nuscenes import NuScenes as devkit_NuScenes
from nuscenes.utils import splits as devkit_splits
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image, ImageFilter
from pyquaternion import Quaternion

from vision3d.datasets import NuScenes3D
from vision3d.datasets.nuscenes import (
    _MAP_LINE_LAYERS,
    _MINI_TRAIN,
    _MINI_VAL,
    _TEST,
    _TRAIN,
    _VAL,
    _get_split_scenes,
    _NuScenesDB,
    _NuScenesMap,
)
from vision3d.tensors import GridMask3D
from vision3d.transforms import ClosePointFilter

# Datasets return CPU tensors by convention.
pytestmark = pytest.mark.skip_device("cuda")


def test_detection_names_match_devkit() -> None:
    # Order matters: ``class_to_idx`` is order-dependent.
    assert NuScenes3D.classes == tuple(DEVKIT_DETECTION_NAMES)


def test_category_mapping_matches_devkit() -> None:
    # Every key/value in our inlined mapping must agree with the devkit.
    for cat, det in NuScenes3D.category_map.items():
        assert devkit_category_to_detection_name(cat) == det


def test_category_mapping_unknown_returns_none() -> None:
    # Categories the devkit ignores (no detection class) must also map to
    # None in our wrapper.
    for cat in ("human.pedestrian.personal_mobility", "vehicle.emergency.ambulance"):
        assert devkit_category_to_detection_name(cat) is None
        assert NuScenes3D.category_map.get(cat) is None


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


@pytest.mark.parametrize("version", ["v1.0-mini", "v1.0-custom"])
def test_split_all_skips_scene_filtering(version: str) -> None:
    # ``None`` means "keep every scene". Unofficial versions are accepted too.
    assert _get_split_scenes(version, "all") is None


def test_unsupported_split_error_offers_all() -> None:
    with pytest.raises(ValueError, match="'all'"):
        _get_split_scenes("v1.0-test", "train")


def test_unavailable_target_keys_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not available"):
        NuScenes3D(tmp_path, target_keys={"boxes", "masks"})


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
    ("target_keys", "keys"),
    [
        (None, {"boxes", "labels"}),
        ({"boxes"}, {"boxes"}),
        ((), set()),
    ],
)
def test_target_keys_select_targets(
    mini_root: Path, target_keys: Any, keys: set[str]
) -> None:
    dataset = NuScenes3D(mini_root, version="v1.0-mini", target_keys=target_keys)
    _, targets = dataset[0]
    assert targets.keys() == keys


def test_split_all_is_every_scene(
    mini_root: Path, datasets: dict[str, NuScenes3D]
) -> None:
    """``all`` covers the union of the official splits."""
    everything = NuScenes3D(mini_root, version="v1.0-mini", split="all")
    assert len(everything) == len(datasets["train"]) + len(datasets["val"])
    assert set(everything._sample_tokens) == set(
        datasets["train"]._sample_tokens
    ) | set(datasets["val"]._sample_tokens)


@pytest.mark.parametrize(
    "table",
    [
        "category",
        "instance",
        "sensor",
        "calibrated_sensor",
        "ego_pose",
        "log",
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
        # ``map_token`` is a legacy map format. We use the vector maps in
        # ``maps/expansion`` instead.
        ref = {k: v for k, v in ref.items() if k != "map_token"}
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


def _devkit_sensor_to_global(
    nusc: devkit_NuScenes, sample_data: dict[str, Any]
) -> np.ndarray:
    pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    return transform_matrix(
        pose["translation"], Quaternion(pose["rotation"])
    ) @ transform_matrix(calib["translation"], Quaternion(calib["rotation"]))


def _devkit_lidar_to_global(nusc: devkit_NuScenes, sample_token: str) -> np.ndarray:
    sample = nusc.get("sample", sample_token)
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return _devkit_sensor_to_global(nusc, lidar_data)


def _devkit_sample_outputs(
    nusc: devkit_NuScenes, sample_token: str
) -> dict[str, torch.Tensor]:
    # Reproduce ``NuScenes3D.__getitem__`` with the devkit so that every
    # tensor field can be compared.
    class_to_idx = {n: i for i, n in enumerate(DEVKIT_DETECTION_NAMES)}
    sample = nusc.get("sample", sample_token)
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    # ``get_sample_data`` returns the annotations as boxes in the lidar frame.
    lidar_path, lidar_boxes, _ = nusc.get_sample_data(lidar_data["token"])
    points = torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5))
    lidar_to_global = _devkit_sensor_to_global(nusc, lidar_data)

    images_list: list[torch.Tensor] = []
    intrinsics_list: list[torch.Tensor] = []
    extrinsics_list: list[torch.Tensor] = []
    for cam_name in NuScenes3D.camera_names:
        cam_data = nusc.get("sample_data", sample["data"][cam_name])
        cam_path, _, K = nusc.get_sample_data(cam_data["token"], selected_anntokens=[])
        img = np.array(Image.open(cam_path).convert("RGB"))
        images_list.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        intrinsics_list.append(torch.from_numpy(K).float())
        cam_to_global = _devkit_sensor_to_global(nusc, cam_data)
        lidar_to_cam = np.linalg.inv(cam_to_global) @ lidar_to_global
        extrinsics_list.append(torch.from_numpy(lidar_to_cam).float())

    boxes: list[list[float]] = []
    label_ids: list[int] = []
    for box in lidar_boxes:
        det_name = devkit_category_to_detection_name(box.name)
        if det_name is None:
            continue
        label_ids.append(class_to_idx[det_name])
        w, l, h = box.wlh
        forward = box.orientation.rotation_matrix[:, 0]
        yaw = np.arctan2(forward[1], forward[0])
        boxes.append([*box.center, l, w, h, yaw])

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
    devkit_db: devkit_NuScenes,
    datasets: dict[str, NuScenes3D],
    data: st.DataObject,
) -> None:
    """``NuScenes3D.__getitem__`` must match a devkit-driven reference."""
    ds = datasets[split]
    index = data.draw(st.integers(min_value=0, max_value=len(ds) - 1))
    inputs, targets = ds[index]
    ref = _devkit_sample_outputs(devkit_db, ds._sample_tokens[index])
    assert torch.equal(inputs["points"], ref["points"])
    assert torch.equal(inputs["images"], ref["images"])
    # Test with reduced precision because NuScenes3D computes poses in float32
    # and the devkit in float64.
    torch.testing.assert_close(
        inputs["extrinsics"], ref["extrinsics"], atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        inputs["intrinsics"], ref["intrinsics"], atol=1e-6, rtol=0
    )
    assert torch.equal(targets["labels"], ref["labels"])
    torch.testing.assert_close(targets["boxes"], ref["boxes"], atol=1e-3, rtol=0)


def _devkit_multisweep(
    nusc: devkit_NuScenes,
    sample_token: str,
    num_sweeps: int,
    min_distance: float = 0.0,
) -> torch.Tensor:
    # Accumulate sweeps with the nuscenes-devkit. The devkit keeps
    # (x, y, z, intensity) and a separate time vector; it drops the ring column.
    sample = nusc.get("sample", sample_token)
    pc, times = LidarPointCloud.from_file_multisweep(
        nusc,
        sample,
        chan="LIDAR_TOP",
        ref_chan="LIDAR_TOP",
        nsweeps=num_sweeps,
        min_distance=min_distance,
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
        assert torch.all(points[: single[index][0]["points"].shape[0], -1] == 0)


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
    assert torch.all(points[:, -1] == 0)
    torch.testing.assert_close(
        _drop_ring(points), _devkit_multisweep(devkit_db, token, 10), atol=1e-3, rtol=0
    )


def test_nuscenes3d_close_point_filter_matches_devkit(
    mini_root: Path, devkit_db: devkit_NuScenes
) -> None:
    """ClosePointFilter matches devkit removal for a single lidar sweep."""
    ds = NuScenes3D(
        mini_root,
        version="v1.0-mini",
        split="train",
        transforms=ClosePointFilter(),
    )
    index = len(ds) // 2
    points = ds[index][0]["points"]
    ref = _devkit_multisweep(
        devkit_db, ds._sample_tokens[index], num_sweeps=1, min_distance=1.0
    )

    # A single-sweep dataset cloud has (x, y, z, intensity, ring), while the
    # devkit result has (x, y, z, intensity, time). Compare their shared data.
    torch.testing.assert_close(points[:, :4], ref[:, :4], atol=1e-6, rtol=0)


def _write_map(path: Path) -> Path:
    """Write a tiny map expansion with one location's worth of layers.

    The drivable area is a 10x10 square with a 6x6 hole, and a second polygon
    fills the center of the hole. A road divider runs along ``y = 15``.

    Returns:
        ``path``.
    """
    nodes = {
        "outer": [(0, 0), (10, 0), (10, 10), (0, 10)],
        "hole": [(2, 2), (8, 2), (8, 8), (2, 8)],
        "island": [(4, 4), (6, 4), (6, 6), (4, 6)],
        "line": [(0, 15), (10, 15)],
    }
    node_records: list[dict[str, Any]] = []
    tokens: dict[str, list[str]] = {}
    for name, coords in nodes.items():
        tokens[name] = [f"{name}{i}" for i in range(len(coords))]
        node_records += [
            {"token": t, "x": float(x), "y": float(y)}
            for t, (x, y) in zip(tokens[name], coords, strict=True)
        ]
    expansion: dict[str, Any] = {
        "version": "1.3",
        "node": node_records,
        "polygon": [
            {
                "token": "p_outer",
                "exterior_node_tokens": tokens["outer"],
                "holes": [{"node_tokens": tokens["hole"]}],
            },
            {
                "token": "p_island",
                "exterior_node_tokens": tokens["island"],
                "holes": [],
            },
        ],
        "line": [{"token": "l0", "node_tokens": tokens["line"]}],
        # List the island first on purpose so we test that the outer polygon's
        # hole cannot erase it.
        "drivable_area": [{"token": "d0", "polygon_tokens": ["p_island", "p_outer"]}],
        "road_divider": [{"token": "r0", "line_token": "l0"}],
    }
    path.write_text(json.dumps(expansion))
    return path


_IDENTITY_CELLS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_PATCH = np.array([-0.5, -0.5, 19.5, 19.5])


def test_map_rasterizes_polygons_and_lines(tmp_path: Path) -> None:
    layers = ["drivable_area", "road_divider"]
    nusc_map = _NuScenesMap(_write_map(tmp_path / "map.json"), layers)
    area, divider = (
        nusc_map.rasterize(name, _IDENTITY_CELLS, (20, 20), _PATCH) for name in layers
    )
    # Masks are indexed [row, column] = [y, x].
    assert area[1, 1]  # Ring between exterior and hole.
    assert not area[3, 3]  # Inside the hole.
    assert area[5, 5]  # Island inside the hole survives the hole.
    assert not area[12, 5]  # Outside the polygon.
    # A 3-cell stroke covers one row on each side of the line.
    assert divider[14:17, 0:11].all()
    assert not divider[12, 5]
    assert not divider[18, 5]


def test_map_rejects_old_expansion(tmp_path: Path) -> None:
    path = _write_map(tmp_path / "map.json")
    expansion = json.loads(path.read_text())
    path.write_text(json.dumps(expansion | {"version": "1.2"}))
    with pytest.raises(ValueError, match="or newer is required"):
        _NuScenesMap(path, ["drivable_area"])


def test_map_compares_versions_numerically(tmp_path: Path) -> None:
    path = _write_map(tmp_path / "map.json")
    expansion = json.loads(path.read_text())
    path.write_text(json.dumps(expansion | {"version": "1.10"}))
    _NuScenesMap(path, ["drivable_area"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"map_resolution": 0.0}, "map_resolution"),
        ({"map_size": (100.0,)}, "two extents"),
        ({"map_size": (0.1, 100.0)}, "at least one cell"),
    ],
)
def test_map_args_are_validated(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        NuScenes3D(tmp_path, **kwargs)


def test_map_missing_expansion_raises(mini_root: Path, tmp_path: Path) -> None:
    (tmp_path / "v1.0-mini").symlink_to(mini_root / "v1.0-mini")
    with pytest.raises(FileNotFoundError, match="map expansion"):
        NuScenes3D(tmp_path, target_keys="all")
    # Without the map target the expansion is not needed.
    NuScenes3D(tmp_path)


@pytest.fixture(scope="module")
def map_dataset(mini_root: Path) -> NuScenes3D:
    if not (mini_root / "maps" / "expansion").is_dir():
        pytest.skip(
            "nuScenes map expansion not found. Download it via "
            "NuScenes3D(root, target_keys='all', download=True)."
        )
    return NuScenes3D(mini_root, version="v1.0-mini", split="all", target_keys="all")


@pytest.fixture(scope="module")
def devkit_maps(mini_root: Path, map_dataset: NuScenes3D) -> dict[str, Any]:
    from nuscenes.map_expansion.map_api import NuScenesMap

    return {loc: NuScenesMap(str(mini_root), loc) for loc in map_dataset._maps}


@pytest.fixture(scope="module")
def devkit_invalid_polygons(
    devkit_maps: dict[str, Any],
) -> defaultdict[tuple[str, str], list[Any]]:
    # The devkit incorrectly drops polygons that shapely considers invalid.
    # They are real map features, which we draw.
    invalid: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    for location, nusc_map in devkit_maps.items():
        for name in nusc_map.non_geometric_polygon_layers:
            for record in getattr(nusc_map, name):
                tokens = record.get("polygon_tokens", [record.get("polygon_token")])
                polygons = (nusc_map.extract_polygon(t) for t in tokens)
                invalid[location, name].extend(p for p in polygons if not p.is_valid)
    return invalid


def _devkit_location(nusc: devkit_NuScenes, sample_token: str) -> str:
    scene = nusc.get("scene", nusc.get("sample", sample_token)["scene_token"])
    return nusc.get("log", scene["log_token"])["location"]


def _devkit_patch(
    lidar_to_global: np.ndarray, shift: float = 0.0
) -> tuple[tuple[float, float, float, float], float]:
    # A 100 m patch along the lidar heading, as BEVFusion requests it, with the
    # center moved by ``shift`` meters along both patch axes.
    yaw = np.arctan2(lidar_to_global[1, 0], lidar_to_global[0, 0])
    cos, sin = np.cos(yaw), np.sin(yaw)
    x, y = lidar_to_global[:2, 3] + np.array([[cos, -sin], [sin, cos]]) @ [shift, shift]
    return (x, y, 100.0, 100.0), np.degrees(yaw)


def _devkit_map_mask(
    nusc_map: Any, lidar_to_global: np.ndarray, shift: float, layers: list[str]
) -> np.ndarray:
    # Rasterized at 0.5 m per cell.
    mask = nusc_map.get_map_mask(
        *_devkit_patch(lidar_to_global, shift), layers, (200, 200)
    )
    # The devkit puts rows along y. Ours run along x, as in BEVFusion.
    return mask.transpose(0, 2, 1).astype(bool)


# PIL and cv2 fill polygon edges and stroke lines slightly differently, so
# masks may disagree by one cell along edges. Allow some stray cells beyond
# that.
_MAX_STRAY_CELLS = 10


def _dilate(mask: np.ndarray) -> np.ndarray:
    # Grow the mask by one cell, diagonals included.
    return np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(3)))


def test_map_matches_devkit(
    map_dataset: NuScenes3D,
    devkit_db: devkit_NuScenes,
    devkit_maps: dict[str, Any],
    devkit_invalid_polygons: defaultdict[tuple[str, str], list[Any]],
) -> None:
    """Map channels match ``NuScenesMap.get_map_mask`` as BEVFusion calls it."""
    from nuscenes.map_expansion.map_api import NuScenesMapExplorer

    layers = map_dataset.map_layers
    line_layers = [name for name in layers if name in _MAP_LINE_LAYERS]
    polygon_layers = [name for name in layers if name not in _MAP_LINE_LAYERS]
    half_cell = map_dataset.map_resolution / 2
    checked: set[str] = set()
    for index in range(0, len(map_dataset), 40):
        token = map_dataset._sample_tokens[index]
        grid = map_dataset[index][1]["map"]
        assert isinstance(grid, GridMask3D)
        assert grid.shape == (len(layers), 200, 200)
        assert grid.dtype == torch.bool
        grid = grid.numpy()
        lidar_to_global = _devkit_lidar_to_global(devkit_db, token)
        location = _devkit_location(devkit_db, token)
        nusc_map = devkit_maps[location]

        # The devkit rounds polygon vertices to integer pixels, which are half
        # a cell off our cell centers. We shift its polygon patch by half a
        # cell to compare. It truncates line vertices instead, which cancels
        # that offset.
        refs = dict(
            zip(
                polygon_layers,
                _devkit_map_mask(nusc_map, lidar_to_global, half_cell, polygon_layers),
                strict=True,
            )
        )
        refs.update(
            zip(
                line_layers,
                _devkit_map_mask(nusc_map, lidar_to_global, 0.0, line_layers),
                strict=True,
            )
        )
        patch = NuScenesMapExplorer.get_patch_coord(*_devkit_patch(lidar_to_global))
        for channel, name in enumerate(layers):
            # Skip layers where the devkit drops an invalid polygon in the patch.
            invalid = devkit_invalid_polygons[location, name]
            if any(polygon.intersects(patch) for polygon in invalid):
                continue
            ours, theirs = grid[channel], refs[name]
            assert (ours & ~_dilate(theirs)).sum() <= _MAX_STRAY_CELLS, (index, name)
            assert (theirs & ~_dilate(ours)).sum() <= _MAX_STRAY_CELLS, (index, name)
            checked.add(name)
    assert checked == set(layers)


# Sampled cells may disagree with the devkit only on polygon edges.
_MIN_CELL_AGREEMENT = 0.95


def test_map_affine_places_cells_on_map(
    mini_root: Path, devkit_db: devkit_NuScenes, devkit_maps: dict[str, Any]
) -> None:
    """Cells mapped through the affine land on the map geometry they label."""
    # A non-square patch catches swapped rows and columns.
    dataset = NuScenes3D(
        mini_root,
        version="v1.0-mini",
        split="all",
        target_keys={"map"},
        map_size=(100.0, 60.0),
    )
    channel = dataset.map_layers.index("drivable_area")
    rng = np.random.default_rng(0)
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        token = dataset._sample_tokens[index]
        grid = dataset[index][1]["map"]
        assert grid.shape[1:] == (200, 120)
        lidar_to_global = _devkit_lidar_to_global(devkit_db, token)
        cells = np.indices((1, *grid.shape[1:])).reshape(3, -1)
        cells = np.vstack([cells, np.ones(cells.shape[1])])
        centers = lidar_to_global @ grid.affine.numpy() @ cells
        # Cells lie in the global ground plane, centered below the lidar.
        np.testing.assert_allclose(centers[2], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            centers[:2].mean(axis=1), lidar_to_global[:2, 3], atol=1e-6
        )
        # The grid projects along the global vertical.
        d_axis = (lidar_to_global @ grid.affine.numpy())[:3, 0]
        np.testing.assert_allclose(
            d_axis / np.linalg.norm(d_axis), [0.0, 0.0, 1.0], atol=1e-6
        )

        nusc_map = devkit_maps[_devkit_location(devkit_db, token)]
        ours = grid[channel].numpy().ravel()
        picked = rng.choice(ours.size, size=300, replace=False)
        inside = np.array(
            [
                bool(nusc_map.record_on_point(*centers[:2, i], "drivable_area"))
                for i in picked
            ]
        )
        assert np.mean(ours[picked] == inside) > _MIN_CELL_AGREEMENT
