"""`nuScenes <https://www.nuscenes.org/>`_ Dataset."""

import os
from typing import Any, ClassVar, override

import numpy as np
import torch
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.eval.detection.utils import category_to_detection_name
from PIL import Image
from torch.utils.data import Dataset

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

# Camera ordering for consistent multi-camera tensor layout
CAMERA_NAMES: list[str] = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


class NuScenes3D(Dataset[tuple[dict[str, Any], dict[str, Any] | None]]):
    """`nuScenes <https://www.nuscenes.org/>`_ 3D object detection dataset.

    Returns samples in the **global frame** with annotations as
    :class:`BoundingBoxes3D` in ``XYZLWHY`` format (yaw extracted from
    quaternion). Multi-camera images, intrinsics, and extrinsics are
    returned for all 6 cameras.

    Requires the ``nuscenes-devkit`` package.

    Args:
        root (str or path): Root directory of the nuScenes dataset.
        version (str): Dataset version. Default: ``"v1.0-mini"``.
        split (str): One of ``"train"`` or ``"val"``. Default: ``"train"``.
        transforms (callable, optional): A function/transform that takes input
            sample and its target as entry and returns a transformed version.
    """

    camera_names: ClassVar[list[str]] = CAMERA_NAMES

    classes: ClassVar[list[str]] = list(DETECTION_NAMES)
    class_to_idx: ClassVar[dict[str, int]] = {name: i for i, name in enumerate(classes)}

    def __init__(
        self,
        root: str | os.PathLike[str],
        version: str = "v1.0-mini",
        split: str = "train",
        transforms: Any | None = None,
    ) -> None:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as e:
            msg = "nuscenes-devkit is required. Install with: uv sync --group nuscenes"
            raise ImportError(msg) from e

        self.root = str(root)
        self.version = version
        self.split = split
        self.transforms = transforms

        self._nusc = NuScenes(version=version, dataroot=self.root, verbose=False)

        # Collect sample tokens for the requested split
        split_scenes = _get_split_scenes(version, split)
        self._sample_tokens: list[str] = []
        for scene in self._nusc.scene:
            if scene["name"] in split_scenes:
                token = scene["first_sample_token"]
                while token:
                    self._sample_tokens.append(token)
                    sample = self._nusc.get("sample", token)
                    token = sample["next"]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._sample_tokens)

    @override
    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load a single sample.

        Args:
            index (int): Index.

        Returns:
            Tuple of ``(inputs, targets)``.

            **inputs** is a dict with keys:

            - ``"points"``: :class:`PointCloud3D` in lidar frame ``[N, 5]``
              (x, y, z, intensity, ring_index).
            - ``"images"``: :class:`CameraImages` ``[6, 3, H, W]``.
            - ``"extrinsics"``: :class:`CameraExtrinsics` ``[6, 4, 4]``
              (lidar-to-camera).
            - ``"intrinsics"``: :class:`CameraIntrinsics` ``[6, 3, 3]``.

            **targets** is a dict with keys:

            - ``"boxes"``: :class:`BoundingBoxes3D` in lidar frame,
              format ``XYZLWHY``.
            - ``"labels"``: :class:`torch.Tensor` of class indices.
        """
        sample = self._nusc.get("sample", self._sample_tokens[index])

        # Lidar
        lidar_data = self._nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        points = self._load_lidar(lidar_data)
        lidar_ego_pose = self._nusc.get("ego_pose", lidar_data["ego_pose_token"])
        lidar_calib = self._nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )

        # Transform from lidar to global
        lidar_to_global = _make_transform(
            lidar_ego_pose["translation"],
            lidar_ego_pose["rotation"],
        ) @ _make_transform(
            lidar_calib["translation"],
            lidar_calib["rotation"],
        )

        # Cameras
        images_list = []
        intrinsics_list = []
        extrinsics_list = []
        for cam_name in self.camera_names:
            cam_data = self._nusc.get("sample_data", sample["data"][cam_name])
            cam_calib = self._nusc.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )
            cam_ego_pose = self._nusc.get("ego_pose", cam_data["ego_pose_token"])

            # Camera image
            img = self._load_image(cam_data)
            images_list.append(img)

            # Intrinsics
            K = torch.tensor(cam_calib["camera_intrinsic"], dtype=torch.float32)
            intrinsics_list.append(K)

            # Extrinsics: lidar-to-camera
            cam_to_global = _make_transform(
                cam_ego_pose["translation"],
                cam_ego_pose["rotation"],
            ) @ _make_transform(
                cam_calib["translation"],
                cam_calib["rotation"],
            )
            lidar_to_cam = torch.linalg.inv(cam_to_global) @ lidar_to_global
            extrinsics_list.append(lidar_to_cam)

        inputs: dict[str, Any] = {
            "points": PointCloud3D(points),
            "images": CameraImages(torch.stack(images_list)),
            "extrinsics": CameraExtrinsics(torch.stack(extrinsics_list)),
            "intrinsics": CameraIntrinsics(torch.stack(intrinsics_list)),
        }

        # Annotations (in global frame -> convert to lidar frame)
        targets = self._load_annotations(sample, lidar_to_global)

        if self.transforms is not None:
            inputs, targets = self.transforms(inputs, targets)

        return inputs, targets

    def _load_lidar(self, lidar_data: dict[str, Any]) -> torch.Tensor:
        path = os.path.join(self.root, lidar_data["filename"])
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        return torch.from_numpy(points)

    def _load_image(self, cam_data: dict[str, Any]) -> torch.Tensor:
        path = os.path.join(self.root, cam_data["filename"])
        img = np.array(Image.open(path).convert("RGB"))
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _load_annotations(
        self, sample: dict[str, Any], lidar_to_global: torch.Tensor
    ) -> dict[str, Any]:
        """Load annotations and convert from global to lidar frame.

        Returns:
            Dict with ``"boxes"`` (:class:`BoundingBoxes3D`, XYZLWHY format),
            ``"labels"`` (int tensor).
        """
        global_to_lidar = torch.linalg.inv(lidar_to_global)

        label_ids: list[int] = []
        boxes: list[list[float]] = []

        for ann_token in sample["anns"]:
            ann = self._nusc.get("sample_annotation", ann_token)
            det_name = category_to_detection_name(ann["category_name"])
            if det_name is None:
                continue
            label_ids.append(self.class_to_idx[det_name])

            # Center: global -> lidar
            center_global = torch.tensor(
                [*ann["translation"], 1.0], dtype=torch.float32
            )
            center_lidar = (global_to_lidar @ center_global)[:3]

            # Dimensions: nuScenes stores (w, l, h), we want (l, w, h)
            w, l, h = ann["size"]

            # Rotation: quaternion -> yaw
            yaw = _quaternion_to_yaw(ann["rotation"], global_to_lidar[:3, :3])

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

        if not boxes:
            return {
                "boxes": BoundingBoxes3D(
                    torch.zeros(0, 7), format=BoundingBox3DFormat.XYZLWHY
                ),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        return {
            "boxes": BoundingBoxes3D(
                torch.tensor(boxes, dtype=torch.float32),
                format=BoundingBox3DFormat.XYZLWHY,
            ),
            "labels": torch.tensor(label_ids, dtype=torch.int64),
        }


def _make_transform(
    translation: list[float], rotation_wxyz: list[float]
) -> torch.Tensor:
    """Build a 4x4 transform from translation + quaternion (wxyz).

    Returns:
        ``[4, 4]`` homogeneous transform matrix.
    """
    from pyquaternion import Quaternion

    q = Quaternion(rotation_wxyz)
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = torch.tensor(q.rotation_matrix, dtype=torch.float32)
    T[:3, 3] = torch.tensor(translation, dtype=torch.float32)
    return T


def _quaternion_to_yaw(
    rotation_wxyz: list[float], global_to_lidar_rot: torch.Tensor
) -> float:
    """Convert a global-frame quaternion to yaw angle in lidar frame.

    Args:
        rotation_wxyz: Quaternion in wxyz format (global frame).
        global_to_lidar_rot: ``[3, 3]`` rotation from global to lidar.

    Returns:
        Yaw angle in radians.
    """
    from pyquaternion import Quaternion

    q = Quaternion(rotation_wxyz)
    # Forward vector in global frame
    forward_global = q.rotate(np.array([1.0, 0.0, 0.0]))
    # Rotate to lidar frame
    forward_lidar = (
        global_to_lidar_rot @ torch.tensor(forward_global, dtype=torch.float32)
    ).numpy()
    # Yaw = atan2(y, x) in lidar frame
    return float(np.arctan2(forward_lidar[1], forward_lidar[0]))


def _get_split_scenes(version: str, split: str) -> set[str]:
    """Get scene names for the given version and split.

    Uses the official split definitions from ``nuscenes.utils.splits``.

    Returns:
        Set of scene name strings.

    Raises:
        ValueError: If version/split combination is not supported.
    """
    from nuscenes.utils import splits

    version_to_splits: dict[str, dict[str, list[str]]] = {
        "v1.0-mini": {"train": splits.mini_train, "val": splits.mini_val},
        "v1.0-trainval": {"train": splits.train, "val": splits.val},
        "v1.0-test": {"test": splits.test},
    }

    if version not in version_to_splits:
        msg = f"Unsupported version: {version}"
        raise ValueError(msg)
    split_map = version_to_splits[version]
    if split not in split_map:
        msg = f"Unsupported split '{split}' for version '{version}'"
        raise ValueError(msg)
    return set(split_map[split])
