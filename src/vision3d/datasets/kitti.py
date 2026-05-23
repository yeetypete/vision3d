"""`KITTI 3D Object Detection <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_ Dataset."""

import csv
import io
import os
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, override

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.io import ImageReadMode, decode_image

from vision3d.datasets import FusionInputs, SampleTargets
from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)


class Kitti3D(Dataset[tuple[FusionInputs, SampleTargets | None]]):
    """`KITTI 3D <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_ Dataset.

    Returns samples in **lidar frame** (X-forward, Y-left, Z-up), converting
    from KITTI's camera-frame annotations automatically.

    Args:
        root (str or ``pathlib.Path``): Root directory where data is downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti3D/
                        └── raw/
                            ├── training/
                            |   ├── velodyne/
                            |   ├── label_2/
                            |   ├── calib/
                            |   └── image_2/
                            └── testing/
                                ├── velodyne/
                                ├── calib/
                                └── image_2/

        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``True``.
        transforms (Callable, optional): A function/transform that takes input
            sample and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet
            and puts it in root directory. If dataset is already downloaded, it is
            not downloaded again.
        mini (bool, optional): If true, downloading fetches only the frames
            named in ``frames`` via HTTP range requests against the
            upstream archives instead of the full dataset. Has no effect
            when ``download`` is ``False``.
        frames (Iterable[int], optional): Frame indices to retrieve
            when ``mini`` is true. Each ``i`` is formatted as
            ``f"{i:06d}"`` and pulled from the requested split; the
            iterable may be contiguous (``range(10)``), strided
            (``range(0, 7481, 700)``), or scattered (``[0, 23, 412]``).
            Defaults to ``range(10)``.
    """

    data_url: ClassVar[str] = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources: ClassVar[list[str]] = [
        "data_object_velodyne.zip",
        "data_object_image_2.zip",
        "data_object_label_2.zip",
        "data_object_calib.zip",
    ]
    velodyne_dir_name: ClassVar[str] = "velodyne"
    image_dir_name: ClassVar[str] = "image_2"
    labels_dir_name: ClassVar[str] = "label_2"
    calib_dir_name: ClassVar[str] = "calib"

    # Single forward-facing left color camera ("P2" in KITTI's calibration).
    camera_names: ClassVar[tuple[str, ...]] = ("CAM_2",)
    camera_grid: ClassVar[tuple[tuple[int, ...], ...] | None] = None

    classes: ClassVar[tuple[str, ...]] = (
        "Car",
        "Pedestrian",
        "Cyclist",
        "Van",
        "Truck",
        "Person_sitting",
        "Tram",
        "Misc",
    )
    class_to_idx: ClassVar[dict[str, int]] = {name: i for i, name in enumerate(classes)}

    def __init__(
        self,
        root: str | os.PathLike[str],
        train: bool = True,
        transforms: Any | None = None,
        download: bool = False,
        mini: bool = False,
        frames: Iterable[int] = range(10),
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transforms = transforms
        self.mini = mini
        self.frames = tuple(frames)
        self._location = "training" if train else "testing"

        if download:
            if mini:
                self._download_mini(self.frames)
            else:
                self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You may use download=True to download it."
            )

        velodyne_dir = self._raw_folder / self._location / self.velodyne_dir_name
        self._frame_ids = sorted(
            p.stem for p in velodyne_dir.iterdir() if p.suffix == ".bin"
        )

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self._frame_ids)

    @override
    def __getitem__(self, index: int) -> tuple[FusionInputs, SampleTargets | None]:
        """Load a single frame.

        Args:
            index (int): Index.

        Returns:
            Tuple of ``(inputs, targets)``.

            **inputs** is a dict with keys:

            - ``"points"``: :class:`PointCloud3D` in lidar frame ``[N, 4]``
              (x, y, z, intensity).
            - ``"images"``: :class:`CameraImages` ``[1, 3, H, W]`` (left camera).
            - ``"extrinsics"``: :class:`CameraExtrinsics` ``[1, 4, 4]``
              (lidar-to-camera).
            - ``"intrinsics"``: :class:`CameraIntrinsics` ``[1, 3, 3]``.

            **targets** is a dict (training) or None (testing) with keys:

            - ``"boxes"``: :class:`BoundingBoxes3D` in lidar frame,
              format ``XYZLWHY``.
            - ``"labels"``: :class:`~torch.Tensor` of class indices.
        """
        frame_id = self._frame_ids[index]
        base = self._raw_folder / self._location

        points = self._load_velodyne(base, frame_id)
        calib = self._load_calib(base, frame_id)
        image = self._load_image(base, frame_id)

        # Filter points to camera FOV using K @ extrinsics[:3, :]
        img_h, img_w = image.shape[2], image.shape[3]
        K = calib["intrinsics"][0]  # [3, 3]
        ext = calib["extrinsics"][0]  # [4, 4]
        lidar_to_img = K @ ext[:3, :]  # [3, 4]
        fov_mask = _get_fov_mask(points[:, :3], lidar_to_img, img_h, img_w)
        points = points[fov_mask]

        inputs: FusionInputs = {
            "points": PointCloud3D(points),
            "images": CameraImages(image),
            "extrinsics": CameraExtrinsics(calib["extrinsics"]),
            "intrinsics": CameraIntrinsics(
                calib["intrinsics"], image_size=(img_h, img_w)
            ),
        }

        targets = None
        if self.train:
            targets = self._load_targets(base, frame_id, calib)

        if self.transforms is not None:
            inputs, targets = self.transforms(inputs, targets)

        return inputs, targets

    @property
    def _raw_folder(self) -> Path:
        return self.root / self.__class__.__name__ / "raw"

    def _check_exists(self) -> bool:
        folders = [self.velodyne_dir_name, self.calib_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all((self._raw_folder / self._location / d).is_dir() for d in folders)

    def download(self) -> None:
        """Download the KITTI dataset if it doesn't exist already."""
        if self._check_exists():
            return

        self._raw_folder.mkdir(parents=True, exist_ok=True)

        for fname in self.resources:
            download_and_extract_archive(
                url=f"{self.data_url}{fname}",
                download_root=str(self._raw_folder),
                filename=fname,
            )

    def _download_mini(self, frames: tuple[int, ...]) -> None:
        """Extract the named frames via HTTP range requests.

        Args:
            frames: Frame indices to extract from the requested
                split. Each ``i`` resolves to member ``{i:06d}.<ext>``.

        Raises:
            ValueError: If ``frames`` is empty.
        """
        if not frames:
            msg = "frames must be non-empty"
            raise ValueError(msg)
        if self._check_exists():
            return

        location_dir = self._raw_folder / self._location
        location_dir.mkdir(parents=True, exist_ok=True)

        frame_ids = [f"{i:06d}" for i in frames]
        # (zip filename, member subdirectory, file extension). The label_2
        # archive only contains the training split.
        archives = [
            ("data_object_velodyne.zip", self.velodyne_dir_name, "bin"),
            ("data_object_image_2.zip", self.image_dir_name, "png"),
            ("data_object_calib.zip", self.calib_dir_name, "txt"),
        ]
        if self.train:
            archives.append(("data_object_label_2.zip", self.labels_dir_name, "txt"))

        for fname, subdir, ext in archives:
            members = [f"{self._location}/{subdir}/{fid}.{ext}" for fid in frame_ids]
            _extract_remote_zip_members(
                url=f"{self.data_url}{fname}",
                members=members,
                dest=self._raw_folder,
            )

    def _load_velodyne(self, base: Path, frame_id: str) -> Tensor:
        path = base / self.velodyne_dir_name / f"{frame_id}.bin"
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return torch.from_numpy(points)

    def _load_image(self, base: Path, frame_id: str) -> Tensor:
        path = base / self.image_dir_name / f"{frame_id}.png"
        if path.exists():
            img = decode_image(str(path), mode=ImageReadMode.RGB)  # [3, H, W] uint8
            return img.unsqueeze(0).float() / 255.0
        return torch.zeros(1, 3, 1, 1)

    def _load_calib(self, base: Path, frame_id: str) -> dict[str, Tensor]:
        """Parse KITTI calibration file.

        Returns:
            Dict with ``"extrinsics"`` (lidar-to-camera, ``[1, 4, 4]``) and
            ``"intrinsics"`` (camera P2 projection, ``[1, 3, 3]``).
        """
        path = base / self.calib_dir_name / f"{frame_id}.txt"
        calib_data: dict[str, np.ndarray] = {}
        with path.open() as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                calib_data[key.strip()] = np.array(
                    [float(x) for x in value.split()], dtype=np.float32
                )

        # P2: 3x4 projection matrix for left color camera
        p2 = calib_data["P2"].reshape(3, 4)
        K = p2[:, :3]
        intrinsics = torch.from_numpy(K).unsqueeze(0)  # [1, 3, 3]

        # R0_rect: 3x3 rectification rotation
        r0 = np.eye(4, dtype=np.float32)
        r0[:3, :3] = calib_data["R0_rect"].reshape(3, 3)

        # Tr_velo_to_cam: 3x4 velodyne-to-camera
        velo_to_cam = np.eye(4, dtype=np.float32)
        velo_to_cam[:3, :] = calib_data["Tr_velo_to_cam"].reshape(3, 4)

        # P2's 4th column encodes a stereo baseline offset in camera frame.
        # Fold it into the extrinsic so that K @ extrinsics[:3, :] gives
        # exact pixel-accurate projection (matching P2 @ R0 @ Tr_velo_to_cam).
        baseline = np.eye(4, dtype=np.float32)
        baseline[:3, 3] = np.linalg.solve(K, p2[:, 3])

        extrinsics = torch.from_numpy(baseline @ r0 @ velo_to_cam).unsqueeze(
            0
        )  # [1, 4, 4]

        return {"extrinsics": extrinsics, "intrinsics": intrinsics}

    def _load_targets(
        self,
        base: Path,
        frame_id: str,
        calib: dict[str, Tensor],
    ) -> SampleTargets:
        """Parse KITTI label file and convert to lidar frame.

        Returns:
            Dict with ``"boxes"`` (:class:`BoundingBoxes3D`, XYZLWHY format),
            ``"labels"`` (int tensor).
        """
        path = base / self.labels_dir_name / f"{frame_id}.txt"
        label_ids: list[int] = []
        boxes_cam: list[list[float]] = []

        with path.open() as f:
            for line in csv.reader(f, delimiter=" "):
                if not line or line[0] == "DontCare":
                    continue
                label_ids.append(self.class_to_idx.get(line[0], -1))
                # KITTI label: h, w, l, x, y, z, rotation_y (camera frame)
                h, w, l = float(line[8]), float(line[9]), float(line[10])
                x, y, z = float(line[11]), float(line[12]), float(line[13])
                ry = float(line[14])
                boxes_cam.append([x, y, z, h, w, l, ry])

        if not boxes_cam:
            return {
                "boxes": BoundingBoxes3D(
                    torch.zeros(0, 7), format=BoundingBox3DFormat.XYZLWHY
                ),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        boxes_cam_t = torch.tensor(boxes_cam, dtype=torch.float32)
        boxes_lidar = _cam_to_lidar_boxes(boxes_cam_t, calib["extrinsics"][0])

        return {
            "boxes": BoundingBoxes3D(boxes_lidar, format=BoundingBox3DFormat.XYZLWHY),
            "labels": torch.tensor(label_ids, dtype=torch.int64),
        }


def _cam_to_lidar_boxes(boxes_cam: Tensor, extrinsics: Tensor) -> Tensor:
    """Convert KITTI camera-frame boxes to lidar-frame XYZLWHY format.

    Args:
        boxes_cam: ``[N, 7]`` boxes as ``(x, y, z, h, w, l, ry)`` in camera
            frame.
        extrinsics: ``[4, 4]`` lidar-to-camera matrix.

    Returns:
        ``[N, 7]`` boxes as ``(cx, cy, cz, l, w, h, yaw)`` in lidar frame.
    """
    x_cam, y_cam, z_cam = boxes_cam[:, 0], boxes_cam[:, 1], boxes_cam[:, 2]
    h, w, l = boxes_cam[:, 3], boxes_cam[:, 4], boxes_cam[:, 5]
    ry = boxes_cam[:, 6]

    # KITTI location is at the bottom center of the box in camera frame.
    # Camera Y points down, so shift up by half height to get the geometric center.
    y_cam = y_cam - h / 2

    # Transform center from camera to lidar using inverse extrinsics
    ones = torch.ones_like(x_cam)
    centers_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # [N, 4]
    cam_to_lidar = torch.linalg.inv(extrinsics)
    centers_lidar = (cam_to_lidar @ centers_cam.T).T[:, :3]  # [N, 3]

    # Dimensions: KITTI h,w,l -> our l,w,h (dx,dy,dz in lidar frame)
    l_lidar = l  # camera Z -> lidar X
    w_lidar = w  # camera X -> lidar Y
    h_lidar = h  # camera Y -> lidar Z

    # Rotation: KITTI rotation_y is around camera Y (pointing down)
    # In lidar frame, yaw is around Z (pointing up)
    yaw = -ry - np.pi / 2

    return torch.stack(
        [
            centers_lidar[:, 0],
            centers_lidar[:, 1],
            centers_lidar[:, 2],
            l_lidar,
            w_lidar,
            h_lidar,
            yaw,
        ],
        dim=-1,
    )


def _get_fov_mask(
    points_3d: Tensor,
    proj_matrix: Tensor,
    img_h: int,
    img_w: int,
) -> Tensor:
    """Get boolean mask for points that project into the camera image.

    Args:
        points_3d: ``[N, 3]`` 3D points.
        proj_matrix: ``[3, 4]`` projection matrix that maps ``points_3d`` to
            image coordinates (e.g. ``P2`` for camera-frame points, or
            ``P2 @ R0 @ Tr`` for lidar-frame points).
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        Boolean mask ``[N]``. True for points with positive depth that project
        within image bounds.
    """
    n = points_3d.shape[0]
    ones = torch.ones(n, 1, dtype=points_3d.dtype)
    pts_hom = torch.cat([points_3d, ones], dim=1)  # [N, 4]

    # Project: [3, 4] @ [4, N] -> [3, N]
    pts_img = (proj_matrix @ pts_hom.T).T  # [N, 3]

    depth = pts_img[:, 2]
    u = pts_img[:, 0] / depth.clamp(min=1e-6)
    v = pts_img[:, 1] / depth.clamp(min=1e-6)

    valid = (depth > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return valid


def _extract_remote_zip_members(
    url: str,
    members: list[str],
    dest: Path,
) -> None:
    """Extract specific members from a remote zip via HTTP range requests.

    Args:
        url: HTTP(S) URL of the zip archive. The server must respond to
            ``Range`` requests.
        members: Member names to extract, as they appear in the zip's
            central directory. Each is written to ``dest / member``. A
            :class:`KeyError` propagates from :mod:`zipfile` if any member
            is missing.
        dest: Destination root directory. Member paths are joined relative
            to it; intermediate directories are created as needed.
    """
    remote = _HttpRangeFile(url)
    with zipfile.ZipFile(remote) as zf:
        for member in members:
            dst = dest / member
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dst.open("wb") as out:
                out.write(src.read())


class _HttpRangeFile(io.RawIOBase):
    """Read-only seekable file-like backed by HTTP ``Range`` requests.

    Lets :class:`zipfile.ZipFile` open a remote archive without downloading
    it in full. This is required for the KITTI mini split, where the upstream zips
    are tens of gigabytes but only a handful of stored (uncompressed)
    members are needed.
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url
        self._pos = 0
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD")
        ) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                msg = f"server did not return Content-Length for {url!r}"
                raise OSError(msg)
            self._size = int(length)

    @override
    def readable(self) -> bool:
        return True

    @override
    def seekable(self) -> bool:
        return True

    @override
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            msg = f"invalid whence: {whence}"
            raise ValueError(msg)
        return self._pos

    @override
    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        if size <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + size, self._size) - 1
        req = urllib.request.Request(
            self._url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        with urllib.request.urlopen(req) as response:
            data = response.read()
        self._pos += len(data)
        return data
