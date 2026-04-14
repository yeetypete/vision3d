"""Types for :mod:`vision3d.datasets` samples."""

import sys
from typing import NotRequired, Required, TypedDict

from torch import Tensor

if sys.version_info >= (3, 13):
    from typing import ReadOnly
else:
    from typing_extensions import ReadOnly

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)


class SampleInputs(TypedDict):
    """Per-frame model inputs; base type with all fields optional.

    Fields are ``ReadOnly`` so dataset-specific subclasses can tighten them
    from ``NotRequired`` to ``Required``.

    Attributes:
        points: Lidar point cloud for the frame.
        images: Multi-camera image tensor, one row per camera.
        extrinsics: Lidar-to-camera transforms, one row per camera.
        intrinsics: Per-camera pinhole intrinsic matrices.
    """

    points: NotRequired[ReadOnly[PointCloud3D]]
    images: NotRequired[ReadOnly[CameraImages]]
    extrinsics: NotRequired[ReadOnly[CameraExtrinsics]]
    intrinsics: NotRequired[ReadOnly[CameraIntrinsics]]


class LidarInputs(SampleInputs):
    """Lidar-only sample: points always present."""

    points: Required[PointCloud3D]


class CameraInputs(SampleInputs):
    """Camera-only sample: images, intrinsics, and extrinsics always present."""

    images: Required[CameraImages]
    intrinsics: Required[CameraIntrinsics]
    extrinsics: Required[CameraExtrinsics]


class FusionInputs(SampleInputs):
    """Fusion sample: lidar plus multi-camera, every field present."""

    points: Required[PointCloud3D]
    images: Required[CameraImages]
    extrinsics: Required[CameraExtrinsics]
    intrinsics: Required[CameraIntrinsics]


class SampleTargets(TypedDict):
    """Per-frame ground-truth annotations.

    Attributes:
        boxes: 3D bounding boxes in the lidar frame.
        labels: Integer class labels, one per box.
    """

    boxes: BoundingBoxes3D
    labels: Tensor
