"""Functional kernels for 3D geometric transforms."""

from typing import TYPE_CHECKING

import torch

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    PointCloud3D,
)

from ._registry import register_kernel

if TYPE_CHECKING:
    from torch import Tensor
    from torchvision.tv_tensors import TVTensor

# Axis indices for flip
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Which rotation angles to negate for each flip axis.
# Convention: yaw=around Z (idx 6), pitch=around Y (idx 7), roll=around X (idx 8).
# A flip negates angles that rotate around axes OTHER than the flip axis.
_FLIP_NEGATE_YPR: dict[str, list[int]] = {
    "x": [6, 7],  # negate yaw (Z) and pitch (Y)
    "y": [6, 8],  # negate yaw (Z) and roll (X)
    "z": [7, 8],  # negate pitch (Y) and roll (X)
}


def flip_3d(inpt: Tensor, *, axis: str) -> Tensor:
    """Flip a tensor along a 3D spatial axis.

    This is the dispatcher entry point. Type-specific kernels are registered
    below.

    Args:
        inpt: Input tensor.
        axis: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Flipped tensor.
    """
    return inpt


def flip_3d_point_cloud(points: Tensor, *, axis: str) -> Tensor:
    """Flip point cloud coordinates along ``axis``.

    Args:
        points: Point cloud tensor ``[..., 3+C]``.
        axis: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Flipped point cloud with the same shape.
    """
    idx = AXIS_INDEX[axis]
    shape = points.shape
    points = points.clone().reshape(-1, shape[-1])
    points[:, idx].neg_()
    return points.reshape(shape)


@register_kernel(flip_3d, PointCloud3D)
def _flip_3d_point_cloud_kernel(points: Tensor, *, axis: str) -> Tensor:
    return flip_3d_point_cloud(points, axis=axis)


def flip_3d_bounding_boxes(
    boxes: Tensor, *, format: BoundingBox3DFormat, axis: str
) -> Tensor:
    """Flip 3D bounding boxes along ``axis``.

    Args:
        boxes: Bounding box tensor ``[..., K]``.
        format: Format of the boxes.
        axis: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Flipped bounding boxes with the same shape.
    """
    idx = AXIS_INDEX[axis]
    shape = boxes.shape
    boxes = boxes.clone().reshape(-1, shape[-1])

    if format is BoundingBox3DFormat.XYZXYZ:
        # Swap and negate: min/max corners flip
        lo, hi = idx, idx + 3
        boxes[:, [lo, hi]] = boxes[:, [hi, lo]].neg_()
    elif format in (
        BoundingBox3DFormat.XYZLWH,
        BoundingBox3DFormat.XYZLWHY,
        BoundingBox3DFormat.XYZLWHYPR,
    ):
        boxes[:, idx].neg_()
        if format is BoundingBox3DFormat.XYZLWHY:
            if axis in ("x", "y"):
                boxes[:, 6].neg_()
        elif format is BoundingBox3DFormat.XYZLWHYPR:
            for angle_idx in _FLIP_NEGATE_YPR[axis]:
                boxes[:, angle_idx].neg_()

    return boxes.reshape(shape)


@register_kernel(flip_3d, BoundingBoxes3D, tv_tensor_wrapper=False)
def _flip_3d_bounding_boxes_dispatch(inpt: BoundingBoxes3D, *, axis: str) -> TVTensor:
    from vision3d.tensors import wrap

    output = flip_3d_bounding_boxes(
        inpt.as_subclass(torch.Tensor), format=inpt.format, axis=axis
    )
    return wrap(output, like=inpt)


def flip_3d_camera_extrinsics(extrinsics: Tensor, *, axis: str) -> Tensor:
    """Update camera extrinsics after flipping the lidar frame along ``axis``.

    The lidar-to-camera extrinsic is right-multiplied by a flip matrix that
    negates the flipped axis, keeping the projection consistent.

    Args:
        extrinsics: Extrinsic matrices ``[..., 4, 4]``.
        axis: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Updated extrinsics with the same shape.
    """
    idx = AXIS_INDEX[axis]
    extrinsics = extrinsics.clone()
    # Right-multiply by diag(..., -1, ..., 1): negate column `idx`
    extrinsics[..., :, idx].neg_()
    return extrinsics


@register_kernel(flip_3d, CameraExtrinsics)
def _flip_3d_camera_extrinsics_kernel(extrinsics: Tensor, *, axis: str) -> Tensor:
    return flip_3d_camera_extrinsics(extrinsics, axis=axis)
