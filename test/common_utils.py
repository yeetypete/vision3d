import math

import torch

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)


def make_bounding_boxes_3d(
    *,
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZWHDYPR,
    num_boxes: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> BoundingBoxes3D:
    """Generate random, valid 3D bounding boxes for testing.

    Produces non-degenerate boxes with positive dimensions and rotation angles
    in (-pi, +pi].

    Returns:
        BoundingBoxes3D with shape ``[num_boxes, K]`` where K depends on format.

    Raises:
        ValueError: If ``format`` is not a supported :class:`BoundingBox3DFormat`.
    """
    dtype = dtype or torch.float32

    # Random positive dimensions
    w = torch.rand(num_boxes) * 10 + 1
    h = torch.rand(num_boxes) * 10 + 1
    d = torch.rand(num_boxes) * 10 + 1

    # Random center positions
    cx = torch.rand(num_boxes) * 100 - 50
    cy = torch.rand(num_boxes) * 100 - 50
    cz = torch.rand(num_boxes) * 100 - 50

    # Random rotation angles in (-pi, +pi]
    yaw = torch.rand(num_boxes) * 2 * math.pi - math.pi
    pitch = torch.rand(num_boxes) * 2 * math.pi - math.pi
    roll = torch.rand(num_boxes) * 2 * math.pi - math.pi

    if format is BoundingBox3DFormat.XYZXYZ:
        x1 = cx - w / 2
        y1 = cy - h / 2
        z1 = cz - d / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        z2 = cz + d / 2
        parts = (x1, y1, z1, x2, y2, z2)
    elif format is BoundingBox3DFormat.XYZWHD:
        parts = (cx, cy, cz, w, h, d)
    elif format is BoundingBox3DFormat.XYZWHDY:
        parts = (cx, cy, cz, w, h, d, yaw)
    elif format is BoundingBox3DFormat.XYZWHDYPR:
        parts = (cx, cy, cz, w, h, d, yaw, pitch, roll)
    else:
        raise ValueError(f"Format {format} is not supported")

    data = torch.stack(parts, dim=-1).to(dtype=dtype, device=device)
    return BoundingBoxes3D(data, format=format)


def make_point_cloud_3d(
    *,
    num_points: int = 100,
    num_features: int = 0,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> PointCloud3D:
    """Generate a random 3D point cloud for testing.

    Points span both positive and negative coordinates.

    Args:
        num_points: Number of points.
        num_features: Extra per-point feature columns beyond xyz.
        dtype: Data type. Defaults to float32.
        device: Device.

    Returns:
        PointCloud3D with shape ``[num_points, 3 + num_features]``.
    """
    dtype = dtype or torch.float32
    xyz = torch.rand(num_points, 3, dtype=dtype, device=device) * 200 - 100
    if num_features > 0:
        features = torch.rand(num_points, num_features, dtype=dtype, device=device)
        data = torch.cat([xyz, features], dim=-1)
    else:
        data = xyz
    return PointCloud3D(data)


def make_camera_images(
    *,
    num_cameras: int = 6,
    channels: int = 3,
    height: int = 224,
    width: int = 224,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> CameraImages:
    """Generate random multi-camera images for testing.

    Returns:
        CameraImages with shape ``[num_cameras, channels, height, width]``.
    """
    dtype = dtype or torch.float32
    return CameraImages(
        torch.rand(num_cameras, channels, height, width, dtype=dtype, device=device)
    )


def make_camera_extrinsics(
    *,
    num_cameras: int = 6,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> CameraExtrinsics:
    """Generate random camera extrinsic matrices for testing.

    Returns:
        CameraExtrinsics with shape ``[num_cameras, 4, 4]``.
    """
    dtype = dtype or torch.float32
    return CameraExtrinsics(
        torch.eye(4, dtype=dtype, device=device).expand(num_cameras, -1, -1).clone()
    )


def make_camera_intrinsics(
    *,
    num_cameras: int = 6,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> CameraIntrinsics:
    """Generate random camera intrinsic matrices for testing.

    Returns:
        CameraIntrinsics with shape ``[num_cameras, 3, 3]``.
    """
    dtype = dtype or torch.float32
    K = torch.eye(3, dtype=dtype, device=device).expand(num_cameras, -1, -1).clone()
    K[:, 0, 0] = 500.0  # fx
    K[:, 1, 1] = 500.0  # fy
    K[:, 0, 2] = 320.0  # cx
    K[:, 1, 2] = 240.0  # cy
    return CameraIntrinsics(K)
