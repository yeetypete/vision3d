import math
import pickle
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)
from vision3d.transforms import Transform


def make_bounding_boxes_3d(
    *,
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
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
    elif format is BoundingBox3DFormat.XYZLWH:
        parts = (cx, cy, cz, w, h, d)
    elif format is BoundingBox3DFormat.XYZLWHY:
        parts = (cx, cy, cz, w, h, d, yaw)
    elif format is BoundingBox3DFormat.XYZLWHYPR:
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
    return CameraIntrinsics(K, image_size=(480, 640))


def box_at(
    cx: float, cy: float, cz: float = 0.0, *, fmt: BoundingBox3DFormat
) -> list[float]:
    """Build a unit-cube box centered at ``(cx, cy, cz)`` in the given format.

    Side length 2 on every axis. Rotation angles default to 0 for rotated
    formats.

    Returns:
        Box parameters as a list of floats whose length depends on ``fmt``.

    Raises:
        ValueError: If ``fmt`` is not a supported :class:`BoundingBox3DFormat`.
    """
    if fmt == BoundingBox3DFormat.XYZXYZ:
        return [cx - 1.0, cy - 1.0, cz - 1.0, cx + 1.0, cy + 1.0, cz + 1.0]
    if fmt == BoundingBox3DFormat.XYZLWH:
        return [cx, cy, cz, 2.0, 2.0, 2.0]
    if fmt == BoundingBox3DFormat.XYZLWHY:
        return [cx, cy, cz, 2.0, 2.0, 2.0, 0.0]
    if fmt == BoundingBox3DFormat.XYZLWHYPR:
        return [cx, cy, cz, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    raise ValueError(fmt)


def make_lidar_sample(
    *,
    num_points: int = 20,
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
) -> dict[str, Any]:
    """Build a lidar-only test sample with three boxes and matching labels.

    Returns:
        ``{"points": PointCloud3D, "boxes": BoundingBoxes3D, "labels": Tensor}``.
    """
    return {
        "points": make_point_cloud_3d(num_points=num_points),
        "boxes": make_bounding_boxes_3d(format=format, num_boxes=3),
        "labels": torch.tensor([0, 1, 2]),
    }


def make_fusion_sample(
    *,
    num_cameras: int = 4,
    image_size: tuple[int, int] = (32, 32),
    num_points: int = 20,
    format: BoundingBox3DFormat = BoundingBox3DFormat.XYZLWHYPR,
) -> dict[str, Any]:
    """Build a fusion test sample covering every vision3d TVTensor type.

    Returns:
        Dict with ``points``, ``boxes``, ``labels``, ``images``,
        ``extrinsics``, ``intrinsics``.
    """
    h, w = image_size
    intr = make_camera_intrinsics(num_cameras=num_cameras)
    intr = CameraIntrinsics(intr.as_subclass(torch.Tensor), image_size=image_size)
    return {
        "points": make_point_cloud_3d(num_points=num_points),
        "boxes": make_bounding_boxes_3d(format=format, num_boxes=3),
        "labels": torch.tensor([0, 1, 2]),
        "images": make_camera_images(num_cameras=num_cameras, height=h, width=w),
        "extrinsics": make_camera_extrinsics(num_cameras=num_cameras),
        "intrinsics": intr,
    }


def check_transform(transform: Transform, sample: Any) -> Any:
    """Audit a transform's input/output convention.

    Mirrors :func:`torchvision`'s ``check_transform`` test helper. Asserts:

    - The transform is pickleable.
    - Output container structure matches input.
    - Items the transform claims to handle (per
      :meth:`Transform._needs_transform_list`) keep their exact type
      after transformation; only their content may change.
    - Items the transform does not claim pass through by identity (same
      object, not just equal).

    Returns:
        The transformed sample.
    """
    pickle.loads(pickle.dumps(transform))

    flat_inputs, in_spec = tree_flatten(sample)
    output = transform(sample)
    flat_outputs, out_spec = tree_flatten(output)
    assert out_spec == in_spec

    needs = transform._needs_transform_list(flat_inputs)
    for inpt, outpt, transformed in zip(flat_inputs, flat_outputs, needs, strict=True):
        if transformed:
            assert type(outpt) is type(inpt), (
                f"{type(transform).__name__} mutated input type "
                f"{type(inpt).__name__} → {type(outpt).__name__}."
            )
        else:
            assert outpt is inpt, (
                f"{type(transform).__name__} did not preserve identity "
                f"for unhandled input of type {type(inpt).__name__}."
            )
    return output
