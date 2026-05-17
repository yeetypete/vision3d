"""Functional kernels for 3D geometric transforms."""

import math

import torch
from torch import Tensor
from torchvision.tv_tensors import TVTensor

from vision3d.tensors import (
    BoundingBox3DFormat,
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraImages,
    CameraIntrinsics,
    PointCloud3D,
)

from ._registry import register_kernel

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
        inpt.as_subclass(Tensor), format=inpt.format, axis=axis
    )
    return wrap(output, like=inpt)


# Which camera-frame axis to reflect for each world flip axis. The choice
# is free under the algebra (any reflection in the camera frame produces
# a valid factorization), but for upright cameras (image_y ≈ -world_Z) a
# world Z-flip is most naturally a vertical pixel flip while world X/Y
# flips are naturally horizontal — this matches what a human expects to
# see in a Rerun side-view.
_CAMERA_REFLECTION_AXIS: dict[str, int] = {"x": 0, "y": 0, "z": 1}


def flip_3d_camera_images(images: Tensor, *, axis: str) -> Tensor:
    """Flip every camera image to match a world-axis flip.

    ``axis="z"`` flips vertically (``dim=-2``); ``"x"`` and ``"y"`` flip
    horizontally (``dim=-1``). Paired with :func:`flip_3d_camera_extrinsics`
    and :func:`flip_3d_camera_intrinsics` to keep the image / intrinsics /
    extrinsics triple consistent. The math works for any camera orientation
    because the new extrinsics absorb the world-flip-to-camera-frame
    discrepancy; the per-axis pixel-flip direction is purely a convention
    chosen to match the visual intuition for upright cameras.

    Args:
        images: Image tensor ``[..., H, W]``.
        axis: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Flipped images with the same shape.
    """
    dim = -1 if _CAMERA_REFLECTION_AXIS[axis] == 0 else -2
    return torch.flip(images, dims=[dim])


@register_kernel(flip_3d, CameraImages)
def _flip_3d_camera_images_kernel(images: Tensor, *, axis: str) -> Tensor:
    return flip_3d_camera_images(images, axis=axis)


def flip_3d_camera_intrinsics(
    intrinsics: Tensor, *, image_size: tuple[int, int], axis: str
) -> Tensor:
    """Update camera intrinsics for a horizontal or vertical image flip.

    For ``axis="x"`` or ``"y"`` (horizontal pixel flip): ``cx → W - cx``
    and the off-diagonal skew sign flips. For ``axis="z"`` (vertical pixel
    flip): ``cy → H - cy`` and the skew sign flips. Focal lengths are
    unchanged.

    Args:
        intrinsics: Intrinsic matrices ``[..., 3, 3]``.
        image_size: ``(H, W)`` of the corresponding images.
        axis: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Updated intrinsics with the same shape.
    """
    height, width = image_size
    K = intrinsics.clone()
    K[..., 0, 1] = -K[..., 0, 1]
    if _CAMERA_REFLECTION_AXIS[axis] == 0:
        K[..., 0, 2] = width - K[..., 0, 2]
    else:
        K[..., 1, 2] = height - K[..., 1, 2]
    return K


@register_kernel(flip_3d, CameraIntrinsics, tv_tensor_wrapper=False)
def _flip_3d_camera_intrinsics_dispatch(
    inpt: CameraIntrinsics, *, axis: str
) -> TVTensor:
    output = flip_3d_camera_intrinsics(
        inpt.as_subclass(Tensor), image_size=inpt.image_size, axis=axis
    )
    return CameraIntrinsics._wrap(output, image_size=inpt.image_size)


def flip_3d_camera_extrinsics(extrinsics: Tensor, *, axis: str) -> Tensor:
    """Update camera extrinsics after flipping the lidar frame.

    A world reflection ``M_world`` makes ``p_cam = E · M_world · p_lidar'``.
    The product ``E · M_world`` has determinant ``-1`` and isn't a valid
    rotation, so we left-multiply by a camera-frame reflection ``M_img``
    to recover a valid rotation in the new extrinsics. The reflected
    camera-frame axis is chosen per world axis (see
    :data:`_CAMERA_REFLECTION_AXIS`) so the corresponding pixel flip is
    visually intuitive for upright cameras: horizontal for world X/Y
    flips, vertical for world Z flip.

    Args:
        extrinsics: Extrinsic matrices ``[..., 4, 4]``.
        axis: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Updated extrinsics with the same shape.
    """
    M_world = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    M_world[AXIS_INDEX[axis], AXIS_INDEX[axis]] = -1.0
    M_img = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    M_img[_CAMERA_REFLECTION_AXIS[axis], _CAMERA_REFLECTION_AXIS[axis]] = -1.0
    return M_img @ extrinsics @ M_world


@register_kernel(flip_3d, CameraExtrinsics)
def _flip_3d_camera_extrinsics_kernel(extrinsics: Tensor, *, axis: str) -> Tensor:
    return flip_3d_camera_extrinsics(extrinsics, axis=axis)


def translate_3d(inpt: Tensor, *, offset: Tensor) -> Tensor:
    """Translate a tensor by a 3D offset.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        offset: Translation ``[3]`` as ``(tx, ty, tz)``.

    Returns:
        Translated tensor.
    """
    return inpt


def translate_3d_point_cloud(points: Tensor, *, offset: Tensor) -> Tensor:
    """Translate point cloud coordinates by ``offset``.

    Args:
        points: Point cloud tensor ``[..., 3+C]``.
        offset: Translation ``[3]`` as ``(tx, ty, tz)``.

    Returns:
        Translated point cloud with the same shape.
    """
    points = points.clone()
    points[..., :3] += offset
    return points


@register_kernel(translate_3d, PointCloud3D)
def _translate_3d_point_cloud_kernel(points: Tensor, *, offset: Tensor) -> Tensor:
    return translate_3d_point_cloud(points, offset=offset)


def translate_3d_bounding_boxes(
    boxes: Tensor, *, format: BoundingBox3DFormat, offset: Tensor
) -> Tensor:
    """Translate 3D bounding boxes by ``offset``.

    Args:
        boxes: Bounding box tensor ``[..., K]``.
        format: Format of the boxes.
        offset: Translation ``[3]`` as ``(tx, ty, tz)``.

    Returns:
        Translated bounding boxes with the same shape.
    """
    boxes = boxes.clone()

    if format is BoundingBox3DFormat.XYZXYZ:
        boxes[..., :3] += offset
        boxes[..., 3:6] += offset
    else:
        # XYZLWH, XYZLWHY, XYZLWHYPR: first 3 are center
        boxes[..., :3] += offset

    return boxes


@register_kernel(translate_3d, BoundingBoxes3D, tv_tensor_wrapper=False)
def _translate_3d_bounding_boxes_dispatch(
    inpt: BoundingBoxes3D, *, offset: Tensor
) -> TVTensor:
    from vision3d.tensors import wrap

    output = translate_3d_bounding_boxes(
        inpt.as_subclass(Tensor), format=inpt.format, offset=offset
    )
    return wrap(output, like=inpt)


def translate_3d_camera_extrinsics(extrinsics: Tensor, *, offset: Tensor) -> Tensor:
    """Update camera extrinsics after translating the lidar frame.

    The lidar-to-camera extrinsic translation changes because the lidar
    origin moved by ``offset`` in the lidar frame.

    Args:
        extrinsics: Extrinsic matrices ``[..., 4, 4]``.
        offset: Translation ``[3]`` as ``(tx, ty, tz)`` in lidar frame.

    Returns:
        Updated extrinsics with the same shape.
    """
    extrinsics = extrinsics.clone()
    # E' = E @ T_inv where T_inv translates by -offset
    # This is equivalent to: E'[:3, 3] -= E[:3, :3] @ offset
    extrinsics[..., :3, 3] -= extrinsics[..., :3, :3] @ offset
    return extrinsics


@register_kernel(translate_3d, CameraExtrinsics)
def _translate_3d_camera_extrinsics_kernel(
    extrinsics: Tensor, *, offset: Tensor
) -> Tensor:
    return translate_3d_camera_extrinsics(extrinsics, offset=offset)


def _rotation_matrix(axis: Tensor, angle: float) -> Tensor:
    """Build a 3x3 rotation matrix from an axis and angle (radians).

    Uses Rodrigues' rotation formula.

    Args:
        axis: Unit vector ``[3]`` defining the rotation axis.
        angle: Rotation angle in radians.

    Returns:
        ``[3, 3]`` rotation matrix.
    """
    axis = axis / axis.norm()
    K = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=axis.dtype,
    )
    R = (
        torch.eye(3, dtype=axis.dtype)
        + math.sin(angle) * K
        + (1 - math.cos(angle)) * (K @ K)
    )
    return R


def rotate_3d(inpt: Tensor, *, rotation_matrix: Tensor) -> Tensor:
    """Rotate a tensor by a 3x3 rotation matrix.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        rotation_matrix: ``[3, 3]`` rotation matrix.

    Returns:
        Rotated tensor.
    """
    return inpt


def rotate_3d_point_cloud(points: Tensor, *, rotation_matrix: Tensor) -> Tensor:
    """Rotate point cloud coordinates by ``rotation_matrix``.

    Args:
        points: Point cloud tensor ``[..., 3+C]``.
        rotation_matrix: ``[3, 3]`` rotation matrix.

    Returns:
        Rotated point cloud with the same shape.
    """
    points = points.clone()
    points[..., :3] = points[..., :3] @ rotation_matrix.T
    return points


@register_kernel(rotate_3d, PointCloud3D)
def _rotate_3d_point_cloud_kernel(points: Tensor, *, rotation_matrix: Tensor) -> Tensor:
    return rotate_3d_point_cloud(points, rotation_matrix=rotation_matrix)


def rotate_3d_bounding_boxes(
    boxes: Tensor, *, format: BoundingBox3DFormat, rotation_matrix: Tensor
) -> Tensor:
    """Rotate 3D bounding boxes by ``rotation_matrix``.

    Only rotated formats are supported:

    - ``XYZLWHY``: only Z-axis rotations (pure yaw).
    - ``XYZLWHYPR``: arbitrary rotations.

    Axis-aligned formats (``XYZXYZ``, ``XYZLWH``) cannot represent rotation
    and will raise :class:`NotImplementedError`.

    Args:
        boxes: Bounding box tensor ``[..., K]``.
        format: Format of the boxes.
        rotation_matrix: ``[3, 3]`` rotation matrix.

    Returns:
        Rotated bounding boxes with the same shape.

    Raises:
        NotImplementedError: If ``format`` is axis-aligned.
        ValueError: If ``format`` is ``XYZLWHY`` and rotation is not pure yaw.
    """
    if format in (BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZLWH):
        msg = f"Rotation is not supported for axis-aligned format {format.value}."
        raise NotImplementedError(msg)

    boxes = boxes.clone()
    boxes[..., :3] = boxes[..., :3] @ rotation_matrix.T

    if format is BoundingBox3DFormat.XYZLWHY:
        # Only pure Z-axis rotation is valid for yaw-only boxes
        if not _is_z_rotation(rotation_matrix):
            msg = "XYZLWHY only supports Z-axis rotation."
            raise ValueError(msg)
        yaw_delta = math.atan2(
            rotation_matrix[1, 0].item(), rotation_matrix[0, 0].item()
        )
        boxes[..., 6] += yaw_delta
    elif format is BoundingBox3DFormat.XYZLWHYPR:
        yaw_delta = math.atan2(
            rotation_matrix[1, 0].item(), rotation_matrix[0, 0].item()
        )
        pitch_delta = math.atan2(
            -rotation_matrix[2, 0].item(),
            math.sqrt(
                rotation_matrix[2, 1].item() ** 2 + rotation_matrix[2, 2].item() ** 2
            ),
        )
        roll_delta = math.atan2(
            rotation_matrix[2, 1].item(), rotation_matrix[2, 2].item()
        )
        boxes[..., 6] += yaw_delta
        boxes[..., 7] += pitch_delta
        boxes[..., 8] += roll_delta

    return boxes


@register_kernel(rotate_3d, BoundingBoxes3D, tv_tensor_wrapper=False)
def _rotate_3d_bounding_boxes_dispatch(
    inpt: BoundingBoxes3D, *, rotation_matrix: Tensor
) -> TVTensor:
    from vision3d.tensors import wrap

    output = rotate_3d_bounding_boxes(
        inpt.as_subclass(Tensor),
        format=inpt.format,
        rotation_matrix=rotation_matrix,
    )
    return wrap(output, like=inpt)


def _is_z_rotation(rotation_matrix: Tensor) -> bool:
    """Check if a rotation matrix is a pure rotation around Z.

    A pure Z rotation has the form ``[[c, -s, 0], [s, c, 0], [0, 0, 1]]``.

    Returns:
        True if the rotation is purely around Z.
    """
    return (
        abs(rotation_matrix[2, 0].item()) < 1e-6
        and abs(rotation_matrix[2, 1].item()) < 1e-6
        and abs(rotation_matrix[0, 2].item()) < 1e-6
        and abs(rotation_matrix[1, 2].item()) < 1e-6
        and abs(rotation_matrix[2, 2].item() - 1.0) < 1e-6
    )


def rotate_3d_camera_extrinsics(
    extrinsics: Tensor, *, rotation_matrix: Tensor
) -> Tensor:
    """Update camera extrinsics after rotating the lidar frame.

    The lidar-to-camera extrinsic ``E`` satisfies ``p_cam = E @ p_lidar``.
    After rotating the lidar frame by ``R``, points become ``p' = R @ p``,
    so ``E' = E @ R_inv`` to keep ``p_cam = E' @ p'``.

    Args:
        extrinsics: Extrinsic matrices ``[..., 4, 4]``.
        rotation_matrix: ``[3, 3]`` rotation matrix.

    Returns:
        Updated extrinsics with the same shape.
    """
    extrinsics = extrinsics.clone()
    R_inv_4x4 = torch.eye(4, dtype=extrinsics.dtype)
    R_inv_4x4[:3, :3] = rotation_matrix.T
    extrinsics = extrinsics @ R_inv_4x4
    return extrinsics


@register_kernel(rotate_3d, CameraExtrinsics)
def _rotate_3d_camera_extrinsics_kernel(
    extrinsics: Tensor, *, rotation_matrix: Tensor
) -> Tensor:
    return rotate_3d_camera_extrinsics(extrinsics, rotation_matrix=rotation_matrix)


def scale_3d(inpt: Tensor, *, factor: float) -> Tensor:
    """Scale a tensor by a uniform factor.

    Dispatcher entry point. Type-specific kernels are registered below.

    Args:
        inpt: Input tensor.
        factor: Scale factor.

    Returns:
        Scaled tensor.
    """
    return inpt


def scale_3d_point_cloud(points: Tensor, *, factor: float) -> Tensor:
    """Scale point cloud coordinates by ``factor``.

    Args:
        points: Point cloud tensor ``[..., 3+C]``.
        factor: Scale factor.

    Returns:
        Scaled point cloud with the same shape.
    """
    points = points.clone()
    points[..., :3] *= factor
    return points


def scale_3d_bounding_boxes(
    boxes: Tensor, *, format: BoundingBox3DFormat, factor: float
) -> Tensor:
    """Scale 3D bounding boxes by ``factor``.

    Scales both position and dimensions. Rotation angles are unchanged.

    Args:
        boxes: Bounding box tensor ``[..., K]``.
        format: Format of the boxes.
        factor: Scale factor.

    Returns:
        Scaled bounding boxes with the same shape.
    """
    boxes = boxes.clone()

    if format is BoundingBox3DFormat.XYZXYZ:
        boxes[..., :6] *= factor
    else:
        # Center+size formats: scale center (0:3) and dimensions (3:6)
        boxes[..., :6] *= factor

    return boxes


@register_kernel(scale_3d, PointCloud3D)
def _scale_3d_point_cloud_kernel(points: Tensor, *, factor: float) -> Tensor:
    return scale_3d_point_cloud(points, factor=factor)


@register_kernel(scale_3d, BoundingBoxes3D, tv_tensor_wrapper=False)
def _scale_3d_bounding_boxes_dispatch(
    inpt: BoundingBoxes3D, *, factor: float
) -> TVTensor:
    from vision3d.tensors import wrap

    output = scale_3d_bounding_boxes(
        inpt.as_subclass(Tensor), format=inpt.format, factor=factor
    )
    return wrap(output, like=inpt)


def scale_3d_camera_extrinsics(extrinsics: Tensor, *, factor: float) -> Tensor:
    """Update camera extrinsics after scaling the lidar frame.

    Args:
        extrinsics: Extrinsic matrices ``[..., 4, 4]``.
        factor: Scale factor applied to the lidar frame.

    Returns:
        Updated extrinsics with the same shape.
    """
    extrinsics = extrinsics.clone()
    extrinsics[..., :3, 3] *= factor
    return extrinsics


@register_kernel(scale_3d, CameraExtrinsics)
def _scale_3d_camera_extrinsics_kernel(extrinsics: Tensor, *, factor: float) -> Tensor:
    return scale_3d_camera_extrinsics(extrinsics, factor=factor)
