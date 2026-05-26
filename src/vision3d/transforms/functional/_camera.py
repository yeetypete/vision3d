"""Kernels that update camera TVTensors under torchvision v2 image-space transforms."""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as _F
from torchvision.transforms.v2.functional import (
    register_kernel as _register_kernel,
)
from torchvision.transforms.v2.functional._geometry import (
    _center_crop_parse_output_size,
    _compute_resized_output_size,
    _parse_pad_padding,
)

from vision3d.tensors import CameraExtrinsics, CameraIntrinsics, wrap


@_register_kernel(_F.resize, CameraIntrinsics)
def resize_camera_intrinsics(
    inpt: CameraIntrinsics,
    size: list[int] | None,
    max_size: int | None = None,
    **kwargs: Any,
) -> CameraIntrinsics:
    """Update :class:`~vision3d.tensors.CameraIntrinsics` for a resize of the corresponding image.

    Scales the focal lengths, skew, and principal point so projection
    through the updated intrinsics matches projection through the
    original intrinsics on the resized image.

    Args:
        inpt: The intrinsics to update.
        size: Target ``(h, w)`` after resize.
        max_size: Optional cap on the longer edge.
        kwargs: Unused; accepted for signature compatibility with
            :func:`torchvision.transforms.v2.functional.resize`.

    Returns:
        Updated intrinsics with the new ``image_size``.
    """
    old_h, old_w = inpt.image_size
    new_h, new_w = _compute_resized_output_size(
        (old_h, old_w), size=size, max_size=max_size
    )
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, :] *= new_w / old_w  # fx, skew, cx
    K[..., 1, :] *= new_h / old_h  # fy, cy
    return wrap(K, like=inpt, image_size=(new_h, new_w))


@_register_kernel(_F.crop, CameraIntrinsics)
def crop_camera_intrinsics(
    inpt: CameraIntrinsics, top: int, left: int, height: int, width: int
) -> CameraIntrinsics:
    """Update :class:`~vision3d.tensors.CameraIntrinsics` for a crop of the corresponding image.

    Shifts the principal point so projection through the updated
    intrinsics matches projection through the original intrinsics on
    the cropped image.

    Args:
        inpt: The intrinsics to update.
        top: Top edge of the crop in pixels.
        left: Left edge of the crop in pixels.
        height: Crop height in pixels.
        width: Crop width in pixels.

    Returns:
        Updated intrinsics with ``image_size`` set to ``(height, width)``.
    """
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, 2] -= left  # cx
    K[..., 1, 2] -= top  # cy
    return wrap(K, like=inpt, image_size=(height, width))


@_register_kernel(_F.center_crop, CameraIntrinsics)
def center_crop_camera_intrinsics(
    inpt: CameraIntrinsics, output_size: list[int]
) -> CameraIntrinsics:
    """Update :class:`~vision3d.tensors.CameraIntrinsics` for a center crop of the corresponding image.

    Args:
        inpt: The intrinsics to update.
        output_size: Target ``(h, w)`` after the center crop.

    Returns:
        Updated intrinsics with ``image_size`` set to ``output_size``.
    """
    crop_h, crop_w = _center_crop_parse_output_size(output_size)
    old_h, old_w = inpt.image_size
    top = (old_h - crop_h) // 2
    left = (old_w - crop_w) // 2
    return crop_camera_intrinsics(inpt, top=top, left=left, height=crop_h, width=crop_w)


@_register_kernel(_F.pad, CameraIntrinsics)
def pad_camera_intrinsics(
    inpt: CameraIntrinsics, padding: int | list[int], **kwargs: Any
) -> CameraIntrinsics:
    """Update :class:`~vision3d.tensors.CameraIntrinsics` for a pad of the corresponding image.

    Shifts the principal point by the top-left pad and grows
    ``image_size`` to include the padded borders.

    Args:
        inpt: The intrinsics to update.
        padding: Padding spec as accepted by
            :func:`torchvision.transforms.v2.functional.pad`.
        kwargs: Unused; accepted for signature compatibility with
            :func:`torchvision.transforms.v2.functional.pad`.

    Returns:
        Updated intrinsics with the padded ``image_size``.
    """
    left, right, top, bottom = _parse_pad_padding(padding)
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, 2] += left  # cx
    K[..., 1, 2] += top  # cy
    old_h, old_w = inpt.image_size
    new_h = old_h + top + bottom
    new_w = old_w + left + right
    return wrap(K, like=inpt, image_size=(new_h, new_w))


@_register_kernel(_F.resized_crop, CameraIntrinsics)
def resized_crop_camera_intrinsics(
    inpt: CameraIntrinsics,
    top: int,
    left: int,
    height: int,
    width: int,
    size: list[int],
    **kwargs: Any,
) -> CameraIntrinsics:
    """Update :class:`~vision3d.tensors.CameraIntrinsics` for a crop followed by a resize.

    Args:
        inpt: The intrinsics to update.
        top: Top edge of the crop in pixels.
        left: Left edge of the crop in pixels.
        height: Crop height in pixels.
        width: Crop width in pixels.
        size: Target ``(h, w)`` after the resize.
        kwargs: Unused; accepted for signature compatibility with
            :func:`torchvision.transforms.v2.functional.resized_crop`.

    Returns:
        Updated intrinsics with ``image_size`` set to ``size``.
    """
    cropped = crop_camera_intrinsics(
        inpt, top=top, left=left, height=height, width=width
    )
    return resize_camera_intrinsics(cropped, size=size)


@_register_kernel(_F.horizontal_flip, CameraIntrinsics)
def horizontal_flip_camera_intrinsics(inpt: CameraIntrinsics) -> CameraIntrinsics:
    """Update :class:`~vision3d.tensors.CameraIntrinsics` for a horizontal flip of the corresponding image.

    Mirrors the principal point about the image's vertical center line and
    negates the skew so projection through the updated intrinsics matches
    projection through the original intrinsics on the flipped image.

    Args:
        inpt: The intrinsics to update.

    Returns:
        Updated intrinsics with the same ``image_size``.
    """
    _, w = inpt.image_size
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, 1] = -K[..., 0, 1]  # skew
    K[..., 0, 2] = w - K[..., 0, 2]  # cx
    return wrap(K, like=inpt)


@_register_kernel(_F.vertical_flip, CameraIntrinsics)
def vertical_flip_camera_intrinsics(inpt: CameraIntrinsics) -> CameraIntrinsics:
    """Update :class:`~vision3d.tensors.CameraIntrinsics` for a vertical flip of the corresponding image.

    Mirrors the principal point about the image's horizontal center line and
    negates the skew so projection through the updated intrinsics matches
    projection through the original intrinsics on the flipped image.

    Args:
        inpt: The intrinsics to update.

    Returns:
        Updated intrinsics with the same ``image_size``.
    """
    h, _ = inpt.image_size
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, 1] = -K[..., 0, 1]  # skew
    K[..., 1, 2] = h - K[..., 1, 2]  # cy
    return wrap(K, like=inpt)


def _flip_camera_extrinsics(E: Tensor, *, world_axis: int, image_axis: int) -> Tensor:
    """Build ``M_img · E · M_world`` for a world-axis reflection paired with a camera-frame reflection.

    Args:
        E: Extrinsic matrices ``[..., 4, 4]``.
        world_axis: Source-frame axis (0=x, 1=y, 2=z) reflected by the flip.
        image_axis: Camera-frame axis reflected to keep projection consistent.

    Returns:
        New extrinsics ``[..., 4, 4]`` whose rotation has det +1 and whose
        projection of any 3D point matches the flipped pixel coordinate.
    """
    M_world = torch.eye(4, dtype=E.dtype, device=E.device)
    M_world[world_axis, world_axis] = -1.0
    M_img = torch.eye(4, dtype=E.dtype, device=E.device)
    M_img[image_axis, image_axis] = -1.0
    return M_img @ E @ M_world


@_register_kernel(_F.horizontal_flip, CameraExtrinsics)
def horizontal_flip_camera_extrinsics(inpt: CameraExtrinsics) -> CameraExtrinsics:
    """Update :class:`~vision3d.tensors.CameraExtrinsics` for a horizontal image flip.

    Reflects the source frame about its Y axis (paired with a camera-frame
    X reflection) so the source-to-camera mapping stays consistent with the
    horizontally flipped image.

    Args:
        inpt: The extrinsics to update.

    Returns:
        Updated extrinsics with the same shape.
    """
    out = _flip_camera_extrinsics(inpt.as_subclass(Tensor), world_axis=1, image_axis=0)
    return wrap(out, like=inpt)


@_register_kernel(_F.vertical_flip, CameraExtrinsics)
def vertical_flip_camera_extrinsics(inpt: CameraExtrinsics) -> CameraExtrinsics:
    """Update :class:`~vision3d.tensors.CameraExtrinsics` for a vertical image flip.

    Reflects the source frame about its Z axis (paired with a camera-frame
    Y reflection) so the source-to-camera mapping stays consistent with the
    vertically flipped image.

    Args:
        inpt: The extrinsics to update.

    Returns:
        Updated extrinsics with the same shape.
    """
    out = _flip_camera_extrinsics(inpt.as_subclass(Tensor), world_axis=2, image_axis=1)
    return wrap(out, like=inpt)
