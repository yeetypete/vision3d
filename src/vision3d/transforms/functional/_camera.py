"""Kernels that update :class:`~vision3d.tensors.CameraIntrinsics` under torchvision v2 image-space transforms."""

from typing import Any

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

from vision3d.tensors import CameraIntrinsics, wrap


@_register_kernel(_F.resize, CameraIntrinsics)
def resize_camera_intrinsics(
    inpt: CameraIntrinsics,
    size: list[int] | None,
    max_size: int | None = None,
    **kwargs: Any,
) -> CameraIntrinsics:
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
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, 2] -= left  # cx
    K[..., 1, 2] -= top  # cy
    return wrap(K, like=inpt, image_size=(height, width))


@_register_kernel(_F.center_crop, CameraIntrinsics)
def center_crop_camera_intrinsics(
    inpt: CameraIntrinsics, output_size: list[int]
) -> CameraIntrinsics:
    crop_h, crop_w = _center_crop_parse_output_size(output_size)
    old_h, old_w = inpt.image_size
    top = (old_h - crop_h) // 2
    left = (old_w - crop_w) // 2
    return crop_camera_intrinsics(inpt, top=top, left=left, height=crop_h, width=crop_w)


@_register_kernel(_F.pad, CameraIntrinsics)
def pad_camera_intrinsics(
    inpt: CameraIntrinsics, padding: int | list[int], **kwargs: Any
) -> CameraIntrinsics:
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
    cropped = crop_camera_intrinsics(
        inpt, top=top, left=left, height=height, width=width
    )
    return resize_camera_intrinsics(cropped, size=size)
