"""Camera-related TVTensor types."""

from typing import Any, Self, override

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as _F
from torchvision.transforms.v2.functional import (
    register_kernel as _register_kernel,
)
from torchvision.transforms.v2.functional._geometry import (
    _center_crop_parse_output_size,
    _compute_resized_output_size,
    _parse_pad_padding,
)
from torchvision.tv_tensors import TVTensor


class CameraImages(tv_tensors.Image):
    """:class:`~torchvision.tv_tensors.Image` subclass for multi-camera images with shape ``[N, C, H, W]``.

    Inherits from :class:`torchvision.tv_tensors.Image` so every
    torchvision v2 image transform dispatches automatically.

    For 3D spatial transforms (flip, rotate, etc.) this type passes through
    unchanged.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type.
        device (torch.device, optional): Desired device.
        requires_grad (bool, optional): Whether autograd should record operations.
    """

    # This stub exists so pyrefly sees the correct constructor signature instead of torch.Tensor.__init__.
    def __init__(
        self,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None: ...

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        if tensor.ndim < 3:
            raise ValueError(
                f"Expected at least 3D tensor [N, C, H, W], got {tensor.ndim}D"
            )
        return tensor.as_subclass(cls)

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


class CameraExtrinsics(TVTensor):
    """:class:`torch.Tensor` subclass for camera extrinsic matrices with shape ``[N, 4, 4]``.

    Each matrix transforms a point from the dataset's *source frame*
    to camera frame. The source frame is dataset-defined: lidar for
    lidar-equipped datasets (e.g. KITTI, nuScenes), ego/world for
    camera-only datasets.

    3D spatial transforms (flip, rotate, etc.) update these matrices
    to keep the source-to-camera mapping consistent after the source
    frame changes.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type.
        device (torch.device, optional): Desired device.
        requires_grad (bool, optional): Whether autograd should record operations.
    """

    # This stub exists so pyrefly sees the correct constructor signature instead of torch.Tensor.__init__.
    def __init__(
        self,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None: ...

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        if tensor.ndim < 2 or tensor.shape[-2:] != (4, 4):
            raise ValueError(
                f"Expected tensor with shape [..., 4, 4], got {tuple(tensor.shape)}"
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(
                f"Extrinsics require floating point tensors, got {tensor.dtype}."
            )
        return tensor.as_subclass(cls)

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


class CameraIntrinsics(TVTensor):
    """:class:`torch.Tensor` subclass for camera intrinsic matrices with shape ``[N, 3, 3]``.

    Each matrix maps from camera-frame 3D coordinates to pixel coordinates.

    Image-space transforms (resize, crop, etc.) update these matrices.
    3D spatial transforms pass them through unchanged.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        image_size (tuple): Height and width of the corresponding images as
            ``(h, w)``. Required for geometric transforms (resize) that need
            to compute scale factors.
        dtype (torch.dtype, optional): Desired data type.
        device (torch.device, optional): Desired device.
        requires_grad (bool, optional): Whether autograd should record operations.
    """

    image_size: tuple[int, int]

    # This stub exists so pyrefly sees the correct constructor signature instead of torch.Tensor.__init__.
    def __init__(
        self,
        data: Any,
        *,
        image_size: tuple[int, int],
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None: ...

    @classmethod
    def _wrap(
        cls,
        tensor: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> Self:
        if (
            not isinstance(image_size, tuple)
            or len(image_size) != 2
            or not all(isinstance(d, int) and d > 0 for d in image_size)
        ):
            raise ValueError(
                f"image_size must be a tuple of two positive ints (h, w), "
                f"got {image_size!r}"
            )
        intrinsics = tensor.as_subclass(cls)
        intrinsics.image_size = image_size
        return intrinsics

    def __new__(
        cls,
        data: Any,
        *,
        image_size: tuple[int, int],
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected tensor with shape [..., 3, 3], got {tuple(tensor.shape)}"
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(
                f"Intrinsics require floating point tensors, got {tensor.dtype}."
            )
        return cls._wrap(tensor, image_size=image_size)

    @classmethod
    @override
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Any = (),
        kwargs: Any = None,
    ) -> Self:

        flat_params, _ = tree_flatten(
            tuple(args) + (tuple(kwargs.values()) if kwargs else ())
        )
        first = next(x for x in flat_params if isinstance(x, CameraIntrinsics))
        if isinstance(output, torch.Tensor) and not isinstance(
            output, CameraIntrinsics
        ):
            output = cls._wrap(output, image_size=first.image_size)
        return output

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(image_size=self.image_size)


@_register_kernel(_F.resize, CameraIntrinsics)
def _resize_intrinsics(
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
    return CameraIntrinsics._wrap(K, image_size=(new_h, new_w))


@_register_kernel(_F.crop, CameraIntrinsics)
def _crop_intrinsics(
    inpt: CameraIntrinsics, top: int, left: int, height: int, width: int
) -> CameraIntrinsics:
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, 2] -= left  # cx
    K[..., 1, 2] -= top  # cy
    return CameraIntrinsics._wrap(K, image_size=(height, width))


@_register_kernel(_F.center_crop, CameraIntrinsics)
def _center_crop_intrinsics(
    inpt: CameraIntrinsics, output_size: list[int]
) -> CameraIntrinsics:
    crop_h, crop_w = _center_crop_parse_output_size(output_size)
    old_h, old_w = inpt.image_size
    top = (old_h - crop_h) // 2
    left = (old_w - crop_w) // 2
    return _crop_intrinsics(inpt, top=top, left=left, height=crop_h, width=crop_w)


@_register_kernel(_F.pad, CameraIntrinsics)
def _pad_intrinsics(
    inpt: CameraIntrinsics, padding: int | list[int], **kwargs: Any
) -> CameraIntrinsics:
    left, right, top, bottom = _parse_pad_padding(padding)
    K = inpt.as_subclass(Tensor).clone()
    K[..., 0, 2] += left  # cx
    K[..., 1, 2] += top  # cy
    old_h, old_w = inpt.image_size
    new_h = old_h + top + bottom
    new_w = old_w + left + right
    return CameraIntrinsics._wrap(K, image_size=(new_h, new_w))


@_register_kernel(_F.resized_crop, CameraIntrinsics)
def _resized_crop_intrinsics(
    inpt: CameraIntrinsics,
    top: int,
    left: int,
    height: int,
    width: int,
    size: list[int],
    **kwargs: Any,
) -> CameraIntrinsics:
    cropped = _crop_intrinsics(inpt, top=top, left=left, height=height, width=width)
    return _resize_intrinsics(cropped, size=size)
