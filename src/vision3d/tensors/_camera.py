"""Camera-related TVTensor types."""

from typing import TYPE_CHECKING, Any, override

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.tv_tensors import TVTensor

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self


class CameraImages(TVTensor):
    """:class:`torch.Tensor` subclass for multi-camera images with shape ``[N, C, H, W]``.

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

    Each matrix transforms a point from lidar frame to camera frame
    (lidar-to-camera convention).

    3D spatial transforms (flip, rotate, etc.) update these matrices to keep
    the lidar-to-camera mapping consistent after the lidar frame changes.

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
        if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected tensor with shape [..., 3, 3], got {tuple(tensor.shape)}"
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(
                f"Intrinsics require floating point tensors, got {tensor.dtype}."
            )
        return tensor.as_subclass(cls)

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


# Register CameraImages kernels for torchvision v2 photometric transforms.
from torchvision.transforms.v2 import functional as _F
from torchvision.transforms.v2.functional import (
    register_kernel as _register_kernel,
)


def _make_camera_kernel(
    image_fn: Callable[..., Tensor],
) -> Callable[..., CameraImages]:
    def kernel(inpt: CameraImages, *args: Any, **kwargs: Any) -> CameraImages:
        output = image_fn(inpt.as_subclass(Tensor), *args, **kwargs)
        return tv_tensors.wrap(output, like=inpt)

    return kernel


for _fn, _img_fn in [
    (_F.adjust_brightness, _F.adjust_brightness_image),
    (_F.adjust_contrast, _F.adjust_contrast_image),
    (_F.adjust_saturation, _F.adjust_saturation_image),
    (_F.adjust_hue, _F.adjust_hue_image),
    (_F.adjust_gamma, _F.adjust_gamma_image),
    (_F.adjust_sharpness, _F.adjust_sharpness_image),
    (_F.rgb_to_grayscale, _F.rgb_to_grayscale_image),
    (_F.permute_channels, _F.permute_channels_image),
    (_F.posterize, _F.posterize_image),
    (_F.solarize, _F.solarize_image),
    (_F.autocontrast, _F.autocontrast_image),
    (_F.equalize, _F.equalize_image),
    (_F.invert, _F.invert_image),
    (_F.normalize, _F.normalize_image),
    (_F.gaussian_blur, _F.gaussian_blur_image),
    (_F.gaussian_noise, _F.gaussian_noise_image),
    (_F.to_dtype, _F.to_dtype_image),
    (_F.erase, _F.erase_image),
    (_F.grayscale_to_rgb, _F.grayscale_to_rgb_image),
    (_F.jpeg, _F.jpeg_image),
]:
    _register_kernel(_fn, CameraImages)(_make_camera_kernel(_img_fn))
