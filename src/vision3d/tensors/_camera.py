"""Camera-related TVTensor types."""

from typing import TYPE_CHECKING, override

import torch
from torchvision.tv_tensors import TVTensor

if TYPE_CHECKING:
    from typing import Any, Self


# TODO(yeetypete): add PerCameraTransform wrapper for torchvision 2D transforms
# (ColorJitter, GaussianBlur, Normalize, etc.). Geometric image transforms
# (resize, crop) need dedicated implementations that also update CameraIntrinsics.
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
