"""Kernels that register vision3d TVTensors with miscellaneous torchvision v2 functionals."""

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as _F
from torchvision.transforms.v2.functional import register_kernel as _register_kernel

from vision3d.tensors import (
    BoundingBoxes3D,
    CameraExtrinsics,
    CameraIntrinsics,
    PointCloud3D,
)


@_register_kernel(_F.to_dtype, PointCloud3D)
@_register_kernel(_F.to_dtype, BoundingBoxes3D)
@_register_kernel(_F.to_dtype, CameraExtrinsics)
@_register_kernel(_F.to_dtype, CameraIntrinsics)
def _to_dtype_tensor_dispatch(
    inpt: Tensor, dtype: torch.dtype, scale: bool = False
) -> Tensor:
    return inpt.to(dtype)
