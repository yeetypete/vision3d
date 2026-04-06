from enum import Enum
from typing import TYPE_CHECKING, Any, override

import torch
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import TVTensor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Self


class BoundingBox3DFormat(Enum):
    """Coordinate format of a 3D bounding box.

    Available formats are:

    * ``XYZXYZ``: axis-aligned box via two opposite corners;
      x1, y1, z1 (min corner), x2, y2, z2 (max corner).
    * ``XYZWHD``: center position and size;
      cx, cy, cz (center), w, h, d (width, height, depth).
    * ``XYZWHDY``: center, size, and yaw rotation;
      cx, cy, cz, w, h, d, yaw. Yaw in radians.
    * ``XYZWHDYPR``: center, size, and full Euler rotation;
      cx, cy, cz, w, h, d, yaw, pitch, roll. Angles in radians (-pi..+pi).
    """

    XYZXYZ = "XYZXYZ"
    XYZWHD = "XYZWHD"
    XYZWHDY = "XYZWHDY"
    XYZWHDYPR = "XYZWHDYPR"

    @staticmethod
    def is_rotated(format: BoundingBox3DFormat) -> bool:
        return (
            format == BoundingBox3DFormat.XYZWHDY
            or format == BoundingBox3DFormat.XYZWHDYPR
        )


class BoundingBoxes3D(TVTensor):
    """:class:`torch.Tensor` subclass for 3D bounding boxes with shape ``[N, K]``.

    Where ``N`` is the number of bounding boxes and ``K`` depends on the format:
    6 for axis-aligned (``XYZXYZ``, ``XYZWHD``), 7 for yaw-only (``XYZWHDY``),
    or 9 for full 9-DOF (``XYZWHDYPR``).

    Rotation angles are Euler angles in radians (-pi to +pi).

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format (BoundingBox3DFormat, str): Format of the 3D bounding box.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If
            omitted, will be inferred from ``data``.
        device (torch.device, optional): Desired device of the bounding box. If
            omitted and ``data`` is a :class:`torch.Tensor`, the device is taken
            from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations
            on the bounding box. If omitted and ``data`` is a :class:`torch.Tensor`,
            the value is taken from it. Otherwise, defaults to ``False``.
    """

    format: BoundingBox3DFormat

    # This stub exists so pyrefly sees the correct constructor signature instead of torch.Tensor.__init__.
    def __init__(
        self,
        data: Any,
        *,
        format: BoundingBox3DFormat | str,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None: ...

    @classmethod
    def _wrap(
        cls,
        tensor: torch.Tensor,
        *,
        format: BoundingBox3DFormat | str,
        check_dims: bool = True,
    ) -> BoundingBoxes3D:
        if check_dims:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                raise ValueError(f"Expected a 1D or 2D tensor, got {tensor.ndim}D")

        if isinstance(format, str):
            format = BoundingBox3DFormat[format.upper()]

        bounding_boxes = tensor.as_subclass(cls)
        bounding_boxes.format = format
        return bounding_boxes

    def __new__(
        cls,
        data: Any,
        *,
        format: BoundingBox3DFormat | str,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        if isinstance(format, str):
            format = BoundingBox3DFormat[format.upper()]
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        if not torch.is_floating_point(tensor):
            raise ValueError(
                f"3D bounding boxes require floating point tensors, got {tensor.dtype}."
            )
        return cls._wrap(tensor, format=format)

    @classmethod
    @override
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> BoundingBoxes3D:
        # Metadata is lost after __torch_function__ calls. Restore it by taking
        # the format from the first BoundingBoxes3D in the args. This is correct
        # in most cases; when it isn't (e.g. mixing formats in one operation),
        # it's likely a mis-use we don't guard against.
        flat_params, _ = tree_flatten(
            tuple(args) + (tuple(kwargs.values()) if kwargs else ())
        )
        first_bbox_from_args = next(
            x for x in flat_params if isinstance(x, BoundingBoxes3D)
        )
        format = first_bbox_from_args.format

        if isinstance(output, torch.Tensor) and not isinstance(output, BoundingBoxes3D):
            output = cls._wrap(output, format=format, check_dims=False)
        elif isinstance(output, (tuple, list)):
            # This branch exists for chunk() and unbind()
            output = type(output)(
                cls._wrap(part, format=format, check_dims=False) for part in output
            )
        return output

    @override
    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(format=self.format)
