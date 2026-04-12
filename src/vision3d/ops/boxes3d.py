"""Convert 3D bounding boxes between formats."""

from typing import TYPE_CHECKING

from vision3d.tensors import BoundingBox3DFormat

if TYPE_CHECKING:
    from torch import Tensor

from ._box3d_convert import _xyzlwh_to_xyzxyz, _xyzxyz_to_xyzlwh


def box3d_convert[*B](
    boxes: Tensor[*B, 6],
    in_fmt: BoundingBox3DFormat | str,
    out_fmt: BoundingBox3DFormat | str,
) -> Tensor[*B, 6]:
    """Convert 3D bounding boxes from ``in_fmt`` to ``out_fmt``.

    Only the lossless ``XYZXYZ`` <-> ``XYZLWH`` conversion is supported.
    All other conversions would discard or fabricate rotation angles.

    Args:
        boxes: Boxes to convert with shape ``[..., K]``. Supports any number
            of leading batch dimensions.
        in_fmt: Source format.
        out_fmt: Target format.

    Returns:
        Converted boxes with the same leading dimensions.
    """
    if isinstance(in_fmt, str):
        in_fmt = BoundingBox3DFormat[in_fmt.upper()]
    if isinstance(out_fmt, str):
        out_fmt = BoundingBox3DFormat[out_fmt.upper()]

    if in_fmt == out_fmt:
        return boxes.clone()

    pair = (in_fmt, out_fmt)

    # Axis-aligned conversions
    if pair == (BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZLWH):
        return _xyzxyz_to_xyzlwh(boxes)
    if pair == (BoundingBox3DFormat.XYZLWH, BoundingBox3DFormat.XYZXYZ):
        return _xyzlwh_to_xyzxyz(boxes)

    raise NotImplementedError(
        f"Conversion from {in_fmt.value} to {out_fmt.value} is not supported."
    )
