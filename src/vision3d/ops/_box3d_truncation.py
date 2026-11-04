"""Fraction of 3D bounding boxes outside a camera's field of view."""

import torch
from torch import Tensor

from vision3d.tensors import BoundingBox3DFormat

from ._project_box3d import _box_faces_in_camera, _clip_polygons, _fov_planes, _promote


def box3d_truncation(
    boxes: Tensor,
    format: BoundingBox3DFormat,
    extrinsics: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
) -> Tensor:
    """Fraction of the volume of 3D bounding boxes outside a camera's view.

    The field of view is the pyramid from the camera center through the image
    borders, as in :func:`project_box3d_to_image`. The part of a box behind
    the camera is outside it. Unlike the projected image area, the volume does
    not depend on a near clipping plane, and it stays meaningful for boxes
    next to the camera, whose nearest parts would dominate a projection.

    The truncation is geometric: it ignores occlusion, and it is not the share
    of the image a box covers. A box beside the camera can fill most of the
    image and still have most of its volume outside the field of view.

    The camera is a pinhole without lens distortion, as in
    :func:`project_to_image`. The op computes in at least double precision,
    as the volume of a small box far from the camera is a small difference of
    large terms.

    Args:
        boxes: 3D bounding boxes in the source frame ``[N, K]``.
        format: Format of the bounding boxes.
        extrinsics: Source-to-camera transformation ``[4, 4]``.
        intrinsics: Camera intrinsic matrix ``[3, 3]``. As in
            :func:`project_to_image`, the last row is not used.
        image_size: Image height and width in pixels as ``(height, width)``.

    Returns:
        Truncation ``[N]`` in ``[0, 1]``, in the dtype of ``boxes``: 0 for a
        box fully inside the field of view, and 1 for a box fully outside.
        For a flat box (one size zero, or negligible against the others), it
        is the fraction of its surface area outside instead. A box without
        surface area, such as a point or a line segment, gives 1.
    """
    out_dtype = boxes.dtype
    boxes, extrinsics, intrinsics = _promote(
        boxes.to(torch.promote_types(boxes.dtype, torch.float64)),
        extrinsics,
        intrinsics,
    )

    whole = _box_faces_in_camera(boxes, format, extrinsics)  # [N, 6, 4, 3]
    planes = _fov_planes(intrinsics, image_size)
    polygons = whole
    empty = torch.zeros(whole.shape[:2], dtype=torch.bool, device=boxes.device)
    for normal in planes:
        polygons, empty = _clip_polygons(polygons, empty, normal, 0.0)
    inside = polygons.masked_fill(empty[..., None, None], 0.0)  # [N, 6, 8, 3]
    # Pad the unclipped faces to the clipped size, so that a box fully in
    # view yields bitwise equal measures and a truncation of exactly 0.
    padding = whole[:, :, -1:].expand(-1, -1, len(planes), -1)
    whole = torch.cat([whole, padding], dim=2)  # [N, 6, 8, 3]

    # The volume of a solid is one third of the sum over its faces of the
    # vector area dotted with a point on the face, for the camera center as
    # reference point. The faces cut by the field of view lie on planes
    # through it and add nothing, so only box faces count. The factor of one
    # third cancels in the ratio, and so does the sign: a box with a negative
    # size, or a mirroring extrinsic, has inward oriented faces.
    area_whole = _vector_area(whole)  # [N, 6, 3]
    area_inside = _vector_area(inside)  # [N, 6, 3]
    anchor = whole[:, :, 0]  # [N, 6, 3]
    volume_whole = (area_whole * anchor).sum(dim=(1, 2))
    volume_inside = (area_inside * anchor).sum(dim=(1, 2))
    surface_whole = area_whole.norm(dim=-1).sum(dim=1)
    surface_inside = area_inside.norm(dim=-1).sum(dim=1)

    # A box is flat when its shortest edge is negligible against its longest:
    # its volume is then lost to rounding. Edges from corner 0 to corners 1,
    # 3 and 4, as the first face is (0, 3, 2, 1) and the third (0, 4, 7, 3).
    corner = whole[:, 0, 0, None]
    edges = torch.stack([whole[:, 0, 3], whole[:, 0, 1], whole[:, 2, 1]], dim=1)
    lengths = (edges - corner).norm(dim=-1)  # [N, 3]
    rel_tol = torch.finfo(boxes.dtype).eps ** 0.5
    solid = lengths.amin(dim=-1) > rel_tol * lengths.amax(dim=-1)

    measure_whole = torch.where(solid, volume_whole, surface_whole)
    measure_inside = torch.where(solid, volume_inside, surface_inside)
    tiny = torch.finfo(boxes.dtype).tiny
    ratio = measure_inside / torch.where(
        measure_whole.abs() > tiny, measure_whole, torch.full_like(measure_whole, tiny)
    )
    return (1 - ratio).clamp(0, 1).to(out_dtype)


def _vector_area(polygons: Tensor) -> Tensor:
    """Vector area of planar 3D polygons: area times the unit normal.

    Args:
        polygons: Vertices in cyclic order ``[..., V, 3]``.

    Returns:
        Vector areas ``[..., 3]``, along the right-handed normal of the
        vertex order.
    """
    # Relative to the first vertex, which keeps the products small.
    rel = polygons - polygons[..., :1, :]
    return torch.linalg.cross(rel, rel.roll(-1, dims=-2), dim=-1).sum(dim=-2) / 2
