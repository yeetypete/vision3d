"""Project 3D bounding boxes to 2D image boxes."""

import torch
from torch import Tensor

from vision3d.tensors import BoundingBox3DFormat

from ._box3d_corners import box3d_corners

# The 6 box faces as corner indices in cyclic order, for the corner ordering
# of ``box3d_corners``.
_BOX_FACES = (
    (0, 3, 2, 1),  # z-
    (4, 5, 6, 7),  # z+
    (0, 4, 7, 3),  # x-
    (1, 2, 6, 5),  # x+
    (0, 1, 5, 4),  # y-
    (3, 7, 6, 2),  # y+
)


def project_box3d_to_image(
    boxes: Tensor,
    format: BoundingBox3DFormat,
    extrinsics: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    near: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Project 3D bounding boxes to axis-aligned 2D boxes in an image.

    Projecting the 8 corners with :func:`project_to_image` and taking their
    extent fails in two ways: corners behind the camera have no pixel
    coordinates, and clamping the extent to the image overestimates the
    visible part of a box that crosses an image corner at an angle. This op
    instead clips each box face to the field of view of the camera and at the
    plane ``z = near`` in the camera frame, then projects it. The result is
    the exact 2D extent of the visible part of the box.

    The field of view is the pyramid from the camera center through the image
    borders. The camera frame follows :func:`project_to_image`: ``z`` is the
    depth along the optical axis, and ``x`` and ``y`` map to image ``u``
    (right) and ``v`` (down) through the intrinsics.

    The camera is a pinhole without lens distortion, as in
    :func:`project_to_image`. A distorted image bends straight box edges, so
    rectify such images first.

    Args:
        boxes: 3D bounding boxes in the source frame ``[N, K]``.
        format: Format of the bounding boxes.
        extrinsics: Source-to-camera transformation ``[4, 4]``.
        intrinsics: Camera intrinsic matrix ``[3, 3]``. As in
            :func:`project_to_image`, pixel coordinates are divided by the
            camera-frame depth, so the last row is not used.
        image_size: Image height and width in pixels as ``(height, width)``.
        near: Camera-frame depth of the clipping plane, in the units of the
            extrinsics. Must be positive. The field of view already excludes
            everything behind the camera, so the result converges as ``near``
            goes to zero and the default only avoids dividing by a zero
            depth. Raise it to ignore the parts of a box closer than the
            camera can see.

    Returns:
        ``(boxes_2d, valid)`` where ``boxes_2d`` ``[N, 4]``, in the dtype of
        ``boxes``, is the extent of the visible part of each box in
        ``(x1, y1, x2, y2)`` pixel coordinates, and ``valid`` ``[N]`` is false
        when no area of the box is visible, that is in front of the clipping
        plane and inside the image, up to rounding. This includes a flat box
        seen edge-on. The ``boxes_2d`` row of a box that is not valid is zero.

    Raises:
        ValueError: If ``near`` is not positive, or the camera matrices do not
            have the shapes above.
    """
    if near <= 0:
        raise ValueError(f"near must be positive, got {near}")
    out_dtype = boxes.dtype
    boxes, extrinsics, intrinsics = _promote(boxes, extrinsics, intrinsics)
    height, width = image_size

    polygons = _box_faces_in_camera(boxes, format, extrinsics)  # [N, 6, 4, 3]
    empty = torch.zeros(polygons.shape[:2], dtype=torch.bool, device=boxes.device)
    for normal in _fov_planes(intrinsics, image_size):
        polygons, empty = _clip_polygons(polygons, empty, normal, 0.0)
    normal_near = torch.tensor([0.0, 0.0, 1.0], dtype=boxes.dtype, device=boxes.device)
    polygons, empty = _clip_polygons(polygons, empty, normal_near, -near)

    # Faces entirely out of view hold arbitrary points; give them a safe
    # depth. They are excluded below.
    depth = polygons[..., 2].masked_fill(empty.unsqueeze(-1), 1.0)
    uv = (polygons @ intrinsics[:2].T) / depth.unsqueeze(-1)  # [N, 6, 9, 2]

    skip = empty[..., None, None]
    uv_min = uv.masked_fill(skip, torch.inf).amin(dim=(1, 2))  # [N, 2]
    uv_max = uv.masked_fill(skip, -torch.inf).amax(dim=(1, 2))  # [N, 2]
    # Clipped vertices lie on the image border up to rounding.
    x1, y1 = uv_min[:, 0].clamp(0, width), uv_min[:, 1].clamp(0, height)
    x2, y2 = uv_max[:, 0].clamp(0, width), uv_max[:, 1].clamp(0, height)
    boxes_2d = torch.stack([x1, y1, x2, y2], dim=-1)

    # The box has visible area when any of its clipped faces has. A face in a
    # plane through the camera projects to a line up to rounding, so require
    # an extent above the rounding error of pixel coordinates.
    has_area = (_polygon_area(uv).masked_fill(empty, 0.0) > 0).any(dim=1)
    tol = 16 * torch.finfo(boxes.dtype).eps * max(width, height)
    valid = has_area & (x2 - x1 > tol) & (y2 - y1 > tol)  # [N]
    boxes_2d = boxes_2d.masked_fill(~valid.unsqueeze(-1), 0.0)
    return boxes_2d.to(out_dtype), valid


def _promote(
    boxes: Tensor, extrinsics: Tensor, intrinsics: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Check the camera shapes and bring all inputs to a common dtype and device.

    Returns:
        ``(boxes, extrinsics, intrinsics)`` in the promoted dtype of the three
        inputs, on the device of ``boxes``.

    Raises:
        ValueError: If ``extrinsics`` is not ``[4, 4]`` or ``intrinsics`` is
            not ``[3, 3]``.
    """
    if extrinsics.shape != (4, 4):
        raise ValueError(f"extrinsics must be [4, 4], got {list(extrinsics.shape)}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be [3, 3], got {list(intrinsics.shape)}")
    dtype = torch.promote_types(
        torch.promote_types(boxes.dtype, extrinsics.dtype), intrinsics.dtype
    )
    return (
        boxes.to(dtype),
        extrinsics.to(boxes.device, dtype),
        intrinsics.to(boxes.device, dtype),
    )


def _box_faces_in_camera(
    boxes: Tensor, format: BoundingBox3DFormat, extrinsics: Tensor
) -> Tensor:
    """The 6 faces of 3D boxes as polygons in the camera frame.

    Args:
        boxes: 3D bounding boxes in the source frame ``[N, K]``.
        format: Format of the bounding boxes.
        extrinsics: Source-to-camera transformation ``[4, 4]``.

    Returns:
        Face vertices ``[N, 6, 4, 3]`` in cyclic order, as in ``_BOX_FACES``.
    """
    rot, trans = extrinsics[:3, :3], extrinsics[:3, 3]
    corners = box3d_corners(boxes, format) @ rot.T + trans  # [N, 8, 3]
    faces = torch.tensor(_BOX_FACES, device=boxes.device)
    return corners[:, faces]


def _fov_planes(intrinsics: Tensor, image_size: tuple[int, int]) -> list[Tensor]:
    """The field of view as half-spaces through the camera center.

    The field of view is ``0 <= u <= width`` and ``0 <= v <= height`` with
    ``u = (k_u @ p) / z`` and ``v = (k_v @ p) / z``, for the first two rows
    ``k_u`` and ``k_v`` of the intrinsics. Together the four half-spaces also
    exclude everything behind the camera.

    Args:
        intrinsics: Camera intrinsic matrix ``[3, 3]``.
        image_size: Image height and width in pixels as ``(height, width)``.

    Returns:
        The 4 inward plane normals ``[3]``; each half-space is
        ``p @ normal >= 0``.
    """
    height, width = image_size
    k_u, k_v = intrinsics[0], intrinsics[1]
    e_z = torch.zeros_like(k_u)
    e_z[2] = 1.0
    return [k_u, width * e_z - k_u, k_v, height * e_z - k_v]


def _clip_polygons(
    polygons: Tensor, empty: Tensor, normal: Tensor, offset: float
) -> tuple[Tensor, Tensor]:
    """Clip convex 3D polygons to the half-space ``p @ normal + offset >= 0``.

    Sutherland-Hodgman clipping on fixed-size polygons: vertices that repeat
    in a row are allowed, which pads a polygon to the output size without
    changing its shape. A convex polygon clipped by a plane gains at most
    one vertex, so the output has one vertex more than the input.

    Args:
        polygons: Vertices in cyclic order ``[..., V, 3]``.
        empty: Mask of polygons with no vertices ``[...]``; their vertices
            are arbitrary.
        normal: Normal of the plane, pointing into the kept side ``[3]``.
        offset: Offset of the plane.

    Returns:
        ``(polygons, empty)`` with polygons ``[..., V + 1, 3]``.
    """
    dist = polygons @ normal + offset  # [..., V]
    following = polygons.roll(-1, dims=-2)
    dist_following = dist.roll(-1, dims=-1)

    # Each edge contributes its start vertex when it is inside, then its
    # crossing point when its ends lie strictly on opposite sides.
    inside = dist >= 0
    crosses = dist * dist_following < 0
    denom = torch.where(crosses, dist - dist_following, torch.ones_like(dist))
    s = (dist / denom).unsqueeze(-1)
    crossings = polygons + s * (following - polygons)

    candidates = torch.stack([polygons, crossings], dim=-2).flatten(-3, -2)
    keep = torch.stack([inside, crosses], dim=-1).flatten(-2)  # [..., 2V]

    # Move the kept vertices to the front in order, then pad with the last.
    order = torch.argsort((~keep).to(torch.uint8), dim=-1, stable=True)
    count = keep.sum(dim=-1, keepdim=True)
    slots = torch.arange(polygons.shape[-2] + 1, device=polygons.device)
    index = order.gather(-1, torch.minimum(slots, (count - 1).clamp_min(0)))
    index = index.unsqueeze(-1).expand(*index.shape, polygons.shape[-1])
    return candidates.gather(-2, index), empty | (count.squeeze(-1) == 0)


def _polygon_area(polygons: Tensor) -> Tensor:
    """Area of 2D polygons by the shoelace formula.

    Args:
        polygons: Vertices in cyclic order ``[..., V, 2]``.

    Returns:
        Areas ``[...]``.
    """
    # Relative to the first vertex, which keeps the products small.
    x, y = (polygons - polygons[..., :1, :]).unbind(dim=-1)
    cross = x * y.roll(-1, dims=-1) - x.roll(-1, dims=-1) * y
    return cross.sum(dim=-1).abs() / 2
