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

    Args:
        boxes: 3D bounding boxes in the source frame ``[N, K]``.
        format: Format of the bounding boxes.
        extrinsics: Source-to-camera transformation ``[4, 4]``.
        intrinsics: Camera intrinsic matrix ``[3, 3]``.
        image_size: Image height and width in pixels as ``(height, width)``.
        near: Camera-frame depth of the clipping plane, in the units of the
            extrinsics. Must be positive.

    Returns:
        ``(boxes_2d, valid)`` where ``boxes_2d`` ``[N, 4]``, in the dtype of
        ``boxes``, is the extent of the visible part of each box in
        ``(x1, y1, x2, y2)`` pixel coordinates, and ``valid`` ``[N]`` is false
        when no area of the box is visible, that is in front of the clipping
        plane and inside the image. This includes a flat box seen edge-on. The
        ``boxes_2d`` row of a box that is not valid is zero.

    Raises:
        ValueError: If ``near`` is not positive.
    """
    if near <= 0:
        raise ValueError(f"near must be positive, got {near}")
    height, width = image_size
    extrinsics = extrinsics.to(boxes.dtype)
    intrinsics = intrinsics.to(boxes.dtype)
    rot, trans = extrinsics[:3, :3], extrinsics[:3, 3]

    corners = box3d_corners(boxes, format) @ rot.T + trans  # [N, 8, 3]
    faces = torch.tensor(_BOX_FACES, device=boxes.device)
    polygons = corners[:, faces]  # [N, 6, 4, 3]

    # The field of view in the camera frame, 0 <= u <= width and
    # 0 <= v <= height, as half-spaces through the camera center, then the
    # near plane.
    k_u, k_v, k_w = intrinsics.unbind(dim=0)
    normal_near = torch.tensor([0.0, 0.0, 1.0], dtype=boxes.dtype, device=boxes.device)
    planes = (
        (k_u, 0.0),
        (width * k_w - k_u, 0.0),
        (k_v, 0.0),
        (height * k_w - k_v, 0.0),
        (normal_near, -near),
    )
    empty = torch.zeros(polygons.shape[:2], dtype=torch.bool, device=boxes.device)
    for normal, offset in planes:
        polygons, empty = _clip_polygons(polygons, empty, normal, offset)

    # Faces entirely out of view hold arbitrary points; give them a safe
    # depth. They are excluded below.
    depth = polygons[..., 2].masked_fill(empty.unsqueeze(-1), 1.0)
    pixels = polygons @ intrinsics.T  # [N, 6, 9, 3]
    uv = pixels[..., :2] / depth.unsqueeze(-1)  # [N, 6, 9, 2]
    # The box has visible area when any of its clipped faces has.
    valid = (_polygon_area(uv).masked_fill(empty, 0.0) > 0).any(dim=1)  # [N]

    skip = empty[..., None, None]
    uv_min = uv.masked_fill(skip, torch.inf).amin(dim=(1, 2))  # [N, 2]
    uv_max = uv.masked_fill(skip, -torch.inf).amax(dim=(1, 2))  # [N, 2]
    # Clipped vertices lie on the image border up to rounding.
    size = torch.tensor([width, height], dtype=boxes.dtype, device=boxes.device)
    boxes_2d = torch.cat([uv_min, uv_max], dim=-1).clamp(min=0).minimum(size.repeat(2))
    return boxes_2d.masked_fill(~valid.unsqueeze(-1), 0.0), valid


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
