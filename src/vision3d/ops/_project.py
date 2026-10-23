"""3D-to-2D camera projection utilities."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from shape_extensions import IntVar


def project_to_image[N: IntVar](
    points_3d: "Tensor[[N, 3]]",
    extrinsics: "Tensor[[4, 4]]",
    intrinsics: "Tensor[[3, 3]]",
) -> "tuple[Tensor[[N, 2]], Tensor[[N]]]":
    """Project 3D points in lidar frame to pixel coordinates.

    Args:
        points_3d: Points in lidar frame ``[N, 3]``.
        extrinsics: Lidar-to-camera transformation ``[4, 4]``.
        intrinsics: Camera intrinsic matrix ``[3, 3]``.

    Returns:
        ``(uv, depth)`` where ``uv`` is the pixel coordinates ``[N, 2]`` (u, v)
        and ``depth`` is the camera-frame depth ``[N]``.
    """
    n = points_3d.shape[0]
    ones = torch.ones(n, 1, dtype=points_3d.dtype, device=points_3d.device)
    pts_hom = torch.cat([points_3d, ones], dim=1)  # [N, 4]

    # Transform to camera frame: [4, 4] @ [4, N] -> [4, N] -> [N, 4]
    pts_cam = (extrinsics @ pts_hom.T).T  # [N, 4]
    pts_cam_3d = pts_cam[:, :3]  # [N, 3]

    depth = pts_cam_3d[:, 2]  # [N]

    # Project to pixel: [3, 3] @ [3, N] -> [3, N] -> [N, 3]
    pts_img = (intrinsics @ pts_cam_3d.T).T  # [N, 3]

    u = pts_img[:, 0] / depth  # pyrefly: ignore[unsupported-operation]
    v = pts_img[:, 1] / depth  # pyrefly: ignore[unsupported-operation]

    # Points behind the camera (depth <= 0) get NaN pixel coordinates
    behind = depth <= 0
    u = u.masked_fill(behind, torch.nan)
    v = v.masked_fill(behind, torch.nan)

    uv = torch.stack([u, v], dim=-1)  # [N, 2]
    return uv, depth  # pyrefly: ignore[bad-return]


def points_in_image[N: IntVar](
    points_3d: "Tensor[[N, 3]]",
    extrinsics: "Tensor[[4, 4]]",
    intrinsics: "Tensor[[3, 3]]",
    image_size: tuple[int, int],
) -> "Tensor[[N]]":
    """Return a mask selecting 3D points that project inside an image.

    Args:
        points_3d: Points in lidar frame ``[N, 3]``.
        extrinsics: Lidar-to-camera transformation ``[4, 4]``.
        intrinsics: Camera intrinsic matrix ``[3, 3]``.
        image_size: Image height and width in pixels as ``(height, width)``.

    Returns:
        Boolean mask ``[N]``. An entry is true when the point has positive
        camera-frame depth and projects within the image bounds.
    """
    height, width = image_size
    uv, depth = project_to_image(points_3d, extrinsics, intrinsics)
    u, v = uv.unbind(dim=-1)
    return (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
