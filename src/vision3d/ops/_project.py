"""3D-to-2D camera projection utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor


def project_to_image[N: IntVar](
    points_3d: Tensor[[N, 3]],
    extrinsics: Tensor[[4, 4]],
    intrinsics: Tensor[[3, 3]],
) -> tuple[Tensor[[N, 2]], Tensor[[N]]]:
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
    pts_hom = torch.cat((points_3d, ones), dim=1)  # [N, 4]

    # Transform to camera frame: [4, 4] @ [4, N] -> [4, N] -> [N, 4]
    pts_cam = (extrinsics @ pts_hom.transpose(0, 1)).transpose(0, 1)  # [N, 4]
    pts_cam_3d = pts_cam[:, :3]  # [N, 3]

    depth = pts_cam_3d[:, 2]  # [N]

    # Project to pixel: [3, 3] @ [3, N] -> [3, N] -> [N, 3]
    pts_img = (intrinsics @ pts_cam_3d.transpose(0, 1)).transpose(0, 1)  # [N, 3]

    u = pts_img[:, 0] / depth
    v = pts_img[:, 1] / depth

    # Points behind the camera (depth <= 0) get NaN pixel coordinates
    behind = depth <= 0
    u = u.masked_fill(behind, float("nan"))
    v = v.masked_fill(behind, float("nan"))

    uv = torch.stack((u, v), dim=-1)  # [N, 2]
    return uv, depth
