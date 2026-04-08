"""3D-to-2D camera projection utilities."""

import torch
from torch import Tensor


def project_to_image(
    points_3d: Tensor,
    extrinsics: Tensor,
    intrinsics: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project 3D points in lidar frame to pixel coordinates.

    Args:
        points_3d: Points in lidar frame ``[N, 3]``.
        extrinsics: Lidar-to-camera transformation ``[4, 4]``.
        intrinsics: Camera intrinsic matrix ``[3, 3]``.

    Returns:
        Tuple of:
        - ``uv``: Pixel coordinates ``[N, 2]`` (u, v).
        - ``depth``: Depth in camera frame ``[N]``.
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

    safe_depth = depth.clamp(min=1e-6)
    u = pts_img[:, 0] / safe_depth
    v = pts_img[:, 1] / safe_depth

    uv = torch.stack([u, v], dim=-1)  # [N, 2]
    return uv, depth
