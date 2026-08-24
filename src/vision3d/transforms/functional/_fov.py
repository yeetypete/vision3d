"""Functional kernel for camera field-of-view filtering."""

import torch
from torch import Tensor


def get_fov_mask(
    points_3d: Tensor,
    projection_matrix: Tensor,
    image_size: tuple[int, int],
) -> Tensor:
    """Return a mask selecting 3D points visible within a camera image.

    Args:
        points_3d: 3D points ``[N, 3]``.
        projection_matrix: Projection matrix ``[3, 4]`` mapping ``points_3d``
            from their current coordinate frame to homogeneous image
            coordinates.
        image_size: Image height and width in pixels as ``(height, width)``.

    Returns:
        Boolean mask ``[N]``. An entry is true when the point has positive
        depth and projects within the image bounds.
    """
    height, width = image_size
    ones = torch.ones(
        (points_3d.shape[0], 1),
        dtype=points_3d.dtype,
        device=points_3d.device,
    )
    points_homogeneous = torch.cat([points_3d, ones], dim=1)
    points_image = (projection_matrix @ points_homogeneous.T).T

    depth = points_image[:, 2]
    u = points_image[:, 0] / depth.clamp(min=1e-6)
    v = points_image[:, 1] / depth.clamp(min=1e-6)

    return (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
