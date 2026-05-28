"""Optional 3D visualization utilities.

Requires the ``viz`` dependency group::

    pip install vision3d[viz]
"""

from ._blueprint import camera_grid, fusion_layout, lidar_view
from ._logging import (
    log_boxes_3d,
    log_cameras,
    log_cylinders_3d,
    log_cylinders_on_cameras,
    log_point_cloud,
    log_sample,
)

__all__ = [
    "camera_grid",
    "fusion_layout",
    "lidar_view",
    "log_boxes_3d",
    "log_cameras",
    "log_cylinders_3d",
    "log_cylinders_on_cameras",
    "log_point_cloud",
    "log_sample",
]
