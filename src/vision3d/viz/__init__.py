"""Optional 3D visualization utilities.

Requires the ``viz`` dependency group::

    pip install vision3d[viz]
"""

from ._blueprint import camera_grid, fusion_layout, lidar_view, time_series_view
from ._logging import (
    MetricLogger,
    log_boxes_3d,
    log_cameras,
    log_point_cloud,
    log_sample,
    log_scalars,
    style_series,
)

__all__ = [
    "MetricLogger",
    "camera_grid",
    "fusion_layout",
    "lidar_view",
    "log_boxes_3d",
    "log_cameras",
    "log_point_cloud",
    "log_sample",
    "log_scalars",
    "style_series",
    "time_series_view",
]
