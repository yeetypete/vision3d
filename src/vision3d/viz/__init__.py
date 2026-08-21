"""Optional 3D visualization utilities.

Requires the ``viz`` dependency group::

    pip install vision3d[viz]
"""

from ._annotate import (
    annotator_layout,
    boxes_to_map,
    colorize_points_from_cameras,
    lidar_to_map,
    log_boxes_3d_editable,
    log_ego,
    log_point_cloud_rgb,
    log_point_cloud_rgb_by_sweep,
    reserve_box_slots,
)
from ._blueprint import camera_grid, fusion_layout, lidar_view
from ._logging import log_boxes_3d, log_cameras, log_point_cloud, log_sample

__all__ = [
    "annotator_layout",
    "boxes_to_map",
    "camera_grid",
    "colorize_points_from_cameras",
    "fusion_layout",
    "lidar_to_map",
    "lidar_view",
    "log_boxes_3d",
    "log_boxes_3d_editable",
    "log_cameras",
    "log_ego",
    "log_point_cloud",
    "log_point_cloud_rgb",
    "log_point_cloud_rgb_by_sweep",
    "log_sample",
    "reserve_box_slots",
]
