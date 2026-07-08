"""Optional 3D visualization utilities.

Requires the ``viz`` dependency group::

    pip install vision3d[viz]
"""

from ._blueprint import camera_grid, fusion_layout, lidar_view, time_series_view
from ._errors import LoggingInputError
from ._logger import RerunLogger
from ._scalars import log_scalars, style_series
from ._scene import log_boxes_3d, log_cameras, log_point_cloud, log_sample

__all__ = [
    "LoggingInputError",
    "RerunLogger",
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
