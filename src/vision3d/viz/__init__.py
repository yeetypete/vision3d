"""Optional 3D visualization utilities built on Rerun.

Requires the ``viz`` dependency group::

    pip install vision3d[viz]
"""

from ._logging import log_boxes_3d, log_cameras, log_point_cloud, log_sample

__all__ = [
    "log_boxes_3d",
    "log_cameras",
    "log_point_cloud",
    "log_sample",
]
