"""A 3D extension of torchvision."""

import importlib.metadata

# Importing transforms here registers torchvision v2 kernels for vision3d
# TVTensors, so importing any vision3d submodule makes the kernels available.
from vision3d import transforms as transforms

__version__: str = importlib.metadata.version("vision3d")
