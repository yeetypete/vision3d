"""setuptools entry point for vision3d's C++ extension."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

_HAS_CUDA = CUDA_HOME is not None

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "src/vision3d/ops/csrc"
_SOURCES = [
    "src/vision3d/ops/csrc/iou_box3d.cpp",
    "src/vision3d/ops/csrc/iou_box3d/iou_box3d_cpu.cpp",
]
if _HAS_CUDA:
    _SOURCES.append("src/vision3d/ops/csrc/iou_box3d/iou_box3d.cu")

setup(
    ext_modules=[
        CUDAExtension(
            name="vision3d._C",
            sources=_SOURCES,
            include_dirs=[str(_CSRC)],
            define_macros=[("WITH_CUDA", None)] if _HAS_CUDA else [],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
