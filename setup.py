"""setuptools entry point for vision3d's C++ extension."""

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
_HAS_CUDA = (torch.cuda.is_available() and CUDA_HOME is not None) or FORCE_CUDA

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "src/vision3d/ops/csrc"
_SOURCES = [
    "src/vision3d/ops/csrc/iou_box3d.cpp",
    "src/vision3d/ops/csrc/iou_box3d/iou_box3d_cpu.cpp",
]
if _HAS_CUDA:
    _SOURCES.append("src/vision3d/ops/csrc/iou_box3d/iou_box3d.cu")

Extension = CUDAExtension if _HAS_CUDA else CppExtension

setup(
    ext_modules=[
        Extension(
            name="vision3d._C",
            sources=_SOURCES,
            include_dirs=[str(_CSRC)],
            define_macros=[("WITH_CUDA", None)] if _HAS_CUDA else [],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
