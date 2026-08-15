"""setuptools entry point for vision3d."""

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
    include_paths,
)

_ROOT = Path(__file__).resolve().parent

FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
_HAS_CUDA = (torch.cuda.is_available() and CUDA_HOME is not None) or FORCE_CUDA

_CSRC = _ROOT / "src/vision3d/ops/csrc"
_SOURCES = [
    "src/vision3d/ops/csrc/iou_box3d.cpp",
    "src/vision3d/ops/csrc/iou_box3d/iou_box3d_cpu.cpp",
    "src/vision3d/ops/csrc/voxelize.cpp",
    "src/vision3d/ops/csrc/voxelize/voxelize_cpu.cpp",
]
if _HAS_CUDA:
    _SOURCES.append("src/vision3d/ops/csrc/iou_box3d/iou_box3d.cu")
    _SOURCES.append("src/vision3d/ops/csrc/voxelize/voxelize.cu")

Extension = CUDAExtension if _HAS_CUDA else CppExtension

# Define TORCH_TARGET_VERSION with min version 2.10 to expose only the
# stable API subset from torch
_DEFINE_MACROS: list[tuple[str, str | None]] = [
    (
        "TORCH_TARGET_VERSION",
        "0x020a000000000000",
    ),
]
if _HAS_CUDA:
    # `USE_CUDA` exposes the CUDA-specific stable C shim functions
    _DEFINE_MACROS.append(("USE_CUDA", None))

# Statically link the CUDA runtime so the wheel doesn't carry a
# `libcudart.so.<MAJOR>` SONAME dependency. Combined with building against
# the oldest CUDA major we support, this produces a single wheel that works
# across all CUDA majors (driver backward compatibility handles execution).
#
# `--cudart=static` makes nvcc emit references to the static cudart symbols
# during .cu compilation. At link time, `CUDAExtension` would normally append
# `-lcudart` (dynamic) automatically; we strip that and explicitly link
# `libcudart_static.a` instead. cudart_static internally calls into
# pthread/dl/rt, so we link those explicitly to keep the wheel importable on
# glibc 2.28+ (manylinux_2_28).
#
# Without this, clang-tidy reports every `STD_TORCH_CHECK` and
# `STABLE_TORCH_LIBRARY_*` use in our own sources, because it blames a macro
# expansion on the expansion site unless the header defining the macro is a
# system header. Marking the torch and CUDA headers `-isystem` fixes that. The
# extension classes add them as `-I`, so `include_dirs` is reset below.
_ISYSTEM = [
    _arg
    for _dir in include_paths(device_type="cuda" if _HAS_CUDA else "cpu")
    for _arg in ("-isystem", _dir)
]
_ext = Extension(
    name="vision3d._C",
    sources=_SOURCES,
    include_dirs=[str(_CSRC)],
    define_macros=_DEFINE_MACROS,
    extra_compile_args=(
        {
            "cxx": ["-std=c++20", *_ISYSTEM],
            "nvcc": ["-std=c++20", "--cudart=static", *_ISYSTEM],
        }
        if _HAS_CUDA
        else {"cxx": ["-std=c++20", *_ISYSTEM]}
    ),
    py_limited_api=True,
)
# Drop the `-I` duplicates the constructor added, since `_ISYSTEM` covers them.
_ext.include_dirs = [str(_CSRC)]
if _HAS_CUDA:
    _ext.libraries = [lib for lib in _ext.libraries if lib != "cudart"]
    _ext.extra_link_args = ["-l:libcudart_static.a", "-lpthread", "-ldl", "-lrt"]

setup(
    ext_modules=[_ext],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
