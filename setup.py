"""setuptools entry point for vision3d."""

import os
import subprocess
from pathlib import Path
from typing import override

import torch
from setuptools import setup
from setuptools.command.sdist import sdist
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

_ROOT = Path(__file__).resolve().parent


def get_version() -> str:
    """Return the project version.

    If the ``BUILD_VERSION`` environment variable is set, it fully overrides the
    base version read from ``version.txt``. Otherwise, for local builds the
    the short git commit SHA is appened as a PEP 440 local version identifier.
    """
    if build_version := os.getenv("BUILD_VERSION"):
        return build_version

    with open(_ROOT / "version.txt") as f:
        version = f.readline().strip()

    if "+" in version:
        return version

    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT))
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return version

    return f"{version}+{sha[:7]}"


class VersionedSdist(sdist):
    """Bake the fully-resolved version into the sdist's ``version.txt``."""

    @override
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        """Write the resolved version into ``version.txt`` in the release tree."""
        super().make_release_tree(base_dir, files)
        (Path(base_dir) / "version.txt").write_text(f"{get_version()}\n")


FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
_HAS_CUDA = (torch.cuda.is_available() and CUDA_HOME is not None) or FORCE_CUDA

_CSRC = _ROOT / "src/vision3d/ops/csrc"
_SOURCES = [
    "src/vision3d/ops/csrc/iou_box3d.cpp",
    "src/vision3d/ops/csrc/iou_box3d/iou_box3d_cpu.cpp",
]
if _HAS_CUDA:
    _SOURCES.append("src/vision3d/ops/csrc/iou_box3d/iou_box3d.cu")

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
    # ``USE_CUDA`` exposes the CUDA-specific stable C shim functions
    _DEFINE_MACROS.append(("USE_CUDA", None))

# Statically link the CUDA runtime so the wheel doesn't carry a
# ``libcudart.so.<MAJOR>`` SONAME dependency. Combined with building against
# the oldest CUDA major we support, this produces a single wheel that works
# across all CUDA majors (driver backward compatibility handles execution).
#
# ``--cudart=static`` makes nvcc emit references to the static cudart symbols
# during .cu compilation. At link time, ``CUDAExtension`` would normally append
# ``-lcudart`` (dynamic) automatically; we strip that and explicitly link
# ``libcudart_static.a`` instead. cudart_static's internal pthread/dl/rt
# references are satisfied by libc on glibc 2.34+. Produced wheels require
# glibc 2.34+ at runtime.
_ext = Extension(
    name="vision3d._C",
    sources=_SOURCES,
    include_dirs=[str(_CSRC)],
    define_macros=_DEFINE_MACROS,
    extra_compile_args={"nvcc": ["--cudart=static"]} if _HAS_CUDA else {},
    py_limited_api=True,
)
if _HAS_CUDA:
    _ext.libraries = [lib for lib in _ext.libraries if lib != "cudart"]
    _ext.extra_link_args = ["-l:libcudart_static.a"]

setup(
    version=get_version(),
    ext_modules=[_ext],
    cmdclass={"build_ext": BuildExtension, "sdist": VersionedSdist},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
