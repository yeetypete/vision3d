"""setuptools entry point for vision3d's C++ extension."""

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

setup(
    version=get_version(),
    ext_modules=[
        Extension(
            name="vision3d._C",
            sources=_SOURCES,
            include_dirs=[str(_CSRC)],
            define_macros=[("WITH_CUDA", None)] if _HAS_CUDA else [],
        ),
    ],
    cmdclass={"build_ext": BuildExtension, "sdist": VersionedSdist},
)
