"""3D oriented bounding box IoU.

Thin Python wrapper around PyTorch3D's ``box3d_overlap`` which lives under
``src/vision3d/ops/csrc/iou_box3d/``.

The extension is compiled on first use via
:func:`torch.utils.cpp_extension.load` and cached in
``~/.cache/torch_extensions/``. Requires a C++ compiler
(``g++``/``clang``/MSVC) available on the user's system.
"""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._box3d_corners import box3d_corners

if TYPE_CHECKING:
    from vision3d.tensors import BoundingBox3DFormat


_CSRC_ROOT: Path = Path(__file__).parent / "csrc"
_EXT_NAME: str = "vision3d_ops"
_ext_lock = threading.Lock()
_ext_loaded: bool = False


def _ensure_extension_loaded() -> None:
    """Compile and load the C++ extension on first call.

    The ``TORCH_LIBRARY`` registration inside the extension hooks itself
    into the global dispatcher as a side effect of being imported, so
    subsequent calls to ``torch.ops.vision3d.iou_box3d`` find the kernel
    automatically. We keep the load guarded by a lock for thread safety.

    When CUDA is available, the ``.cu`` kernel is compiled too and
    registered under the ``CUDA`` dispatch key via ``WITH_CUDA``.
    """
    global _ext_loaded
    if _ext_loaded:
        return
    with _ext_lock:
        if _ext_loaded:
            return
        from torch.utils.cpp_extension import load

        sources = [
            str(_CSRC_ROOT / "register.cpp"),
            str(_CSRC_ROOT / "iou_box3d" / "iou_box3d_cpu.cpp"),
        ]
        extra_cflags: list[str] = []
        extra_cuda_cflags: list[str] = []
        with_cuda = torch.cuda.is_available()
        if with_cuda:
            sources.append(str(_CSRC_ROOT / "iou_box3d" / "iou_box3d.cu"))
            extra_cflags.append("-DWITH_CUDA")
            extra_cuda_cflags.append("-DWITH_CUDA")
            # PyTorch may detect a device capability newer than the
            # installed ``nvcc`` supports (e.g. Blackwell sm_120 with
            # CUDA 12.4 which caps at sm_90). Fall back to the highest
            # arch ``nvcc`` reliably supports plus PTX so the CUDA
            # runtime JITs for the real device on first launch. Honour
            # a user-supplied override if set.
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0 9.0+PTX")

        # is_python_module=False: the extension registers its ops via
        # TORCH_LIBRARY as a side effect of being dlopen'd, and has no
        # pybind11 PyInit function, so we load it for side effects only.
        load(
            name=_EXT_NAME,
            sources=sources,
            extra_include_paths=[str(_CSRC_ROOT)],
            extra_cflags=extra_cflags or None,
            extra_cuda_cflags=extra_cuda_cflags or None,
            with_cuda=with_cuda,
            is_python_module=False,
            verbose=False,
        )
        _ext_loaded = True


@torch.no_grad()
def box3d_iou(
    boxes1: Tensor,
    boxes2: Tensor,
    format: BoundingBox3DFormat,
) -> Tensor:
    """Compute the pairwise intersection-over-union of 3D ``boxes1`` and ``boxes2``.

    ``iou = vol / (vol1 + vol2 - vol)``, where ``vol`` is the volume of
    the intersecting convex polyhedron and ``vol1``, ``vol2`` are the
    volumes of the two input boxes.

    The same algorithm handles every supported box format — including
    full 9-DOF orientation (``XYZLWHYPR``) — because the clipping step
    operates on the 8 box corners regardless of how they were produced.

    Note: This function is not differentiable.

    Args:
        boxes1: First set of boxes ``[N, K]``.
        boxes2: Second set of boxes ``[M, K]``.
        format: Format of both box sets.

    Returns:
        ``[N, M]`` matrix of IoU values in ``[0, 1]``.
    """
    _ensure_extension_loaded()
    corners1 = box3d_corners(boxes1, format).to(torch.float32)  # [N, 8, 3]
    corners2 = box3d_corners(boxes2, format).to(torch.float32)  # [M, 8, 3]
    _, iou = torch.ops.vision3d.iou_box3d(corners1, corners2)
    return iou
