"""Load vision3d's compiled C++/CUDA extension.

Importing this module has the side effect of loading the ops into the dispatcher.
"""

import importlib.machinery
import os

import torch


def _get_extension_path(lib_name: str) -> str:
    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )
    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None or ext_specs.origin is None:
        msg = f"Could not find module '{lib_name}' in {lib_dir}. "
        raise ImportError(msg)
    return ext_specs.origin


def _load_library(lib_name: str) -> None:
    torch.ops.load_library(_get_extension_path(lib_name))


_load_library("_C")
