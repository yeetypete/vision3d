# vision3d

This library is a 3D extension of
[torchvision](https://docs.pytorch.org/vision/stable/), providing datasets,
tensor types, transforms, ops, metrics, and visualization utilities for 3D
perception tasks.

Documentation is available at [vision3d.dev](https://vision3d.dev).

> [!WARNING]
> `vision3d` is in active early development. The API may change
> without notice and documentation may be incomplete.

## Requirements

- Python 3.12 or newer.
- PyTorch 2.10 or newer.
- Recommended: A CUDA-capable NVIDIA GPU for GPU execution.
- For building from source: the
  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) matching your
  PyTorch build.

## Installation

`vision3d` is distributed as a source package on PyPI. There are also pre-built
wheels attached to each
[GitHub release](https://github.com/yeetypete/vision3d/releases). We recommend
using [`uv`](https://docs.astral.sh/uv/) your package manager.

### From PyPI

```bash
uv add vision3d
```

> [!NOTE]
> Installing from PyPI builds the C++/CUDA extension on your machine, so
> the build-time requirements above apply. Use a pre-built wheel if you want to
> skip the compile step.

### From a release wheel

Pre-built wheels are published as assets on each
[GitHub release](https://github.com/yeetypete/vision3d/releases). vision3d
wheels are built against the
[LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html),
so a single wheel per CUDA variant works for any Python 3.12+ and torch 2.10+.
Wheels are currently built for:

- CUDA 12.8 and CUDA 13.0
- Linux x86_64

Pick the wheel that matches your CUDA runtime, then install it directly from the
release URL:

```bash
uv add https://github.com/yeetypete/vision3d/releases/download/v0.2.2/vision3d-0.2.2+cu130-cp312-abi3-linux_x86_64.whl
```

### From source

Clone the repository and sync the environment:

```bash
git clone https://github.com/yeetypete/vision3d.git
cd vision3d
uv sync --all-extras
```

`uv sync` compiles the C++/CUDA extension as part of installing the project. On
machines where CUDA is installed but no GPU is visible (for example, inside
containers), force a CUDA build with:

```bash
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0+PTX" uv sync --all-extras
```

> [!NOTE]
> `TORCH_CUDA_ARCH_LIST` selects which NVIDIA compute capabilities to
> compile CUDA kernels for (e.g. `12.0` for RTX 50-series). See the
> [PyTorch docs](https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension)
> for the full syntax.

To produce a wheel locally:

```bash
uv build
```

### Extras

- `nuscenes`: pulls in `nuscenes-devkit` for the nuScenes dataset loader.
- `viz`: pulls in `rerun-sdk` for the visualization utilities in `vision3d.viz`.

Request them at install time, for example: `uv add 'vision3d[nuscenes,viz]'`.

## Contributing

Contributions are welcome! See
[CONTRIBUTING.md](https://github.com/yeetypete/vision3d/blob/main/CONTRIBUTING.md)
for how to get started.

## License

`vision3d` is released under the
[BSD 3-Clause License](https://github.com/yeetypete/vision3d/blob/main/LICENSE).
