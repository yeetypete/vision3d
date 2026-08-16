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
- For building from source: [Nix](https://nixos.org/download/), which supplies
  the toolchain, or a
  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) matching your
  PyTorch build.

## Installation

`vision3d` is published on PyPI as a pre-built wheel and sdist.
The wheel is built against the
[LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
and statically links the CUDA runtime, so one wheel works for any
Python 3.12+, torch 2.10+, and any
[NVIDIA driver that supports CUDA 12.8 or newer](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
(Linux driver ≥ 570).

### From PyPI

We recommend using [`uv`](https://docs.astral.sh/uv/) as your package manager:

```bash
uv add vision3d
```

Or with `pip`:

```bash
pip install vision3d
```

### From source

Clone the repository, enter the dev shell, and sync the environment:

```bash
git clone https://github.com/yeetypete/vision3d.git
cd vision3d
nix develop # or `direnv allow`, once
uv sync --all-extras
```

`nix develop` supplies `uv`, the CUDA toolkit and a matching host compiler from
the checked-in [flake](https://github.com/yeetypete/vision3d/blob/main/flake.nix),
so none of them have to be installed
system-wide. Without Nix, these dependencies need to be installed manually.

`uv sync` compiles the C++/CUDA extension as part of installing the project,
targeting the GPUs it can see. See
[CONTRIBUTING.md](https://github.com/yeetypete/vision3d/blob/main/CONTRIBUTING.md)
for how to target a specific GPU on a machine with no GPU.

To produce a wheel locally:

```bash
uv build
```

`uv build` resolves in an isolated environment that reads neither `uv.lock` nor
the dependency group pinning the dev shell's torch, so it takes torch from PyPI,
which may not match the toolkit at hand. To build against a specific CUDA
version, name the index:

```bash
uv build --index https://download.pytorch.org/whl/cu132
```

`just wheel` automatically builds the release wheel in a shell pinned to the
oldest (CUDA, torch) pair we support, so releases build reproducibly.

### Extras

- `viz`: pulls in `rerun-sdk` for the visualization utilities in `vision3d.viz`.

Request it at install time, for example: `uv add 'vision3d[viz]'`.

## Contributing

Contributions are welcome! See
[CONTRIBUTING.md](https://github.com/yeetypete/vision3d/blob/main/CONTRIBUTING.md)
for how to get started.

## License

`vision3d` is released under the
[BSD 3-Clause License](https://github.com/yeetypete/vision3d/blob/main/LICENSE).
