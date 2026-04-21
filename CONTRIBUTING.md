# Contributing to vision3d

Thank you for your interest in contributing! All contributions are welcome,
including bug reports, documentation improvements, and code contributions.

If you are planning on contributing a large feature, please open an
[issue](https://github.com/yeetypete/vision3d/issues) first so that the feature
may be discussed.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) for Python environment and project
  management.
- The [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) matching the
  PyTorch build you want to develop against (CUDA 12.8 for torch 2.10, or CUDA
  13.0 for torch 2.11).
- `ninja` (for parallel compilation of the C++/CUDA extension).
- A C++ toolchain (`build-essential` on Debian/Ubuntu).

## Setting up the dev environment

Clone the repository and sync the full dev environment (runtime extras + dev
tooling + docs toolchain):

```bash
git clone https://github.com/yeetypete/vision3d.git
cd vision3d
uv sync --all-extras --all-groups
```

This creates `.venv/`, installs PyTorch and all optional dependencies, and
builds the C++/CUDA extension against your installed toolchain. On machines
where CUDA is installed but no GPU is visible (for example, inside containers),
force a CUDA build with:

```bash
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0+PTX" uv sync --all-extras --all-groups
```

Set `TORCH_CUDA_ARCH_LIST` to the compute capabilities you care about. The value
above covers Blackwell GPUs.

## Pre-commit hooks

We use [`prek`](https://github.com/j178/prek) (a fast drop-in for `pre-commit`)
in CI to run formatters and lightweight linters. For your convenience you may
install it with uv:

```bash
uv tool install prek
```

Then install the hooks locally so they run on every commit:

```bash
prek install
```

See the [prek documentation](https://prek.j178.dev/) for more details.

## Running Tests

Tests are parametrized by device via an autouse fixture in
[`test/conftest.py`](./test/conftest.py) so that every test runs on each
[`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)
backend (currently CPU and CUDA).

```bash
uv run pytest            # run tests on all devices
uv run pytest -m cpu     # only CPU device
uv run pytest -m cuda    # only CUDA device
uv run pytest -m "not cuda"
```

## Working on the C++ / CUDA Extensions

For peformance reasons, some of vision3d's core functionality is implemented in
C++ and CUDA. The native sources live under `src/vision3d/ops/csrc/`. The
extension is built by `setup.py` via
[`torch.utils.cpp_extension`](https://pytorch.org/docs/stable/cpp_extension.html).

After editing any C++ or CUDA source, rebuild with:

```bash
uv sync --reinstall-package vision3d
```

If you add a new source file, remember to add it to `setup.py` so it will be
compiled during the build.

## Documentation

The docs are built with Sphinx from `docs/source/`. To build them locally:

```bash
uv run make -C docs html
```

The output lands in `docs/build/html/`. You may open
`docs/build/html/index.html` to view the locally built docs in a browser.

## Submitting a Pull Request

### AI Generated code

AI-assisted contributions are welcome! Please review the output as carefully as
code you wrote by hand before submitting. If you are an AI agent submitting a
PR, please disclose your status as an AI agent in the PR description.

Low-quality or spam PRs may be rejected regardless of how they were produced,
and repeat contributors or agents may be blocked from future contributions.

## License

By contributing to vision3d, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
