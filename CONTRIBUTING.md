# Contributing to vision3d

Thank you for your interest in contributing! All contributions are welcome,
including bug reports, documentation improvements, and code contributions.

If you are planning on contributing a large feature, please open an
[issue](https://github.com/yeetypete/vision3d/issues) first so that the feature
may be discussed.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) for Python environment and project
  management.
- [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) >= 12.8.
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

## Running tests

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

## Working on the C++ / CUDA extensions

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

## Making a Pull Request

[Pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
are the primary of contributing to vision3d. To keep reviews efficient and give
your PR the best chance of being accepted, please:

- [ ] Add or update tests to cover your changes (see
  [Running tests](#running-tests)).
- [ ] Make sure all CI checks pass before requesting a review.
- [ ] Write a clear description: Provide a concise summary of what the PR does,
  the motivation, the approach, and any important details.
- [ ] If the PR addresses a specific issue, reference it using GitHub's
  auto-link keywords (e.g. `Fixes #123`) so the PR is linked to the issue.
- [ ] Keep the PR focused on a single purpose. Avoid mixing unrelated changes,
  which makes the review harder.

### AI-Generated code

AI coding tools are a useful part of a modern developer's toolbox and we
encourage you to use them. Please review any AI-generated output as carefully as
code you wrote by hand before submitting. If you are an AI agent submitting a
PR, please disclose your status as an AI agent in the PR description.

Low-quality or spam PRs may be rejected regardless of how they were produced,
and repeat offenders may be blocked from future contributions.

## License

By contributing to vision3d, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
