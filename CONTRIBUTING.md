# Contributing to vision3d

Thank you for your interest in contributing! All contributions are welcome,
including bug reports, documentation improvements, and code contributions.

If you are planning on contributing a large feature, please open an
[issue](https://github.com/yeetypete/vision3d/issues) first so that the feature
may be discussed.

## Prerequisites

- [Nix](https://nixos.org/download/), with flakes enabled.
- Optionally [`direnv`](https://direnv.net/), to enter the dev shell
  automatically.

## Setting up the dev environment

Clone the repository, enter the dev shell, and sync the full Python environment
(runtime extras + dev tooling + docs toolchain):

```bash
git clone https://github.com/yeetypete/vision3d.git
cd vision3d
nix develop  # or `direnv allow`, once
just sync
```

Nix supplies the system toolchain. `uv` still manages the Python
environment. The toolkit is always present, but a GPU may not be, so the shell
exports `FORCE_CUDA=1` to compile the CUDA sources either way, and
`TORCH_CUDA_ARCH_LIST` to explicitely set the CUDA architectures to build for.
See `cudaCapabilities` and `cudaForwardCompat` in [`nix/cuda.nix`](./nix/cuda.nix).

### Torch and CUDA variants

The toolkit that builds the extension must be the one torch was built against,
so each dev shell pairs a toolkit with a torch build. `variants` in
[`nix/cuda.nix`](./nix/cuda.nix) lists the pairs and `pyproject.toml` carries a
dependency group per pair, enabled by default. The dev shell disables the groups
its toolkit does not pair with, so that every `uv` command run inside it correctly
installs the torch version matching the dev shell's toolkit. To switch to a
different pair, run:

```bash
nix develop .#torch211-cu128  # or .#torch212-cu130, .#torch213-cu132
just sync
```

CI runs the test suite in every variant. A plain `nix develop` gives the newest
pair (`torch213-cu132`).

To add or move a pair, edit `variants`, add the matching group to
`pyproject.toml`, and relock with `uv lock`. Note that the PyTorch wheel index
publishes only some (toolkit, torch) combinations.

## Project commands

We use [`just`](https://just.systems/) as our task runner. Recipes are declared
in the [`justfile`](./justfile) and, for Sphinx, in
[`docs/justfile`](./docs/justfile). To see what is available:

```bash
just
```

## Pre-commit hooks

Formatters and lightweight linters run as git hooks, installed when you enter
the dev shell. They are declared in [`nix/git-hooks.nix`](./nix/git-hooks.nix)
and rendered to a generated, gitignored `.pre-commit-config.yaml`. Edit the
flake module to change a hook.

To run every hook over the whole tree:

```bash
nix fmt
```

Some hooks shell out to `uv run`, so they need a synced environment and cannot
run under `nix flake check`, which builds in a sandbox with no network. Use
`nix fmt` or the git hook instead.

## Linting, formatting, and type checking

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting,
[`pyrefly`](https://pyrefly.org/) for type checking the Python sources, and
[`clang-tidy`](https://clang.llvm.org/extra/clang-tidy/) for C++/CUDA.
All of them run in CI and must be clean on a PR.

`just lint` syncs the environment, then runs `pyrefly` and `clang-tidy` over the
whole project:

```bash
just lint                       # from the dev shell
nix develop --command just lint # or from outside it
```

You may also run any of them directly:

```bash
uv run pyrefly check          # type check
just tidy                     # clang-tidy
```

## Running tests

Tests are parametrized by device via an autouse fixture in
[`test/conftest.py`](./test/conftest.py) so that every test runs on each
[`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)
backend. Currently CPU and CUDA are supported.

```bash
just test                # run tests on all devices
just test -m cpu         # only CPU device
just test -m cuda        # only CUDA device
just test -m "not cuda"
```

## Working on the C++ / CUDA extensions

For peformance reasons, some of vision3d's core functionality is implemented in
C++ and CUDA. The native sources live under `src/vision3d/ops/csrc/`. The
extension is built by `setup.py` via
[`torch.utils.cpp_extension`](https://pytorch.org/docs/stable/cpp_extension.html).

Editing any C++ or CUDA source rebuilds the extension on the next sync. If a
change does not seem to have been picked up, sync explicitly:

```bash
just sync
```

If you add a new source file, remember to add it to `setup.py` so it will be
compiled during the build.

`clang-tidy` runs over the same sources with:

```bash
just tidy
```

## Documentation

The docs are built with Sphinx from [`docs/source/`](./docs/source/). To build
them locally:

```bash
just docs html      # build once
just docs livehtml  # live preview, rebuilds on save
```

The output lands in `docs/build/html/`. You may open
`docs/build/html/index.html` in the browser to view the locally built docs.

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

## Creating a release (maintainers only)

The project follows [Semantic Versioning](https://semver.org/). Releases are
created via the
[Release workflow](https://github.com/yeetypete/vision3d/actions/workflows/release.yaml)
([source](./.github/workflows/release.yaml)) from the Actions UI:

1. Bump `project.version` in `pyproject.toml` to the new version (e.g. `0.1.0`
   --> `0.1.1`) in its own PR and merge to `main`. `uv version --bump patch`
   (or `minor`/`major`) does the edit for you, and `uv version` prints the
   current value.
1. From the Actions tab, run the
   [Release workflow](https://github.com/yeetypete/vision3d/actions/workflows/release.yaml)
   and pass the new version (e.g. `0.1.1`) as input. The workflow verifies the
   input matches the project version, runs lint and tests, builds the full wheel
   matrix (every supported Python + CUDA combination) plus the sdist, atomically
   creates the GitHub release with tag `v<version>` and all artifacts attached,
   and publishes the sdist to PyPI.

> [!NOTE]
> The `announce` (GitHub release creation) and `publish-pypi` jobs both
> run inside the `pypi` deployment environment, so required reviewers configured
> there gate the actual release and PyPI push.

To produce the artifacts locally:

```bash
just wheel
```

This builds the sdist and wheel in the manylinux_2_28 shell against the oldest
CUDA version we support. To build against a different one, change `cudaWheel`
in [`flake.nix`](./flake.nix).

## License

By contributing to vision3d, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
