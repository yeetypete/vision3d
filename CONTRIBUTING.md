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

Set `TORCH_CUDA_ARCH_LIST` to the compute capabilities you require. The value
above covers Blackwell GPUs.

### Using a different CUDA toolkit version

`uv.lock` pins torch to the variant published on PyPI (currently the `cu130`
build). If your local CUDA toolkit is a different major version vision3d will
not build. To resolve this, point uv at the PyTorch wheel index which matches
your installed CUDA toolkit during sync:

```bash
uv sync --all-extras --all-groups --index https://download.pytorch.org/whl/cu128
```

Replace `cu128` with whatever CUDA major version your installed CUDA toolkit
ships, e.g. `cu130`, `cu132`.

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

## Linting, formatting, and type checking

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and
[`pyrefly`](https://pyrefly.org/) for type checking. All three run in CI and
must be clean on a PR. You may run them locally via `uv`:

```bash
uv run ruff check             # lint
uv run ruff format            # auto-format (writes changes)
uv run ruff format --check    # check-only; fails if formatting is off
uv run pyrefly check          # type check
```

## Running tests

Tests are parametrized by device via an autouse fixture in
[`test/conftest.py`](./test/conftest.py) so that every test runs on each
[`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)
backend. Currently CPU and CUDA are supported.

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

Editing any C++ or CUDA source rebuilds the extension on the next sync. If a
change does not seem to have been picked up, sync explicitly:

```bash
uv sync --reinstall-package vision3d
```

If you add a new source file, remember to add it to `setup.py` so it will be
compiled during the build.

`clang-tidy` can be set to run on the C++/CUDA sources via the `make tidy` target:

```bash
make tidy
```

## Documentation

The docs are built with Sphinx from [`docs/source/`](./docs/source/). To build
them locally:

```bash
uv run make -C docs html
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

Low-quality or spam PRs may be rejected regardless of how they were produced,
and repeat offenders may be blocked from future contributions.

### Pull request size

Large PRs are harder to review carefully and take longer to merge. See Google's
[Small CLs] guide for concrete size guidance and the benefits of keeping a
change small.

Splitting "preparatory" commits that are independently reviewable out of the
main PR and submitting them separately is strongly encouraged. Each commit
should be reviewable on its own. Where the parts depend on each other, submit
them as a stack of dependent PRs, for example with the
[gh-stack](https://github.com/github/gh-stack) extension for the GitHub CLI.

If a change is more than roughly 500 lines of substantial (non-whitespace) code,
please file a design [issue](https://github.com/yeetypete/vision3d/issues) first
so that the approach can be reviewed by the maintainers before it is
implemented.

## AI-assisted contributions

AI coding tools are a useful part of a modern developer's toolbox and we
encourage you to use them. This section sets expectations for using them here.
The guidance below is inspired by Oxide's [RFD 576] on using LLMs, and by the
conventions used in [bootc] and [chunkah].

### You own what you submit

However the code was produced, the contributor bears full responsibility for it.
AI is a tool and the human using it is accountable for the result. **This means
you MUST understand every line of code you submit** and be able to explain the
design decisions behind it.

No part of this codebase should be understandable only to an LLM. Comments and
docstrings especially are written for human readers: they should explain why the
code does what it does, and give a reader context that is not already obvious
from the code itself. See the [Google Python Style Guide] for guidance on writing
clear docstrings and comments.

### Self-review is mandatory

**DO NOT request reviews on AI-generated code you have not reviewed yourself
first.** Check for correctness and clarity, as carefully as you would check code
you wrote by hand. If you wouldn't be comfortable defending a line of code in
review, don't submit it. Leaving obvious AI slop for reviewers to catch wastes
their time and erodes trust. An LLM produces a plausible diff faster than anyone
can review one, so it is easy for the work of getting a change into shape to
slide onto the reviewer. That work is part of the contribution, not the review.

### Disclose AI usage

Disclose AI involvement with a trailer on the commit and in the PR description:
`Assisted-by: AI` when a change contains substantial assistance, and
`Generated-by: AI` when it is effectively entirely generated. Do not name the
tool or model as that can be considered a form of advertising. Also outline
your level of review and domain knowledge in the PR description, for example:

```text
Assisted-by: AI
Unit tests are LLM generated.
```

```text
Generated-by: AI
I am knowledgeable in this problem domain and reviewed it carefully.
```

If you are an AI agent submitting a PR, please disclose your status as an AI
agent in the PR description.

This isn't about shaming or gatekeeping. Every commit in vision3d is held to the
same standard, regardless of how it was created. Disclosure is about providing
useful context to make review easier, and helping us learn collectively about
where AI assistance works well and where it doesn't.

### The AGENTS.md file

The [AGENTS.md](AGENTS.md) file provides detailed guidance to AI coding agents
working in this repository, including project overview, contribution workflow,
and code conventions. If you use an AI coding agent, make sure it has access to
this file.

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

## License

By contributing to vision3d, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

[bootc]: https://github.com/bootc-dev/bootc/blob/main/AGENTS.md
[chunkah]: https://github.com/coreos/chunkah/blob/main/CONTRIBUTING.md
[Google Python Style Guide]: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[RFD 576]: https://rfd.shared.oxide.computer/rfd/0576
[Small CLs]: https://google.github.io/eng-practices/review/developer/small-cls.html
