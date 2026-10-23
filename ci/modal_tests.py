"""Run vision3d tests on a Modal GPU.

Usage::

    uv run --only-group modal modal run ci/modal_tests.py

Requires Modal credentials, either from ``modal token new`` or the
``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` environment variables.

See: https://modal.com/docs/examples/ci-on-modal
"""

import functools
import os
import subprocess
from pathlib import Path

import modal

_REMOTE_ROOT = "/root/vision3d"
_ROOT = Path(__file__).resolve().parents[1]
_PYTHON = os.environ.get("UV_PYTHON") or (_ROOT / ".python-version").read_text().strip()


@functools.cache
def _git_files() -> frozenset[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return frozenset(Path(p) for p in out.split("\0") if p)


image = (
    modal.Image.debian_slim(python_version=_PYTHON)
    .uv_pip_install("uv==0.13.5")
    .env(
        {
            "UV_LINK_MODE": "copy",
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
            # Overrides `.python-version` copied into the image.
            "UV_PYTHON": _PYTHON,
        }
    )
    .workdir(_REMOTE_ROOT)
    .add_local_file(_ROOT / "pyproject.toml", f"{_REMOTE_ROOT}/", copy=True)
    .add_local_file(_ROOT / "uv.lock", f"{_REMOTE_ROOT}/", copy=True)
    # `--frozen` installs from the URLs in `uv.lock` without re-locking against
    # Modal's PyPI mirror, which the builder sets in `UV_INDEX_URL`.
    .run_commands("uv sync --frozen --all-extras --no-install-project")
    .add_local_dir(
        _ROOT, _REMOTE_ROOT, copy=True, ignore=lambda p: p not in _git_files()
    )
    .run_commands("uv sync --frozen --all-extras")
)

app = modal.App("vision3d-tests", image=image)


@app.function(gpu="L4", cpu=8, timeout=30 * 60)
def run_tests() -> None:
    """Run vision3d tests."""
    subprocess.run(["uv", "run", "--frozen", "--all-extras", "pytest"], check=True)
