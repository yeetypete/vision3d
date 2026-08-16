# Top-level developer commands. See pyproject.toml for the actual build, and
# CONTRIBUTING.md for the workflow around these.

set positional-arguments

mod docs

build := 'build'
db := build / 'compile_commands.json'
uv := env('UV', 'uv')
# The torch variant to install, as a dependency group in pyproject.toml. Every
# dev shell exports the one matching the CUDA toolkit it provides.
torch_group := env('TORCH_GROUP', 'torch213-cu132')
# Requires clang-tidy 22 or newer, which the dev shell supplies. Override with
# CLANG_TIDY=<binary> to use another.
clang_tidy := env('CLANG_TIDY', 'run-clang-tidy')

# List the available recipes.
default:
    @just --list --list-submodules

# The torch variants are mutually exclusive, so `--all-groups` cannot be used.
[doc('Sync the Python environment to uv.lock.')]
sync:
    {{ uv }} sync --locked --all-extras --no-default-groups \
        --group dev --group docs --group {{ torch_group }}

# Type check the Python sources.
typecheck:
    {{ uv }} run pyrefly check

# Type check the Python sources and run clang-tidy over the native ones.
lint: sync typecheck tidy

# Extra arguments go to pytest, e.g. `just test -m cpu -x`.
[doc('Run the test suite.')]
test *args: sync
    {{ uv }} run pytest "$@"

# The database is regenerated first rather than tracked for staleness, since
# ninja already recompiles only what changed.
[doc('Run clang-tidy on C++/CUDA sources.')]
tidy: compile-db
    #!/usr/bin/env bash
    set -euo pipefail
    args=()
    # clang cannot parse .cu files against the CCCL the toolkit bundles, so
    # point it at the newer one the dev shell exports.
    if [ -n "${CCCL_INCLUDE_DIRS:-}" ]; then
        IFS=: read -ra dirs <<<"$CCCL_INCLUDE_DIRS"
        for dir in "${dirs[@]}"; do
            args+=("-extra-arg-before=-isystem$dir")
        done
    fi
    # clang cannot parse nvcc's -gencode, and RemovedArgs in .clang-tidy
    # matches literally, so it cannot strip a flag whose value depends on the
    # GPU the build saw. Read the values back out of the database instead.
    while IFS= read -r gencode; do
        args+=("-removed-arg=$gencode")
    done < <(grep -oh -- '-gencode=[^ "]*' '{{ db }}' | sort -u)
    {{ clang_tidy }} -quiet -hide-progress -p '{{ build }}' "${args[@]}" \
        'src/vision3d/ops/csrc/.*\.(cpp|cu)$'

# `build_ext` writes `build/build.ninja`, and `ninja -t compdb` turns it into
# the database.
[doc('Regenerate the compile database clangd and clang-tidy read.')]
compile-db: sync
    {{ uv }} run python setup.py --quiet build_ext --build-temp '{{ build }}'
    ninja -C '{{ build }}' -t compdb > '{{ db }}'

[doc('Build the release sdist and manylinux_2_28 wheel into dist/.')]
wheel:
    nix develop .#wheel --command {{ just_executable() }} _wheel

[private]
_wheel:
    {{ uv }} build
    auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --exclude '*' \
        --wheel-dir dist dist/*-linux_x86_64.whl
    rm dist/*-linux_x86_64.whl

# Remove the build directory.
clean:
    rm -rf '{{ build }}'
