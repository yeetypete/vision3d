# Top-level developer commands. See pyproject.toml for the actual build, and
# CONTRIBUTING.md for the workflow around these.

mod docs

build := 'build'
db := build / 'compile_commands.json'
uv := env('UV', 'uv')
# Requires clang-tidy 22 or newer, which the dev shell supplies. Override with
# CLANG_TIDY=<binary> to use another.
clang_tidy := env('CLANG_TIDY', 'run-clang-tidy')

# List the available recipes.
default:
    @just --list --list-submodules

# Sync the Python environment to uv.lock.
sync:
    {{ uv }} sync --locked --all-extras --all-groups

# Type check the Python sources.
typecheck:
    {{ uv }} run pyrefly check

# Type check the Python sources and run clang-tidy over the native ones.
lint: sync typecheck tidy

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
    FORCE_CUDA=1 {{ uv }} run python setup.py --quiet build_ext --build-temp '{{ build }}'
    ninja -C '{{ build }}' -t compdb > '{{ db }}'

# The PyTorch wheel index has to match the toolkit `devShells.wheel` provides,
# so the default moves with it. Pass another tag positionally to override.
[doc('Build the release sdist and manylinux_2_28 wheel into dist/.')]
wheel cuda="cu128":
    nix develop .#wheel --command {{ just_executable() }} _wheel {{ cuda }}

[private]
_wheel cuda:
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;9.0;10.0;12.0+PTX' \
        {{ uv }} build --index https://download.pytorch.org/whl/{{ cuda }}
    auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --exclude '*' \
        --wheel-dir dist dist/*-linux_x86_64.whl
    rm dist/*-linux_x86_64.whl

# Remove the build directory.
clean:
    rm -rf '{{ build }}'
