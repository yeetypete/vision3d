# Top-level developer commands. See pyproject.toml for the actual build, and
# CONTRIBUTING.md for the workflow around these.

mod docs

build := 'build'
# Only the throwaway wheel goes here. The objects land in setuptools' usual
# `build/temp.<platform>-<abi>/`, so the compile database is generated from
# whatever an ordinary build already compiled instead of building twice.
lint_build := build / 'lint-wheel'
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

# Run clang-tidy over the C++/CUDA sources.
tidy:
    #!/usr/bin/env bash
    set -euo pipefail
    # Regenerate the database only when it is out of date.
    if [ ! -s '{{ db }}' ] || [ -n "$(find pyproject.toml setup.py uv.lock -newer '{{ db }}')" ]; then
        {{ just_executable() }} compile-db
    fi
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
    {{ clang_tidy }} -p '{{ build }}' "${args[@]}" 'src/vision3d/ops/csrc/.*\.(cpp|cu)$'

# Regenerate the compile database clangd and clang-tidy read.
compile-db:
    #!/usr/bin/env bash
    set -euo pipefail
    # Generated from a real build, so clang-tidy sees the flags the extension
    # is actually compiled with. --no-build-isolation is required: an isolated
    # build deletes its torch on exit, leaving every `-isystem
    # .../torch/include` in the database dangling. FORCE_CUDA covers machines
    # with a toolkit but no GPU.
    FORCE_CUDA=1 {{ uv }} build --wheel --no-build-isolation --out-dir '{{ lint_build }}'
    # An interpreter upgrade leaves the previous temp dir behind, so take the
    # one this build just wrote.
    temp=$(ls -dt '{{ build }}'/temp.*/ | head -1)
    # Written through a temporary file: a failed `ninja -t compdb` must not
    # leave a truncated database behind, which would then look newer than its
    # inputs and never be regenerated.
    (cd "$temp" && ninja -t compdb) > '{{ db }}.tmp'
    mv '{{ db }}.tmp' '{{ db }}'

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
