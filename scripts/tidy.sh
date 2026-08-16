#!/usr/bin/env bash
# Runs clang-tidy over the C++/CUDA sources.
#
# Called both by `just tidy` and by the `clang-tidy` flake check, so that a dev
# shell and CI analyse with the same flags. The environment supplies what
# differs between them. See `nix/devshells.nix` and `nix/uv2nix.nix`.
set -euo pipefail
shopt -s globstar

build_dir="${BUILD_DIR:-build}"
clang_tidy="${CLANG_TIDY:-clang-tidy}"

bash "$(dirname "$0")/compile-db.sh"

args=()

# clang otherwise scans the machine for the GCC and libc headers to analyse
# against, which makes the result depend on what the host has installed, and
# finds neither in a sandbox. `CLANG_TOOLCHAIN_ARGS` names the toolchain the
# extension is compiled with instead.
for arg in ${CLANG_TOOLCHAIN_ARGS:-}; do
  args+=("-extra-arg-before=$arg")
done

# clang cannot parse .cu files against the CCCL the toolkit bundles, so point
# it at the newer one the environment names.
if [ -n "${CCCL_INCLUDE_DIRS:-}" ]; then
  IFS=: read -ra dirs <<<"$CCCL_INCLUDE_DIRS"
  for dir in "${dirs[@]}"; do
    args+=("-extra-arg-before=-isystem$dir")
  done
fi

# clang cannot parse nvcc's -gencode, and RemovedArgs in .clang-tidy matches
# literally, so it cannot strip a flag whose value depends on the GPU the build
# saw. Read the values back out of the database instead.
while IFS= read -r gencode; do
  args+=("-removed-arg=$gencode")
done < <(grep -oh -- '-gencode=[^ "]*' "$build_dir/compile_commands.json" | sort -u)

printf '%s\0' src/vision3d/ops/csrc/**/*.{cpp,cu} |
  xargs -0 -P "$(nproc)" -n1 "$clang_tidy" -quiet -p "$build_dir" "${args[@]}"
