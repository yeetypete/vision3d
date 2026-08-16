#!/usr/bin/env bash
# Writes the compile database clangd and clang-tidy read.
#
# `build_ext` writes `build.ninja`, and `ninja -t compdb` turns it into the
# database. It is regenerated rather than tracked for staleness, since ninja
# already recompiles only what changed.
set -euo pipefail

build_dir="${BUILD_DIR:-build}"

python setup.py --quiet build_ext --build-temp "$build_dir"
ninja -C "$build_dir" -t compdb >"$build_dir/compile_commands.json"
