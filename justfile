# Top-level developer commands. See pyproject.toml for the actual build, and
# CONTRIBUTING.md for the workflow around these.

set positional-arguments

mod docs

build := 'build'

# List the available recipes.
default:
    @just --list --list-submodules

# Compiles the extension into the source tree, which an editable install does
# not do on its own. The environment itself comes from the dev shell.
[doc('Rebuild the extension in place.')]
sync:
    build-editable

# Type check the Python sources.
typecheck:
    pyrefly check

# Type check the Python sources and run clang-tidy over the native ones.
lint: typecheck tidy

# Extra arguments go to pytest, e.g. `just test -m cpu -x`.
[doc('Run the test suite.')]
test *args: sync
    pytest "$@"

# `scripts/` holds what these run, so the flake checks of the same names run
# the same thing. See `nix/uv2nix.nix`.
[doc('Run clang-tidy on C++/CUDA sources.')]
tidy:
    BUILD_DIR='{{ build }}' bash scripts/tidy.sh

[doc('Regenerate the compile database clangd and clang-tidy read.')]
compile-db:
    BUILD_DIR='{{ build }}' bash scripts/compile-db.sh

# See `nix/uv2nix.nix` for what the derivation builds.
[doc('Build the release sdist and manylinux_2_28 wheel, linked as result/.')]
wheel:
    nix build '.#dist'

# Remove the build directory.
clean:
    rm -rf '{{ build }}'
