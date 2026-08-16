#!/usr/bin/env bash
# Fails if a wheel carries a run path.
#
# The loader consults `DT_RUNPATH`/`DT_RPATH` and nothing else in a wheel, so a
# store path left in one is the leak that can select the wrong library on
# someone else's machine. The wrapped linker writes both the output's own lib
# directory and one entry per store directory it links against. `nix/uv2nix.nix`
# turns those off, and this checks that they stayed off.
#
# Absolute paths in debug info and in `__FILE__` are not checked: they are inert
# strings, and auditwheel's --strip already removes most of them.
set -euo pipefail

wheel="$1"

unpacked=$(mktemp -d)
trap 'rm -rf "$unpacked"' EXIT
unzip -q "$wheel" -d "$unpacked"

status=0
while IFS= read -r -d '' object; do
  if run_path=$(readelf -d "$object" | grep -E 'RUNPATH|RPATH'); then
    echo "error: ${object#"$unpacked"/} carries a run path:" >&2
    echo "  $(echo "$run_path" | tr -s ' ')" >&2
    status=1
  fi
done < <(find "$unpacked" -name '*.so' -print0)

exit "$status"
