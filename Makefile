# Top-level developer Makefile. See pyproject.toml for the actual build.

BUILD := build
DB := $(BUILD)/compile_commands.json
UV ?= uv
CLANG_TIDY ?= run-clang-tidy-22
# Force CUDA to be enabled for the build, so that the compile database contains
# the flags for compiling .cu files.
export FORCE_CUDA ?= 1

# clang cannot parse .cu files against the CCCL a toolkit bundles, so tidy reads
# the newer one `nvidia-cuda-cccl` installs. See the pin in pyproject.toml.
ifeq ($(origin CCCL_INCLUDE_DIRS),undefined)
CCCL_INCLUDE_DIRS := $(wildcard $(CURDIR)/.venv/lib/python*/site-packages/nvidia/cu*/include/cccl)
endif

TIDY_ARGS := $(foreach d,$(subst :, ,$(CCCL_INCLUDE_DIRS)),-extra-arg-before=-I$(d))

# Remove -gencode flags from the database, since clang cannot parse them.
GENCODE_ARGS = $(addprefix -removed-arg=,\
    $(shell grep -oh -- '-gencode=[^ "]*' $(DB) | sort -u))

.PHONY: help tidy compile-db clean-build

help:
	@echo "Targets:"
	@echo "  tidy         Run clang-tidy on C++/CUDA sources"
	@echo "  compile-db   Regenerate $(DB)"
	@echo "  clean-build  Remove $(BUILD)/"
	@echo ""
	@echo "Requires clang-tidy >= 22 (override with CLANG_TIDY=<binary>)."

# Generated from a real build, so clang-tidy and clangd see the flags the
# extension is actually compiled with. `build_ext` writes `$(BUILD)/build.ninja`
# and `ninja -t compdb` turns it into compile_commands.json.
compile-db:
	$(UV) run --no-sync python setup.py --quiet build_ext --build-temp '$(BUILD)'
	ninja -C '$(BUILD)' -t compdb > '$(DB)'

tidy: compile-db
	@[ -n "$(TIDY_ARGS)" ] \
	    || { echo "No CCCL headers found. Run '$(UV) sync' first" >&2; exit 1; }
	$(CLANG_TIDY) -quiet -hide-progress -p '$(BUILD)' $(TIDY_ARGS) $(GENCODE_ARGS) \
	    'src/vision3d/ops/csrc/.*\.(cpp|cu)$$'

clean-build:
	rm -rf $(BUILD)
