# Top-level developer Makefile. See pyproject.toml for the actual build.

BUILD := build
# Only the throwaway wheel goes here. The objects land in setuptools' usual
# `build/temp.<platform>-<abi>/`, so `make tidy` reuses whatever an ordinary
# build already compiled instead of building everything twice.
LINT_BUILD := $(BUILD)/lint-wheel
CLANG_TIDY ?= run-clang-tidy
UV ?= uv

# clang cannot parse .cu files against the CCCL the toolkit bundles, so point
# it at the newer one `nix develop` exports.
TIDY_ARGS := $(foreach d,$(subst :, ,$(CCCL_INCLUDE_DIRS)),-extra-arg-before=-isystem$(d))

# clang cannot parse nvcc's -gencode, and RemovedArgs in .clang-tidy matches
# literally, so it cannot strip a flag whose value depends on the GPU the build
# saw. Read the values back out of the database instead.
# Assigned with `=` so it expands in the recipe, once the database exists.
GENCODE_ARGS = $(addprefix -removed-arg=,\
    $(shell grep -oh -- '-gencode=[^ "]*' $(BUILD)/compile_commands.json | sort -u))

.PHONY: help tidy clean-build

help:
	@echo "Targets:"
	@echo "  tidy         Run clang-tidy on C++/CUDA sources"
	@echo "  clean-build  Remove $(BUILD)/"
	@echo ""
	@echo "Requires clang-tidy >= 22 (override with CLANG_TIDY=<binary>)."
	@echo "Run under 'nix develop', which provides it."

# Generated from a real build, so clang-tidy sees the flags the extension is
# actually compiled with. --no-build-isolation is required: an isolated build
# deletes its torch on exit, leaving every `-isystem .../torch/include` in the
# database dangling. FORCE_CUDA covers machines with a toolkit but no GPU.
$(BUILD)/compile_commands.json: pyproject.toml setup.py
	FORCE_CUDA=1 $(UV) build --wheel --no-build-isolation --out-dir $(LINT_BUILD)
	cd $(BUILD)/temp.*/ && ninja -t compdb > $(CURDIR)/$@

tidy: $(BUILD)/compile_commands.json
	$(CLANG_TIDY) -p $(BUILD) $(TIDY_ARGS) $(GENCODE_ARGS) \
	    'src/vision3d/ops/csrc/.*\.(cpp|cu)$$'

clean-build:
	rm -rf $(BUILD)
