# Top-level developer Makefile. See pyproject.toml for the actual build.

BUILD := build
CLANG_TIDY ?= run-clang-tidy-22

.PHONY: help tidy clean-build

help:
	@echo "Targets:"
	@echo "  tidy         Run clang-tidy on C++ sources (CUDA via clangd in IDE)"
	@echo "  clean-build  Remove $(BUILD)/"
	@echo ""
	@echo "Requires clang-tidy >= 22 (override with CLANG_TIDY=<binary>)."

$(BUILD)/compile_commands.json: CMakeLists.txt
	cmake -B $(BUILD)

tidy: $(BUILD)/compile_commands.json
	$(CLANG_TIDY) -p $(BUILD) 'src/vision3d/ops/csrc/.*\.cpp$$'

clean-build:
	rm -rf $(BUILD)
