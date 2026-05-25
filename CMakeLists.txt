# CMakeLists.txt for editor ``build/compile_commands.json`` generation
# for clangd / clang-tidy. Run ``cmake -B build`` at the repo root.
#
# NOTE: The real build is driven by setup.py (nvcc). This file only
# produces a compile-commands database for tooling.

cmake_minimum_required(VERSION 3.18)
project(vision3d_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  find_program(CMAKE_CUDA_COMPILER REQUIRED
    NAMES clang++ clang++-22 clang++-21 clang++-20)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  # Clang reports its own version as the CUDA compiler version, which makes
  # Torch's cuda.cmake error with "two conflicting CUDA versions". Override
  # with the actual toolkit version so Torch's check passes.
  set(CMAKE_CUDA_COMPILER_VERSION "${CUDAToolkit_VERSION}")
  # USE_CUDA + TORCH_TARGET_VERSION must match setup.py so the compile DB
  # reflects the same preprocessor state as the real nvcc build.
  add_definitions(-DUSE_CUDA -DTORCH_TARGET_VERSION=0x020a000000000000)
endif()

set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/.venv" CACHE PATH "")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE _torch_cmake_prefix
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
find_package(Torch REQUIRED)

# Reset CMAKE_CUDA_FLAGS to drop the nvcc-only flags Torch appended, then add
# the cccl include so clang resolves <thrust/...> (CUDA 12.6+ moved it here).
if(CUDAToolkit_FOUND)
  set(CMAKE_CUDA_FLAGS "-isystem ${CUDAToolkit_TARGET_DIR}/include/cccl")
endif()

set(_csrc src/vision3d/ops/csrc)
set(_sources
  ${_csrc}/iou_box3d.cpp
  ${_csrc}/iou_box3d/iou_box3d_cpu.cpp
  ${_csrc}/voxelize.cpp
  ${_csrc}/voxelize/voxelize_cpu.cpp
)
if(CUDAToolkit_FOUND)
  list(APPEND _sources ${_csrc}/iou_box3d/iou_box3d.cu)
endif()
add_library(vision3d_ops OBJECT ${_sources})
target_include_directories(vision3d_ops PRIVATE ${_csrc})
target_link_libraries(vision3d_ops PRIVATE ${TORCH_LIBRARIES} Python3::Module)
target_compile_definitions(vision3d_ops PRIVATE TORCH_EXTENSION_NAME=vision3d_ops)
