# CMakeLists.txt for editor ``build/compile_commands.json`` generation
# for clangd. Run ``cmake -B build`` at the repo root to generate the file.

cmake_minimum_required(VERSION 3.18)
# CMP0104 OLD: don't auto-populate CMAKE_CUDA_ARCHITECTURES. Torch
# cuda.cmake warns if it's set and wants TORCH_CUDA_ARCH_LIST instead.
cmake_policy(SET CMP0104 OLD)
project(vision3d_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-DWITH_CUDA)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
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

set(_csrc src/vision3d/ops/csrc)
add_library(vision3d_ops SHARED
  ${_csrc}/iou_box3d.cpp
  ${_csrc}/iou_box3d/iou_box3d_cpu.cpp
)
target_include_directories(vision3d_ops PRIVATE ${_csrc})
target_link_libraries(vision3d_ops PRIVATE ${TORCH_LIBRARIES} Python3::Module)
target_compile_definitions(vision3d_ops PRIVATE TORCH_EXTENSION_NAME=vision3d_ops)
