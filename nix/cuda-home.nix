# uv installs every CUDA wheel into one nvidia/cu13 tree, and nvcc finds
# headers, CCCL and cicc relative to its own path within it. Nix keeps each
# wheel in its own store path, so join them back into a single CUDA_HOME.
{ symlinkJoin, pythonSet }:
symlinkJoin {
  name = "cuda-home";
  paths = map (dep: "${dep}/${pythonSet.python.sitePackages}/nvidia/cu13") [
    pythonSet.nvidia-cuda-cccl
    pythonSet.nvidia-cuda-crt
    pythonSet.nvidia-cuda-nvcc
    pythonSet.nvidia-cuda-runtime
    pythonSet.nvidia-nvvm
  ];
}
