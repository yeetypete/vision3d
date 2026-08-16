# The CUDA toolkit outputs the extension compiles against, joined under a single
# prefix because `torch.utils.cpp_extension` and clang both want one. Only the
# compiler and headers are included, since the torch wheels carry the CUDA
# runtime libraries and setup.py links cudart statically.
{
  lib,
  symlinkJoin,
  cudaPackages,
  # Compiler nvcc shells out to for host passes. nixpkgs bakes
  # `compiler-bindir` into nvcc.profile, so without this nvcc ignores whatever
  # gcc is on PATH and mixes two libstdc++ versions into one shared object.
  hostCC ? null,
}:
let
  # nixpkgs ships CUDA as separate redistributables, some of them
  # multi-output. `static` is left out because nothing links it.
  outputsOf = p: map (out: p.${out}) (lib.filter (out: out != "static") p.outputs);
in
symlinkJoin {
  name = "cuda-merged-${cudaPackages.cudaMajorMinorVersion}";
  paths = lib.concatMap outputsOf (
    [
      cudaPackages.cuda_nvcc
      cudaPackages.cuda_cudart
      cudaPackages.cccl
      # Workaround: clang's CUDA wrapper header includes
      # <curand_mtgp32_kernel.h> unconditionally, so .cu files do not parse
      # without it.
      cudaPackages.libcurand
    ]
    # Supplies <crt/host_config.h>, which CUDA 13 split out of cuda_nvcc. Older
    # toolkits keep the attribute but as a stub that fails to evaluate.
    ++ lib.optionals (lib.versionAtLeast cudaPackages.cudaMajorMinorVersion "13") [
      cudaPackages.cuda_crt
    ]
  );
  postBuild = ''
    # torch passes -L$CUDA_HOME/lib64, which only a real CUDA install has.
    ln -s $out/lib $out/lib64
  ''
  + lib.optionalString (hostCC != null) ''
    rm $out/bin/nvcc.profile
    sed 's|^compiler-bindir = .*|compiler-bindir = ${hostCC}/bin|' \
      ${cudaPackages.cuda_nvcc}/bin/nvcc.profile > $out/bin/nvcc.profile
  '';
}
