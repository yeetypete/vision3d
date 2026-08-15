# Development environment for vision3d. Nix supplies the system toolchain
# (uv, CUDA, LLVM, ninja). uv still manages the Python environment.
#
# See CONTRIBUTING.md for more details.
{
  description = "vision3d - a 3D extension of torchvision";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Every packaged CUDA toolkit bundles a CCCL older than 3.4, and nixpkgs
    # has no standalone one, so it is pinned here as a source input. See where
    # CCCL_INCLUDE_DIRS is set for why a newer CCCL is needed.
    cccl = {
      url = "github:NVIDIA/cccl/v3.4.2";
      flake = false;
    };
  };

  outputs =
    { nixpkgs, cccl, ... }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems =
        f:
        lib.genAttrs systems (
          system:
          f {
            inherit system;
            pkgs = lib.fix (
              pkgs:
              import nixpkgs {
                inherit system;
                config.allowUnfreePredicate = pkgs._cuda.lib.allowUnfreeCudaPredicate;
              }
            );
          }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs, ... }:
        let
          cuda = pkgs.cudaPackages_13;

          # nixpkgs ships CUDA as separate redistributables, some of them
          # multi-output, while `torch.utils.cpp_extension` and clang both want
          # a single toolkit prefix. Merge the components the build needs,
          # minus the `static` outputs, which nothing links.
          cudaOutputs = p: map (out: p.${out}) (lib.filter (out: out != "static") p.outputs);

          # Only the compiler and headers are needed. The torch wheels already
          # carry the CUDA runtime libraries, and setup.py links cudart
          # statically.
          cudaHome = pkgs.symlinkJoin {
            name = "cuda-home-${cuda.cudaMajorMinorVersion}";
            paths = lib.concatMap cudaOutputs [
              cuda.cuda_nvcc
              cuda.cuda_cudart
              cuda.cccl
              # Supplies <crt/host_config.h>, which CUDA 13 split out of
              # cuda_nvcc.
              cuda.cuda_crt
              # clang's CUDA wrapper header includes <curand_mtgp32_kernel.h>
              # unconditionally, so .cu files do not parse without it.
              cuda.libcurand
            ];
            postBuild = ''
              # CMake derives the --cuda-path it passes to clang from
              # NVVMIR_LIBRARY_DIR, which nixpkgs points at the cuda_nvcc
              # output, a tree with no include/. Restoring the relocatable
              # spelling NVIDIA ships upstream makes it resolve inside this
              # merged tree instead. See:
              #   - https://github.com/NixOS/nixpkgs/issues/224291
              rm $out/bin/nvcc.profile
              sed 's|^NVVMIR_LIBRARY_DIR = .*|NVVMIR_LIBRARY_DIR = $(TOP)/nvvm/libdevice|' \
                ${cuda.cuda_nvcc}/bin/nvcc.profile > $out/bin/nvcc.profile

              # torch passes -L$CUDA_HOME/lib64, which only a real CUDA
              # install has.
              ln -s $out/lib $out/lib64
            '';
          };
        in
        {
          # nvcc rejects a host gcc newer than the toolkit supports, and the
          # nixpkgs default gcc moves independently of what NVIDIA allows.
          # `backendStdenv` is the gcc this CUDA package set was built against,
          # so the shell always has one nvcc accepts.
          default = (pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
            packages = [
              pkgs.uv
              pkgs.llvmPackages_22.clang-unwrapped
              pkgs.ninja
              cudaHome
            ];
            env = {
              CUDA_HOME = "${cudaHome}";
              # The CCCL bundled with the toolkit annotates the
              # <cuda/std/string_view> deduction guides __host__-only, which
              # clang rejects outright. That breaks clang-tidy on any file
              # reaching <cub/...>, so it is overridden with CCCL 3.4, the
              # first release carrying the fix. See:
              #   - https://github.com/NVIDIA/cccl/issues/7896
              #   - https://github.com/llvm/llvm-project/pull/168711
              CCCL_INCLUDE_DIRS = "${cccl}/libcudacxx/include:${cccl}/cub:${cccl}/thrust";
            };
          };
        }
      );
    };
}
