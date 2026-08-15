# Development environment for vision3d. Nix supplies the system toolchain
# (uv, CUDA, LLVM, ninja). uv still manages the Python environment.
#
# See CONTRIBUTING.md for more details.
{
  description = "vision3d - a 3D extension of torchvision";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";

    # Every packaged CUDA toolkit bundles a CCCL older than 3.4, and nixpkgs
    # has no standalone one, so it is pinned here as a source input. See where
    # CCCL_INCLUDE_DIRS is set for why a newer CCCL is needed.
    cccl = {
      url = "github:NVIDIA/cccl/v3.4.2";
      flake = false;
    };

    # The manylinux_2_28 build environment for the release wheel. See
    # `manylinux` below.
    kernels = {
      url = "github:huggingface/kernels";
      flake = false;
    };

    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        ./nix/git-hooks.nix
        ./nix/llvm.nix
      ];

      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      perSystem =
        {
          config,
          lib,
          pkgs,
          system,
          ...
        }:
        let
          cuda = pkgs.cudaPackages_13;

          # nixpkgs ships CUDA as separate redistributables, some of them
          # multi-output. `static` is left out because nothing links it.
          cudaOutputs = p: map (out: p.${out}) (lib.filter (out: out != "static") p.outputs);

          # `torch.utils.cpp_extension` and clang both want a single toolkit
          # prefix, so merge the pieces the build needs. Only the compiler and
          # headers: the torch wheels carry the CUDA runtime libraries, and
          # setup.py links cudart statically.
          mkCudaHome =
            {
              c,
              # Compiler nvcc shells out to for host passes. nixpkgs bakes
              # `compiler-bindir` into nvcc.profile, so without this nvcc
              # ignores whatever gcc is on PATH and mixes two libstdc++
              # versions into one shared object.
              hostCC ? null,
            }:
            pkgs.symlinkJoin {
              name = "cuda-home-${c.cudaMajorMinorVersion}";
              paths = lib.concatMap cudaOutputs (
                [
                  c.cuda_nvcc
                  c.cuda_cudart
                  c.cccl
                  # Workaround: clang's CUDA wrapper header includes
                  # <curand_mtgp32_kernel.h> unconditionally, so .cu files do
                  # not parse without it.
                  c.libcurand
                ]
                # Supplies <crt/host_config.h>, which CUDA 13 split out of
                # cuda_nvcc. Older toolkits keep the attribute but as a stub
                # that fails to evaluate.
                ++ lib.optionals (lib.versionAtLeast c.cudaMajorMinorVersion "13") [ c.cuda_crt ]
              );
              postBuild = ''
                # torch passes -L$CUDA_HOME/lib64, which only a real CUDA
                # install has.
                ln -s $out/lib $out/lib64
              ''
              + lib.optionalString (hostCC != null) ''
                rm $out/bin/nvcc.profile
                sed 's|^compiler-bindir = .*|compiler-bindir = ${hostCC}/bin|' \
                  ${c.cuda_nvcc}/bin/nvcc.profile > $out/bin/nvcc.profile
              '';
            };

          cudaHome = mkCudaHome { c = cuda; };

          # setup.py links cudart statically, so the toolkit version sets the
          # minimum driver the wheel runs on. Build against the oldest CUDA we
          # support and let driver backward compatibility cover newer majors.
          cudaWheel = pkgs.cudaPackages_12_8;

          # The manylinux_2_28 environment itself, as RHEL 8 packages unpacked
          # from RPMs, so the wheel is built against the real glibc and
          # libstdc++ it claims to support.
          manylinux =
            let
              root = "${inputs.kernels}/nix-builder/pkgs/manylinux";
              inherit (pkgs.stdenv.hostPlatform.uname) processor;
            in
            pkgs.callPackage root { } {
              packageMetadata = lib.importJSON "${root}/manylinux-2.28-${processor}-metadata.json";
            };

          # gcc 13 instead of 14, because torch's host-compiler table caps
          # CUDA 12.8 at gcc < 14 before torch 2.12, and `uv build` resolves
          # whatever torch the CUDA index offers.
          wheelStdenv = manylinux.gcc13Stdenv;
        in
        {
          _module.args.pkgs = lib.fix (
            pkgs:
            import inputs.nixpkgs {
              inherit system;
              config.allowUnfreePredicate = pkgs._cuda.lib.allowUnfreeCudaPredicate;
            }
          );

          # nvcc rejects a host gcc newer than the toolkit supports, and the
          # nixpkgs default moves independently of what NVIDIA allows.
          # `backendStdenv` is the gcc this toolkit was built against.
          devShells.default = (pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
            # Writes `.pre-commit-config.yaml` and installs the git hook automatically
            # when entering the shell. See `nix/git-hooks.nix` for details.
            inherit (config.pre-commit) shellHook;
            packages = [
              pkgs.uv
              pkgs.just
              config.llvmPackages.clang-unwrapped
              pkgs.ninja
              cudaHome
            ]
            ++ config.pre-commit.settings.enabledPackages;
            env = {
              CUDA_HOME = "${cudaHome}";
              # Workaround: the CCCL bundled with the toolkit annotates the
              # <cuda/std/string_view> deduction guides __host__-only, which
              # clang rejects outright. That breaks clang-tidy on any file
              # reaching <cub/...>, so it is overridden with CCCL 3.4, the
              # first release carrying the fix. See:
              #   - https://github.com/NVIDIA/cccl/issues/7896
              #   - https://github.com/llvm/llvm-project/pull/168711
              CCCL_INCLUDE_DIRS = "${inputs.cccl}/libcudacxx/include:${inputs.cccl}/cub:${inputs.cccl}/thrust";
            };
          };

          # Builds a vision3d release wheel for manylinux_2_28. The wheel is
          # built against the oldest CUDA we support, so it runs on any newer driver.
          # Linting and the compile database stay in `default`, so this needs only uv
          # and the toolchain.
          devShells.wheel =
            let
              cudaHomeWheel = mkCudaHome {
                c = cudaWheel;
                hostCC = wheelStdenv.cc;
              };
            in
            (pkgs.mkShell.override { stdenv = wheelStdenv; }) {
              packages = [
                pkgs.uv
                pkgs.ninja
                # Retags the wheel manylinux, and refuses to if the binary does
                # not satisfy the tag.
                pkgs.auditwheel
                cudaHomeWheel
              ];
              env.CUDA_HOME = "${cudaHomeWheel}";
            };
        };
    };
}
