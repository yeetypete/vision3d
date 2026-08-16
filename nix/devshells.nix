# The dev shells. One per torch/CUDA variant from `nix/cuda.nix`, plus the
# release-wheel shell.
{ inputs, ... }:
{
  perSystem =
    {
      config,
      lib,
      pkgs,
      ...
    }:
    let
      inherit (config.cuda) variants wheelToolkit;

      noDot = lib.replaceStrings [ "." ] [ "" ];

      # Names both the dev shell and the dependency group in `pyproject.toml`.
      variantName = v: "torch${noDot v.torch}-cu${noDot v.toolkit.cudaMajorMinorVersion}";

      # What `nix develop` gives you, and what `uv.lock` installs by default.
      defaultVariant = lib.last variants;

      # Force compilation of CUDA sources even when a gpu is not present.
      buildEnv = c: {
        FORCE_CUDA = "1";
        # torch wants the capabilities as a `;`-separated list, `+PTX` on the
        # one to emit PTX for (`cudaForwardCompat` configuration option).
        TORCH_CUDA_ARCH_LIST = lib.concatStringsSep ";" (
          lib.init c.flags.cudaCapabilities
          ++ [
            (lib.last c.flags.cudaCapabilities + lib.optionalString c.flags.cudaForwardCompat "+PTX")
          ]
        );
      };

      # We use uv-managed python because the nixpkgs one resolves neither
      # the libraries the manylinux wheels link against nor the NVIDIA
      # driver. Both shells put it on PATH via their python packages.
      uvOwnsPython = {
        env.UV_PYTHON_PREFERENCE = "only-managed";
        shellHook = "unset PYTHONPATH";
      };

      mkDevShell =
        v:
        let
          cudaHome = pkgs.callPackage ./cuda-home.nix { cudaPackages = v.toolkit; };
        in
        # nvcc rejects a host gcc newer than the toolkit supports, and the
        # nixpkgs default moves independently of what NVIDIA allows.
        # `backendStdenv` is the gcc this toolkit was built against.
        (pkgs.mkShell.override { stdenv = v.toolkit.backendStdenv; }) {
          # Writes `.pre-commit-config.yaml` and installs the git hook automatically
          # when entering the shell. See `nix/git-hooks.nix` for details.
          shellHook = ''
            ${config.pre-commit.shellHook}
            ${uvOwnsPython.shellHook}
          '';
          packages = [
            pkgs.uv
            pkgs.just
            config.llvmPackages.clang-unwrapped
            pkgs.ninja
            cudaHome
          ]
          ++ config.pre-commit.settings.enabledPackages;
          env =
            uvOwnsPython.env
            // buildEnv v.toolkit
            // {
              CUDA_HOME = "${cudaHome}";
              # `pyproject.toml` enables every variant group by default, so
              # disabling the ones this toolkit does not pair with leaves uv
              # holding the torch built against it, whichever command runs.
              # TODO: Select the group directly with `UV_GROUP` once uv has it:
              # https://github.com/astral-sh/uv/issues/11958
              UV_NO_GROUP = lib.concatMapStringsSep " " variantName (
                lib.filter (o: variantName o != variantName v) variants
              );
              # Workaround: the CCCL bundled with the toolkit annotates the
              # <cuda/std/string_view> deduction guides __host__-only, which
              # clang rejects outright. That breaks clang-tidy on any file
              # reaching <cub/...>, so it is overridden with CCCL 3.4, the
              # first release carrying the fix. See:
              #   - https://github.com/NVIDIA/cccl/issues/7896
              #   - https://github.com/llvm/llvm-project/pull/168711
              CCCL_INCLUDE_DIRS = "${inputs.cccl}/libcudacxx/include:${inputs.cccl}/cub:${inputs.cccl}/thrust";
              # The toolkit brings its own host compiler through
              # `backendStdenv`, and nvcc enforces it via <crt/host_config.h>,
              # so torch's redundant bounds check is disabled.
              TORCH_DONT_CHECK_COMPILER_ABI = "1";
            };
        };

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
      # One shell per variant, named after it, so `nix develop .#torch212-cu130`
      # pairs that toolkit with the torch `just sync` installs.
      devShells =
        lib.listToAttrs (map (v: lib.nameValuePair (variantName v) (mkDevShell v)) variants)
        // {
          default = mkDevShell defaultVariant;

          # Builds a vision3d release wheel for manylinux_2_28. The wheel is
          # built against the oldest CUDA we support, so it runs on any newer driver.
          # Linting and the compile database stay in `default`, so this needs only uv
          # and the toolchain.
          wheel =
            let
              cudaHomeWheel = pkgs.callPackage ./cuda-home.nix {
                cudaPackages = wheelToolkit;
                hostCC = wheelStdenv.cc;
              };
            in
            (pkgs.mkShell.override { stdenv = wheelStdenv; }) {
              packages = [
                pkgs.uv
                pkgs.just
                pkgs.ninja
                # Retags the wheel manylinux, and refuses to if the binary does
                # not satisfy the tag.
                pkgs.auditwheel
                cudaHomeWheel
              ];
              inherit (uvOwnsPython) shellHook;
              env =
                uvOwnsPython.env
                // buildEnv wheelToolkit
                // {
                  CUDA_HOME = "${cudaHomeWheel}";
                  # `uv build` resolves in an isolated environment, which reads
                  # neither `uv.lock` nor its dependency groups, so the torch
                  # matching this toolkit is selected by index instead.
                  UV_INDEX = "https://download.pytorch.org/whl/cu${noDot wheelToolkit.cudaMajorMinorVersion}";
                };
            };
        };
    };
}
