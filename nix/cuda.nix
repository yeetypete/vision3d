# Which CUDA toolkits the project is built with, and the torch builds they pair
# with. The toolkit prefix itself is a package, see `nix/cuda-home.nix`.
{ inputs, flake-parts-lib, ... }:
{
  options.perSystem = flake-parts-lib.mkPerSystemOption (
    { lib, pkgs, ... }:
    {
      options.cuda = {
        variants = lib.mkOption {
          type = lib.types.listOf (
            lib.types.submodule {
              options = {
                toolkit = lib.mkOption {
                  type = lib.types.raw;
                  description = "nixpkgs `cudaPackages` set that builds the extension.";
                };
                torch = lib.mkOption {
                  type = lib.types.str;
                  description = "torch `major.minor` published for that toolkit.";
                };
              };
            }
          );
          description = ''
            The (toolkit, torch) pairs the dev shells cover, oldest first: the
            release artifacts are built against the head of this list, since
            setup.py links cudart statically and the oldest CUDA we support sets
            the minimum driver the wheel runs on. PyPI publishes one
            CUDA build per torch release, so each pair is that release paired
            with the toolkit it was built against, rather than a crossing.
            `uv.lock` carries a dependency group per pair.
          '';
          default = [
            # 12.8 is the oldest toolkit we support, because it is the first
            # whose nvcc knows the Blackwell capabilities (`10.0`, `12.0`).
            {
              toolkit = pkgs.cudaPackages_12_8;
              torch = "2.10";
            }
            # torch moved to CUDA 13.0 in 2.11 and has stayed there since.
            {
              toolkit = pkgs.cudaPackages_13_0;
              torch = "2.11";
            }
            {
              toolkit = pkgs.cudaPackages_13_0;
              torch = "2.12";
            }
            {
              toolkit = pkgs.cudaPackages_13_0;
              torch = "2.13";
            }
          ];
        };

        variantName = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = ''
            Takes a variant and returns the name it goes by: the dev shell, the
            dependency group in `pyproject.toml`, and the environments built
            from it all use this.
          '';
        };

        torchArchList = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = ''
            Takes a variant and returns its `TORCH_CUDA_ARCH_LIST`: the
            capabilities as a `;`-separated list, `+PTX` on the one to emit PTX
            for. torch derives an empty list from the machine when a GPU is
            absent, so anything compiling the extension sets this.
          '';
        };

        clangToolchainArgs = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = ''
            Takes a variant and returns the clang flags naming the toolchain to
            analyse against: the GCC that toolkit compiles the extension with,
            and its libc headers. Left to itself clang scans the machine for
            both, which makes clang-tidy's result depend on what the host
            distribution ships, and finds neither in a sandbox.
          '';
        };

        defaultVariant = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = ''
            Pair a plain `nix develop` gives, and the one tooling outside a
            variant shell resolves against. The newest we support.
          '';
        };

      };
    }
  );

  # Wrapped in `config` because this module also declares `options`.
  config.perSystem =
    {
      config,
      lib,
      pkgs,
      system,
      ...
    }:
    {
      cuda.defaultVariant = lib.last config.cuda.variants;

      cuda.variantName =
        let
          noDot = lib.replaceStrings [ "." ] [ "" ];
        in
        v: "torch${noDot v.torch}-cu${noDot v.toolkit.cudaMajorMinorVersion}";

      cuda.torchArchList =
        v:
        let
          inherit (v.toolkit.flags) cudaCapabilities cudaForwardCompat;
        in
        lib.concatStringsSep ";" (
          lib.init cudaCapabilities
          ++ [ (lib.last cudaCapabilities + lib.optionalString cudaForwardCompat "+PTX") ]
        );

      cuda.clangToolchainArgs =
        v:
        let
          inherit (v.toolkit.backendStdenv.cc) cc libc_dev;
        in
        lib.concatStringsSep " " [
          "--gcc-install-dir=${cc}/lib/gcc/${pkgs.stdenv.hostPlatform.config}/${cc.version}"
          "-idirafter ${libc_dev}/include"
        ];

      _module.args.pkgs = lib.fix (
        pkgs:
        import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfreePredicate = pkgs._cuda.lib.allowUnfreeCudaPredicate;
            cudaCapabilities = [
              "7.5"
              "8.0"
              "8.6"
              "9.0"
              "10.0"
              "12.0"
            ];
            cudaForwardCompat = true;
          };
        }
      );
    };
}
