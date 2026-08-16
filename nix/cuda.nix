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
            The (toolkit, torch) pairs the dev shells cover. The PyTorch wheel
            index publishes only some combinations, so the pairs are listed
            rather than crossed. `uv.lock` carries a dependency group per pair.
          '';
          default = [
            {
              toolkit = pkgs.cudaPackages_12_8;
              torch = "2.11";
            }
            {
              toolkit = pkgs.cudaPackages_13_0;
              torch = "2.12";
            }
            {
              toolkit = pkgs.cudaPackages_13_2;
              torch = "2.13";
            }
          ];
        };

        wheelToolkit = lib.mkOption {
          type = lib.types.raw;
          description = ''
            Toolkit the release wheel is built with. setup.py links cudart
            statically, so this sets the minimum driver the wheel runs on. Build
            against the oldest CUDA we support and let driver backward
            compatibility cover newer majors.
          '';
          default = pkgs.cudaPackages_12_8;
        };
      };
    }
  );

  # Wrapped in `config` because this module also declares `options`.
  config.perSystem =
    { lib, system, ... }:
    {
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
