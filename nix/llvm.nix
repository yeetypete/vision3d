# LLVM shared by the dev shell, the clang-format hook and `just tidy`.
{ flake-parts-lib, ... }:
{
  options.perSystem = flake-parts-lib.mkPerSystemOption (
    { lib, pkgs, ... }:
    {
      options.llvmPackages = lib.mkOption {
        type = lib.types.raw;
        description = ''
          nixpkgs `llvmPackages` set behind clang-tidy, clang-format and
          clangd.
        '';
        default = pkgs.llvmPackages_22;
      };
    }
  );
}
