# The manylinux_2_28 toolchain the release artifacts are built with, shared by
# the wheel package in `nix/uv2nix.nix` and the shell in `nix/devshells.nix`.
{ inputs, flake-parts-lib, ... }:
{
  options.perSystem = flake-parts-lib.mkPerSystemOption (
    { lib, pkgs, ... }:
    {
      options.manylinux = {
        packages = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = ''
            The manylinux_2_28 environment itself, as RHEL 8 packages unpacked
            from RPMs, so that what is built against it meets the glibc and
            libstdc++ the tag promises.
          '';
        };

        stdenv = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = ''
            The stdenv out of that environment the extension is compiled with.
            gcc 13 rather than 14, because torch's host-compiler table caps CUDA
            12.8 at gcc < 14 before torch 2.12, and the wheel is built against
            the oldest torch we support.
          '';
        };
      };
    }
  );

  config.perSystem =
    {
      config,
      lib,
      pkgs,
      ...
    }:
    {
      manylinux.packages =
        let
          root = "${inputs.kernels}/nix-builder/pkgs/manylinux";
          inherit (pkgs.stdenv.hostPlatform.uname) processor;
        in
        pkgs.callPackage root { } {
          packageMetadata = lib.importJSON "${root}/manylinux-2.28-${processor}-metadata.json";
        };

      manylinux.stdenv = config.manylinux.packages.gcc13Stdenv;
    };
}
