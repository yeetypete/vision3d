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
    # `manylinux` in `nix/devshells.nix`.
    kernels = {
      url = "github:huggingface/kernels";
      flake = false;
    };

    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Builds the test environments from `uv.lock`. See `nix/uv2nix.nix`.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        ./nix/cuda.nix
        ./nix/devshells.nix
        ./nix/git-hooks.nix
        ./nix/llvm.nix
        ./nix/manylinux.nix
        ./nix/uv2nix.nix
      ];

      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
    };
}
