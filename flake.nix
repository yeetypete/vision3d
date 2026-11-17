{
  description = "A 3D extension of torchvision.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

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
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
      ];

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          python = pkgs.python314;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              (import ./nix/overrides.nix { inherit pkgs; })
              (import ./nix/tests.nix { inherit pkgs; })
            ]
          )
      );
    in
    {
      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "vision3d-env" workspace.deps.default;
      });

      checks = forAllSystems (system: {
        inherit (pythonSets.${system}.vision3d.passthru.tests) pytest-cpu pytest-cuda;
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "vision3d-dev-env" workspace.deps.all;
        in
        {
          # Use the host compiler that matches the CUDA version of the wheels.
          default = pkgs.mkShell.override { stdenv = pkgs.cudaPackages_13_0.backendStdenv; } {
            packages = [
              virtualenv
              pkgs.uv
              # Compiles the extension in the source tree.
              pyproject-nix.packages.${system}.build-editable
            ];

            env = {
              # setup.py otherwise uses the nvcc wheel's store path as CUDA_HOME,
              # which lacks the headers from the other CUDA wheels.
              CUDA_HOME = pkgs.callPackage ./nix/cuda-home.nix { inherit pythonSet; };
              UV_NO_SYNC = "1";
              UV_PROJECT_ENVIRONMENT = virtualenv;
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
              UV_PYTHON_PREFERENCE = "only-system";
              VIRTUAL_ENV = virtualenv;
              # Triton locates libcuda with `/sbin/ldconfig`, which NixOS lacks.
              TRITON_LIBCUDA_PATH = "${pkgs.addDriverRunpath.driverLink}/lib";
              # torch.compile otherwise hashes precompiled headers with the
              # `openssl` CLI. See https://github.com/pytorch/pytorch/issues/204380.
              TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS = "0";
            };

            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-tree);
    };
}
