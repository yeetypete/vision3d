# The test environments as derivations, built from `uv.lock` by uv2nix.
#
# The dev shells take the editable ones (`devVenvs`), and `nix build
# .#venv-torch213-cu130` gives the installed counterpart with the extension
# already compiled. `nix flake check` runs the CPU suite in every variant, each
# check's output being that variant's HTML coverage report.
#
# The variant groups conflict, and uv2nix resolves one conflict selection per
# overlay, so every variant gets its own package set.
{ inputs, flake-parts-lib, ... }:
{
  options.perSystem = flake-parts-lib.mkPerSystemOption (
    { lib, ... }:
    {
      options.uv2nix = {
        python = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = "Interpreter the generated environments are built for.";
        };

        defaultVenv = lib.mkOption {
          type = lib.types.raw;
          readOnly = true;
          description = ''
            Editable environment for `cuda.defaultVariant`, for tooling that
            wants a locked interpreter without picking a variant, such as the
            git hooks.
          '';
        };

        devVenvs = lib.mkOption {
          type = lib.types.attrsOf lib.types.raw;
          readOnly = true;
          description = ''
            Editable virtual environment per variant, keyed by variant name.
            The dev shells put these on `PATH`; see `nix/devshells.nix`.
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
    let
      inherit (config.cuda) variants variantName;
      inherit (inputs) pyproject-nix pyproject-build-systems uv2nix;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };

      # The oldest interpreter `requires-python` admits, which is the one the
      # release wheel targets.
      python = lib.head (
        pyproject-nix.lib.util.filterPythonInterpreters {
          inherit (workspace) requires-python;
          inherit (pkgs) pythonInterpreters;
        }
      );

      # Groups and extras to resolve and install. `dev` and `docs` are groups,
      # `viz` is the extra `--all-extras` in the justfile pulls in.
      depsOf = v: {
        vision3d = [
          (variantName v)
          "dev"
          "docs"
          "viz"
        ];
      };

      # Filtered per package rather than at the workspace root, which uv2nix
      # reads at evaluation time. Docs and the gallery do not affect the build,
      # so they do not belong in its hash.
      buildFiles = lib.fileset.unions [
        ../pyproject.toml
        ../setup.py
        ../MANIFEST.in
        ../README.md
        ../LICENSE
        ../src
        ../test
      ];

      source = lib.fileset.toSource {
        root = ../.;
        fileset = buildFiles;
      };

      # pyrefly's `search-path` reaches the gallery and the sphinx extensions,
      # and clang-tidy runs the same script `just tidy` does, so a check and a
      # dev shell analyse with the same flags.
      lintSource = lib.fileset.toSource {
        root = ../.;
        fileset = lib.fileset.unions [
          buildFiles
          ../.clang-tidy
          ../scripts
          ../gallery
          ../docs/source/_ext
        ];
      };

      # `autoPatchelfHook` resolves each wheel on its own, but the CUDA stack is
      # split across wheels that only meet inside the venv: torch's libraries
      # live in `nvidia-*`, torchvision's in torch, and nvshmem carries
      # transports (MPI, UCX, libfabric) that are optional at runtime. All of
      # them are dlopened after `import torch` has pulled its dependencies into
      # the process, so the loader resolves them by SONAME.
      splitCudaStack = [
        "torch"
        "torchvision"
        "triton"
      ];

      cudaWheelFixups =
        _final: prev:
        lib.mapAttrs (
          _: package:
          package.overrideAttrs (_old: {
            autoPatchelfIgnoreMissingDeps = true;
          })
        ) (lib.filterAttrs (name: _: lib.elem name splitCudaStack || lib.hasPrefix "nvidia-" name) prev);

      pyprojectOverrides =
        {
          variant,
          stdenv,
        }:
        let
          v = variant;
          cudaHome = pkgs.callPackage ./cuda-home.nix {
            cudaPackages = v.toolkit;
            # nvcc bakes a `compiler-bindir` into `nvcc.profile`, which is the
            # toolkit's own gcc unless the caller brings another. A release
            # build brings the manylinux one, so that host and device passes
            # agree on a libstdc++.
            hostCC = if stdenv == v.toolkit.backendStdenv then null else stdenv.cc;
          };
        in
        final: prev: {
          # torchvision's extension links libtorch directly, so point
          # `autoPatchelfHook` at the package carrying it.
          torchvision = prev.torchvision.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ final.torch ];
          });

          # The rerun wheel bundles a viewer binary linking libudev, which no
          # wheel provides.
          rerun-sdk = prev.rerun-sdk.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.systemdLibs ];
          });

          vision3d =
            let
              # One environment for every check: the tests import vision3d,
              # and pyrefly resolves the sphinx extensions against the docs
              # group.
              checkEnv = final.mkVirtualEnv "vision3d-${variantName v}-check-env" (depsOf v);
            in
            prev.vision3d.overrideAttrs (old: {
              src = source;

              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                cudaHome
                pkgs.ninja
              ];

              # The same environment the dev shells export, since `setup.py`
              # reads it the same way here.
              env = (old.env or { }) // {
                CUDA_HOME = "${cudaHome}";
                FORCE_CUDA = "1";
                TORCH_CUDA_ARCH_LIST = config.cuda.torchArchList v;
                TORCH_DONT_CHECK_COMPILER_ABI = "1";
              };

              # Runtime and test dependencies are not available during a package
              # build, so the suite is a derivation of its own, discovered through
              # `passthru.tests` the way nixpkgs does it.
              passthru = old.passthru // {
                tests = (old.passthru.tests or { }) // {
                  pytest = pkgs.stdenv.mkDerivation {
                    name = "${final.vision3d.name}-pytest";
                    inherit (final.vision3d) src;

                    nativeBuildInputs = [
                      checkEnv
                      # inductor shells out to it when hashing what it compiles
                      # for the `torch.compile` tests.
                      pkgs.openssl
                    ];

                    dontConfigure = true;

                    # GPU tests are deselected, since the sandbox has no device.
                    buildPhase = ''
                      runHook preBuild
                      export HOME=$TMPDIR
                      pytest -m cpu --cov-report=html
                      runHook postBuild
                    '';

                    installPhase = ''
                      runHook preInstall
                      mv htmlcov $out
                      runHook postInstall
                    '';
                  };

                  pyrefly = pkgs.stdenvNoCC.mkDerivation {
                    name = "${final.vision3d.name}-pyrefly";
                    src = lintSource;

                    nativeBuildInputs = [ checkEnv ];

                    dontConfigure = true;

                    # pyrefly resolves imports through the active environment,
                    # the same way the dev shells point it at one.
                    env.VIRTUAL_ENV = "${checkEnv}";

                    buildPhase = ''
                      runHook preBuild
                      export HOME=$TMPDIR
                      pyrefly check
                      runHook postBuild
                    '';

                    installPhase = "touch $out";
                  };

                  # Reads the compile database `setup.py` emits, so this needs
                  # what a build needs, plus clang. It runs on the toolkit's own
                  # stdenv, so the database names the compiler the dev shells
                  # compile with.
                  clang-tidy = v.toolkit.backendStdenv.mkDerivation {
                    name = "${final.vision3d.name}-clang-tidy";
                    src = lintSource;

                    nativeBuildInputs = [
                      pkgs.ninja
                      cudaHome
                      config.llvmPackages.clang-unwrapped
                      checkEnv
                    ];

                    dontConfigure = true;

                    env = {
                      CUDA_HOME = "${cudaHome}";
                      FORCE_CUDA = "1";
                      TORCH_CUDA_ARCH_LIST = config.cuda.torchArchList v;
                      TORCH_DONT_CHECK_COMPILER_ABI = "1";
                      # Both as exported by the dev shells.
                      CCCL_INCLUDE_DIRS = "${inputs.cccl}/libcudacxx/include:${inputs.cccl}/cub:${inputs.cccl}/thrust";
                      CLANG_TOOLCHAIN_ARGS = config.cuda.clangToolchainArgs v;
                    };

                    buildPhase = ''
                      runHook preBuild
                      export HOME=$TMPDIR
                      bash scripts/tidy.sh
                      runHook postBuild
                    '';

                    installPhase = "touch $out";
                  };
                };
              };
            });
        };

      # nvcc rejects a host gcc newer than the toolkit supports, and the nixpkgs
      # default moves independently of what NVIDIA allows, so the default is the
      # gcc that toolkit was built against.
      pythonSetFor =
        {
          variant,
          stdenv ? variant.toolkit.backendStdenv,
        }:
        let
          v = variant;
        in
        (pkgs.callPackage pyproject-nix.build.packages { inherit python stdenv; }).overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            (workspace.mkPyprojectOverlay {
              sourcePreference = "wheel";
              # uv2nix resolves the whole workspace without this, which
              # `tool.uv.conflicts` forbids.
              dependencies = depsOf v;
            })
            cudaWheelFixups
            (pyprojectOverrides { inherit variant stdenv; })
          ]
        );

      # Virtual environments have no main program and carry no metadata of
      # their own, so the package's tests are surfaced on them.
      mkVenv =
        v:
        let
          pythonSet = pythonSetFor { variant = v; };
        in
        (pythonSet.mkVirtualEnv "vision3d-${variantName v}-env" (depsOf v)).overrideAttrs (old: {
          passthru = lib.recursiveUpdate (old.passthru or { }) {
            inherit (pythonSet.vision3d.passthru) tests;
          };
        });

      # The dev shell counterpart: vision3d is a pointer to the working tree
      # rather than an installed copy, so edits take effect without a rebuild.
      # `$REPO_ROOT` is exported by the shell hook, because an editable root
      # may not be a store path.
      mkDevVenv =
        v:
        ((pythonSetFor { variant = v; }).overrideScope (
          workspace.mkEditablePyprojectOverlay {
            root = "$REPO_ROOT";
          }
        )).mkVirtualEnv
          "vision3d-${variantName v}-dev-env"
          (depsOf v);

      forVariants =
        prefix: f: lib.listToAttrs (map (v: lib.nameValuePair "${prefix}${variantName v}" (f v)) variants);

      # The release build: the oldest pair we support, compiled against the
      # manylinux_2_28 toolchain rather than the toolkit's own gcc, so the
      # artifact meets the glibc and libstdc++ its tag promises.
      wheelSet = pythonSetFor {
        variant = lib.head variants;
        inherit (config.manylinux) stdenv;
      };

      # `pyprojectDistHook` installs what the backend produced into `$out`
      # instead of unpacking it, so the output is the sdist or the wheel.
      mkDist =
        buildType:
        (wheelSet.vision3d.override { pyprojectHook = wheelSet.pyprojectDistHook; }).overrideAttrs (old: {
          env = (old.env or { }) // {
            uvBuildType = buildType;
            # The wrapped linker otherwise writes the store into the
            # extension: an rpath per store directory it links against, plus
            # the output's own lib directory. Neither may travel in a wheel,
            # and `scripts/check-wheel.sh` fails the build if one does.
            NIX_NO_SELF_RPATH = "1";
            "NIX_DONT_SET_RPATH_${config.manylinux.stdenv.cc.suffixSalt}" = "1";
            # The hook's own guard is a byte-level search for store paths
            # anywhere in the archive, which a compiled extension cannot pass:
            # the toolchain's loader is named in `PT_INTERP`, and CUB records
            # its own header path in an error message. Neither is read at load
            # time. `scripts/check-wheel.sh` checks what is.
            dontUsePyprojectInstallDistCheck = true;
          };
        });
    in
    {
      uv2nix = {
        inherit python;
        devVenvs = forVariants "" mkDevVenv;
        defaultVenv = mkDevVenv config.cuda.defaultVariant;
      };

      packages = forVariants "venv-" mkVenv // {
        # What a release uploads: the sdist as built, and the wheel retagged
        # manylinux by auditwheel, which refuses the tag if the binary does not
        # satisfy it.
        dist =
          pkgs.runCommand "vision3d-dist"
            {
              nativeBuildInputs = [
                pkgs.auditwheel
                # auditwheel shells out to both: `--strip` to drop the debug
                # info naming this machine, patchelf to rewrite what it tags.
                pkgs.binutils
                pkgs.patchelf
                # `scripts/check-wheel.sh` reads the wheel back.
                pkgs.unzip
              ];
            }
            ''
              mkdir -p $out
              cp ${mkDist "sdist"}/*.tar.gz $out/
              auditwheel repair --strip \
                --plat manylinux_2_28_${pkgs.stdenv.hostPlatform.uname.processor} --only-plat \
                --exclude '*' --wheel-dir $out ${mkDist "wheel"}/*.whl
                  bash ${../scripts/check-wheel.sh} $out/*.whl
            '';
      };

      checks =
        forVariants "tests-" (v: (pythonSetFor { variant = v; }).vision3d.passthru.tests.pytest)
        // {
          # Neither linter's result varies with the torch version, so they run
          # once, against the pair a plain `nix develop` gives.
          inherit ((pythonSetFor { variant = config.cuda.defaultVariant; }).vision3d.passthru.tests)
            pyrefly
            clang-tidy
            ;
        };
    };
}
