# Formatters and lightweight linters, run as git hooks and by `nix fmt`.
{ inputs, ... }:
{
  imports = [ inputs.git-hooks.flakeModule ];

  perSystem =
    {
      config,
      lib,
      pkgs,
      ...
    }:
    {
      # Source of truth for `.pre-commit-config.yaml`, which git-hooks
      # generates on shell entry and which is gitignored.
      pre-commit = {
        settings = {
          package = pkgs.prek;
          hooks = {
            actionlint.enable = true;
            # `uv.lock` grew past the 500 kB default.
            check-added-large-files = {
              enable = true;
              args = [ "--maxkb=1024" ];
            };
            check-case-conflicts.enable = true;
            check-merge-conflicts.enable = true;
            check-symlinks.enable = true;
            check-toml.enable = true;
            check-yaml.enable = true;
            detect-private-keys.enable = true;
            end-of-file-fixer.enable = true;
            mixed-line-endings.enable = true;
            python-debug-statements.enable = true;
            trim-trailing-whitespace.enable = true;
            nixfmt.enable = true;

            check-json = {
              enable = true;
              excludes = [ "^\\.zed/" ];
            };
            pretty-format-json = {
              enable = true;
              settings.autofix = true;
              excludes = [ "^\\.zed/" ];
            };
            markdownlint = {
              enable = true;
              excludes = [ "^CLAUDE\\.md$" ];
            };

            # `just --fmt` takes a single `--justfile`, but a hook is handed
            # every matching path at once.
            just-fmt = {
              enable = true;
              name = "just-fmt";
              entry = lib.getExe (
                pkgs.writeShellApplication {
                  name = "just-fmt";
                  runtimeInputs = [ pkgs.just ];
                  text = ''
                    for file in "$@"; do
                      just --fmt --justfile "$file"
                    done
                  '';
                }
              );
              language = "system";
              files = "(^|/)justfile$";
            };

            clang-format = {
              enable = true;
              package = config.llvmPackages.clang-unwrapped;
              types_or = lib.mkForce [
                "c"
                "c++"
                "cuda"
              ];
            };

            # `ruff-check` and `ruff-fmt` rather than the built-in `ruff` and
            # `ruff-format` attributes, because those run nixpkgs' ruff. We want
            # the one `uv.lock` pins, which uv2nix puts in the environment, so
            # the hooks agree with what CI and the dev shell run.
            ruff-check = {
              enable = true;
              name = "ruff";
              entry = "${config.uv2nix.defaultVenv}/bin/ruff check --force-exclude";
              language = "system";
              types_or = [
                "python"
                "pyi"
              ];
              require_serial = true;
            };
            ruff-fmt = {
              enable = true;
              name = "ruff-format";
              entry = "${config.uv2nix.defaultVenv}/bin/ruff format --force-exclude";
              language = "system";
              types_or = [
                "python"
                "pyi"
              ];
              require_serial = true;
            };
            uv-lock = {
              enable = true;
              name = "uv-lock";
              # By store path, and told which interpreter to evaluate markers
              # with, because the check derivation this also runs in has
              # neither uv on `PATH` nor anything to discover a python from.
              # `--offline` because it has no network either. The lock
              # carries what verifying it needs.
              entry = lib.concatStringsSep " " [
                "env"
                "UV_PYTHON=${config.uv2nix.python.interpreter}"
                "UV_PYTHON_DOWNLOADS=never"
                (lib.getExe pkgs.uv)
                "lock --check --offline"
              ];
              language = "system";
              files = "^(pyproject\\.toml|uv\\.lock|uv\\.toml)$";
              pass_filenames = false;
            };
          };
        };
      };

      # `nix fmt` runs every hook over the whole tree and lets the fixers
      # write to it, the same thing the git hook does on a commit.
      formatter = pkgs.writeShellScriptBin "pre-commit-run" ''
        # `nix fmt` forwards the paths it was given, and means the whole tree
        # when given none.
        if [ "$#" -gt 0 ]; then
          exec ${lib.getExe config.pre-commit.settings.package} run --files "$@" \
            --config ${config.pre-commit.settings.configFile}
        fi
        exec ${lib.getExe config.pre-commit.settings.package} run --all-files \
          --config ${config.pre-commit.settings.configFile}
      '';
    };
}
