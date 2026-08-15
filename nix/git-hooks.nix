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
        # Some hooks shell out to `uv run`, which needs a synced `.venv`
        # and a network. `nix flake check` has neither, so the hooks run
        # from `nix fmt` and the git hook instead.
        check.enable = false;

        settings = {
          package = pkgs.prek;
          hooks = {
            actionlint.enable = true;
            check-added-large-files.enable = true;
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
            # `ruff-format` attributes, because those run nixpkgs' ruff and add
            # it to the dev shell. We want to track ruff in uv.lock.
            ruff-check = {
              enable = true;
              name = "ruff";
              entry = "uv run --no-sync ruff check --force-exclude";
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
              entry = "uv run --no-sync ruff format --force-exclude";
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
              entry = "uv lock --check";
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
        ${lib.getExe config.pre-commit.settings.package} run --all-files \
          --config ${config.pre-commit.settings.configFile}
      '';
    };
}
