# The dev shells, one per torch/CUDA variant from `nix/cuda.nix`. The release
# artifacts are built by the `dist` package in `nix/uv2nix.nix`.
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
      inherit (config.cuda) variants variantName;

      mkDevShell =
        v:
        let
          cudaHome = pkgs.callPackage ./cuda-home.nix { cudaPackages = v.toolkit; };
          venv = config.uv2nix.devVenvs.${variantName v};
        in
        # nvcc rejects a host gcc newer than the toolkit supports, and the
        # nixpkgs default moves independently of what NVIDIA allows.
        # `backendStdenv` is the gcc this toolkit was built against.
        (pkgs.mkShell.override { stdenv = v.toolkit.backendStdenv; }) {
          # Writes `.pre-commit-config.yaml` and installs the git hook automatically
          # when entering the shell. See `nix/git-hooks.nix` for details.
          shellHook = ''
            ${config.pre-commit.shellHook}
            unset PYTHONPATH
            # Where the editable install points, so it may not be a store path.
            export REPO_ROOT=$(git rev-parse --show-toplevel)
            # `setup.py` builds a CUDA extension, and an editable install only
            # points at the sources. This writes the shared object next to them.
            build-editable

            # GPU driver libs: NixOS provides /run/opengl-driver/lib; on other
            # distros, symlink just the NVIDIA libs into a temp dir so we don't
            # pull in the host glibc. From https://github.com/NVlabs/cutile-rs.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            else
              _nv_drv_dir=$(mktemp -d /tmp/nix-nvidia-driver.XXXXXX)
              for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib /usr/lib64; do
                if [ -e "$d/libcuda.so.1" ]; then
                  for lib in "$d"/libcuda.so* "$d"/libnvidia-ptxjitcompiler.so* "$d"/libnvidia-gpucomp.so*; do
                    [ -e "$lib" ] && ln -sf "$lib" "$_nv_drv_dir/"
                  done
                  break
                fi
              done
              if [ -n "$(ls -A "$_nv_drv_dir" 2>/dev/null)" ]; then
                export LD_LIBRARY_PATH="$_nv_drv_dir:$LD_LIBRARY_PATH"
              else
                rm -rf "$_nv_drv_dir"
              fi
            fi
          '';
          packages = [
            venv
            # Runs the build backend for its side effects, so a changed .cu or
            # .cpp can be recompiled in place without leaving the shell.
            inputs.pyproject-nix.packages.${pkgs.stdenv.hostPlatform.system}.build-editable
            # The shell hook and the git hooks both shell out to it.
            pkgs.git
            # For `uv lock`. uv2nix owns the environment.
            pkgs.uv
            pkgs.just
            config.llvmPackages.clang-unwrapped
            pkgs.ninja
            cudaHome
          ]
          ++ config.pre-commit.settings.enabledPackages;
          env = {
            CUDA_HOME = "${cudaHome}";
            # The toolkit is always present, a GPU may not be, so the CUDA
            # sources are compiled either way and for stated capabilities.
            FORCE_CUDA = "1";
            TORCH_CUDA_ARCH_LIST = config.cuda.torchArchList v;
            # Tools that resolve imports through the active environment, such
            # as pyrefly and editors, look for one rather than reading `PATH`,
            # and this one is a store path rather than a `.venv` in the tree.
            VIRTUAL_ENV = "${venv}";
            # Workaround: the CCCL bundled with the toolkit annotates the
            # <cuda/std/string_view> deduction guides __host__-only, which
            # clang rejects outright. That breaks clang-tidy on any file
            # reaching <cub/...>, so it is overridden with CCCL 3.4, the
            # first release carrying the fix. See:
            #   - https://github.com/NVIDIA/cccl/issues/7896
            #   - https://github.com/llvm/llvm-project/pull/168711
            CCCL_INCLUDE_DIRS = "${inputs.cccl}/libcudacxx/include:${inputs.cccl}/cub:${inputs.cccl}/thrust";
            # The toolchain clang-tidy analyses against. See
            # `clangToolchainArgs` in `nix/cuda.nix`.
            CLANG_TOOLCHAIN_ARGS = config.cuda.clangToolchainArgs v;
            # The toolkit brings its own host compiler through
            # `backendStdenv`, and nvcc enforces it via <crt/host_config.h>,
            # so torch's redundant bounds check is disabled.
            TORCH_DONT_CHECK_COMPILER_ABI = "1";
            # uv2nix owns the environment, so uv only reads and writes the lock.
            UV_NO_SYNC = "1";
            UV_PYTHON = config.uv2nix.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };
        };

    in
    {
      # One shell per variant, named after it, so `nix develop .#torch212-cu130`
      # pairs that toolkit with the torch it was built against.
      devShells =
        lib.listToAttrs (map (v: lib.nameValuePair (variantName v) (mkDevShell v)) variants)
        // {
          default = mkDevShell config.cuda.defaultVariant;
        };
    };
}
