{ pkgs }:
final: prev:
let
  virtualenv = final.mkVirtualEnv "vision3d-test-env" {
    vision3d = [
      "dev"
      "viz"
    ];
  };

  # Triton links against libcuda in TRITON_LIBCUDA_PATH, but the linker wrapper
  # skips /run/opengl-driver/lib as an impure path. Link to it from a store path.
  libcuda = pkgs.runCommandLocal "libcuda" { } ''
    mkdir -p $out/lib
    ln -s ${pkgs.addDriverRunpath.driverLink}/lib/libcuda.so.1 $out/lib/libcuda.so.1
  '';

  mkPytest =
    name: marker:
    pkgs.stdenv.mkDerivation {
      name = "${final.vision3d.name}-${name}";
      inherit (final.vision3d) src;
      nativeBuildInputs = [ virtualenv ];
      dontConfigure = true;

      # torch.compile otherwise hashes precompiled headers with the `openssl`
      # CLI. See https://github.com/pytorch/pytorch/issues/204380.
      env.TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS = "0";

      buildPhase = ''
        runHook preBuild
        pytest -p no:cacheprovider -m "${marker}"
        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall
        touch $out
        runHook postInstall
      '';
    };
in
{
  vision3d = prev.vision3d.overrideAttrs (old: {
    passthru = old.passthru // {
      tests = (old.passthru.tests or { }) // {
        pytest-cpu = mkPytest "pytest-cpu" "not cuda";

        # Runs only on hosts that expose the GPU to builds requiring `cuda`,
        # e.g. NixOS with programs.nix-required-mounts.presets.nvidia-gpu.
        pytest-cuda = (mkPytest "pytest-cuda" "cuda").overrideAttrs (old: {
          requiredSystemFeatures = [ "cuda" ];
          # torch.compile and Triton cache compiled CUDA kernels under $HOME.
          nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.writableTmpDirAsHomeHook ];
          env = old.env // {
            TRITON_LIBCUDA_PATH = "${libcuda}/lib";
          };
        });
      };
    };
  });
}
