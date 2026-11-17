# Fixes for building the uv.lock packages with Nix.
# See https://pyproject-nix.github.io/uv2nix/overriding/index.html.
{ pkgs }:
final: prev:
let
  # Have autoPatchelfHook also search site-packages, where wheels install their
  # shared libraries.
  addSitePackagesSearchPath =
    deps: old:
    (old.preFixup or "")
    + pkgs.lib.concatMapStrings (dep: ''
      addAutoPatchelfSearchPath ${dep}/${final.python.sitePackages}
    '') deps;
in
{
  # torch links against the NVIDIA wheels, which each have their own store path.
  torch = prev.torch.overrideAttrs (old: {
    preFixup = addSitePackagesSearchPath [
      final.nvidia-cublas
      final.nvidia-cuda-cupti
      final.nvidia-cuda-nvrtc
      final.nvidia-cuda-runtime
      final.nvidia-cudnn-cu13
      final.nvidia-cufft
      final.nvidia-cufile
      final.nvidia-curand
      final.nvidia-cusolver
      final.nvidia-cusparse
      final.nvidia-cusparselt-cu13
      final.nvidia-nccl-cu13
      final.nvidia-nvshmem-cu13
    ] old;
    # The NVIDIA driver provides libcuda at runtime.
    autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
    appendRunpaths = [ "${pkgs.addDriverRunpath.driverLink}/lib" ];
  });

  # libcudart loads libcuda from the NVIDIA driver with dlopen(3).
  nvidia-cuda-runtime = prev.nvidia-cuda-runtime.overrideAttrs {
    appendRunpaths = [ "${pkgs.addDriverRunpath.driverLink}/lib" ];
  };

  vision3d =
    (prev.vision3d.override { stdenv = pkgs.cudaPackages_13_0.backendStdenv; }).overrideAttrs
      {
        env = {
          CUDA_HOME = pkgs.callPackage ./cuda-home.nix { pythonSet = final; };
          FORCE_CUDA = "1";
          TORCH_CUDA_ARCH_LIST = "7.5;8.0;8.6;9.0;10.0;12.0+PTX";
        };
      };

  torchvision = prev.torchvision.overrideAttrs (old: {
    preFixup = addSitePackagesSearchPath [
      final.nvidia-cuda-runtime
      final.torch
    ] old;
  });

  torchcodec = prev.torchcodec.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [
      pkgs.ffmpeg
      pkgs.libheif
    ];
    preFixup = addSitePackagesSearchPath [
      final.nvidia-cuda-nvrtc
      final.nvidia-cuda-runtime
      final.torch
    ] old;
    # Ignore torchcodec's core libraries for other ffmpeg major versions. It
    # loads the one matching pkgs.ffmpeg.
    autoPatchelfIgnoreMissingDeps = [
      "libav*.so.*"
      "libsw*.so.*"
    ];
  });

  nvidia-cusolver = prev.nvidia-cusolver.overrideAttrs (old: {
    preFixup = addSitePackagesSearchPath [
      final.nvidia-cublas
      final.nvidia-cusparse
      final.nvidia-nvjitlink
    ] old;
  });

  nvidia-cusparse = prev.nvidia-cusparse.overrideAttrs (old: {
    preFixup = addSitePackagesSearchPath [ final.nvidia-nvjitlink ] old;
  });

  # cuDNN loads its sub-libraries from $ORIGIN with dlopen(3).
  nvidia-cudnn-cu13 = prev.nvidia-cudnn-cu13.overrideAttrs {
    autoPatchelfFlags = [ "--preserve-origin" ];
  };

  rerun-sdk = prev.rerun-sdk.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.udev ];
  });

  # The RDMA plugin, which requires these libraries, is optional and not needed.
  nvidia-cufile = prev.nvidia-cufile.overrideAttrs {
    autoPatchelfIgnoreMissingDeps = [
      "libibverbs.so.1"
      "libmlx5.so.1"
      "librdmacm.so.1"
    ];
  };

  # The multi-node plugins, which require these libraries, are optional and not
  # needed.
  nvidia-nvshmem-cu13 = prev.nvidia-nvshmem-cu13.overrideAttrs {
    autoPatchelfIgnoreMissingDeps = [
      "libfabric.so.1"
      "libmlx5.so.1"
      "libmpi.so.40"
      "liboshmem.so.40"
      "libpmix.so.2"
      "libucp.so.0"
      "libucs.so.0"
    ];
  };
}
