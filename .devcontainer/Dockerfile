FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

USER ubuntu

# Pre-create the uv cache and Python interpreter dirs so docker does not create
# them with root permissions when mounted.
RUN mkdir -p /home/ubuntu/.cache/uv /home/ubuntu/.local/share/uv/python
