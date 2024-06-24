# syntax=docker/dockerfile:1
FROM --platform=$TARGETPLATFORM python:3.10-slim

ARG TARGETPLATFORM

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
      build-essential \
      pkg-config \
      cmake \
      libzip-dev \
      libjpeg-dev \
      git

# Clone your forked repository
RUN git clone https://github.com/rohankshah04/eland.git /eland
WORKDIR /eland

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      python3 -m pip install \
        --no-cache-dir --disable-pip-version-check --extra-index-url https://download.pytorch.org/whl/cpu  \
        torch==2.1.2+cpu .[all]; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      python3 -m pip install \
        --no-cache-dir --disable-pip-version-check \
        .[all]; \
    else \
      echo "Unsupported platform: $TARGETPLATFORM" && exit 1; \
    fi

# Set the entrypoint to eland_import_hub_model
ENTRYPOINT ["eland_import_hub_model"]

# Default command (can be overridden)
CMD ["--help"]
