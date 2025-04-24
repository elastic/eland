# syntax=docker/dockerfile:1
FROM python:3.10-slim

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
      build-essential \
      pkg-config \
      cmake \
      libzip-dev \
      libjpeg-dev

ADD . /eland
WORKDIR /eland

ARG TARGETPLATFORM
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      python3 -m pip install \
        --no-cache-dir --disable-pip-version-check --extra-index-url https://download.pytorch.org/whl/cpu  \
        torch==2.5.1+cpu .[all]; \
    else \
      python3 -m pip install \
        --no-cache-dir --disable-pip-version-check \
        .[all]; \
    fi

CMD ["/bin/sh"]
