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

RUN --mount=type=cache,target=/root/.cache python3 -m pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache python3 -m pip install --no-cache-dir --disable-pip-version-check .[all]

CMD ["/bin/sh"]
