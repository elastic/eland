#!/usr/bin/env bash

set -eo pipefail
export LC_ALL=en_US.UTF-8

docker buildx rm --force eland-multiarch-builder || true
docker buildx create --name eland-multiarch-builder --bootstrap --use

for platform in linux/amd64 linux/arm64; do
  echo "--- Building $platform"
  # Create builder that supports QEMU emulation (needed for linux/arm64)
  docker buildx build --push \
    --file Dockerfile.wolfi \
    --load --platform $platform \
    .
