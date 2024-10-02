#!/usr/bin/env bash

set -eo pipefail
export LC_ALL=en_US.UTF-8

# Create builder that supports QEMU emulation (needed for linux/arm64)
docker buildx rm --force eland-multiarch-builder || true
docker buildx create --name eland-multiarch-builder --bootstrap --use
docker buildx build --file Dockerfile.wolfi --load --platform $PLATFORM .
