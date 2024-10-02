#!/usr/bin/env bash

set -eo pipefail
export LC_ALL=en_US.UTF-8

echo "--- Building the Wolfi image"
# Building the linux/arm64 image takes about one hour on Buildkite, which is too slow
docker build --file Dockerfile.wolfi .

echo "--- Building the public image"
docker build .
