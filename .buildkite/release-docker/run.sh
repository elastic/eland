#!/usr/bin/env bash

set -eo pipefail
export LC_ALL=en_US.UTF-8

echo "Publishing Eland $RELEASE_VERSION Docker image to $ENVIRONMENT"

set +x
# login to docker registry
docker_registry=$(vault read -field registry "secret/ci/elastic-eland/container-library/eland-$ENVIRONMENT")
docker_username=$(vault read -field username "secret/ci/elastic-eland/container-library/eland-$ENVIRONMENT")
docker_password=$(vault read -field password "secret/ci/elastic-eland/container-library/eland-$ENVIRONMENT")

echo "$docker_password" | docker login "$docker_registry" --username "$docker_username" --password-stdin
unset docker_username docker_password
set -x

tmp_dir=$(mktemp --directory)
pushd "$tmp_dir"
git clone https://github.com/elastic/eland
pushd eland
git checkout "v${RELEASE_VERSION}"
git --no-pager show

# Create builder that supports QEMU emulation (needed for linux/arm64)
docker buildx rm --force eland-multiarch-builder || true
docker buildx create --name eland-multiarch-builder --bootstrap --use
docker buildx build --push \
  --file Dockerfile.wolfi \
  --tag "$docker_registry/eland/eland:$RELEASE_VERSION" \
  --tag "$docker_registry/eland/eland:latest" \
  --platform linux/amd64,linux/arm64 \
  "$PWD"

popd
popd
rm -rf "$tmp_dir"
