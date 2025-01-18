#!/usr/bin/env bash

docker build --file .buildkite/Dockerfile --tag elastic/eland --build-arg PYTHON_VERSION=${PYTHON_VERSION} .
docker run \
  --name doc_build \
  --rm \
  elastic/eland \
  bash -c "apt-get update && apt-get install --yes pandoc && nox -s docs"
