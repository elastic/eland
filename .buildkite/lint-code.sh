#!/usr/bin/env bash
docker build --file .buildkite/Dockerfile --tag elastic/eland --build-arg PYTHON_VERSION=${PYTHON_VERSION} .
docker run \
       --name linter \
       --rm \
       elastic/eland \
       nox -s lint
