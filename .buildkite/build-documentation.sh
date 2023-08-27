#!/usr/bin/env bash
sudo apt-get update
sudo apt-get install -y pandoc python3 python3-pip
python3 -m pip install nox
/opt/buildkite-agent/.local/bin/nox -s docs

# I couldn't make this work, for some reason pandoc is not found in the docker container repository:
# docker build --file .buildkite/Dockerfile --tag elastic/eland --build-arg PYTHON_VERSION=${PYTHON_VERSION} .
# docker run \
#        --name doc_build \
#        --rm \
#        elastic/eland \
#        apt-get update && \
#        sudo apt-get install --yes pandoc && \
#        nox -s docs
