#!/usr/bin/env bash
sudo apt-get update
sudo apt-get install -y pandoc python3 python3-pip
python3 -m pip install nox
/opt/buildkite-agent/.local/bin/nox -s docs
