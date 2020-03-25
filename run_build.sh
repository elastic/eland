#!/usr/bin/env bash

python -m pip install nox
nox -s lint test
