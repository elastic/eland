ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}

ENV FORCE_COLOR=1

WORKDIR /code/eland
# https://github.com/wntrblm/nox/issues/930
RUN python -m pip install nox==2024.10.09

COPY . .
