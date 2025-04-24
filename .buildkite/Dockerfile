ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}

ENV FORCE_COLOR=1

WORKDIR /code/eland
RUN python -m pip install nox

COPY . .
