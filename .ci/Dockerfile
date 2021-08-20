ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}

WORKDIR /code/eland
RUN python -m pip install nox

COPY . .
