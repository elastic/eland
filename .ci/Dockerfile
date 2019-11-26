ARG PYTHON_VERSION=3
FROM python:${PYTHON_VERSION}

WORKDIR /code/eland
COPY . .
RUN chmod +x ./run_build.sh
RUN pip install -r requirements-dev.txt
