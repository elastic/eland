FROM python:3.10-slim-bookworm@sha256:d364435d339ad318ac4c533b9fbe709739f9ba006b0721fd25e8592d0bb857cb

RUN apt-get update && \
    apt-get install -y build-essential pkg-config cmake \
                       libzip-dev libjpeg-dev && \
    apt-get clean

ADD . /eland
WORKDIR /eland

RUN --mount=type=cache,target=/root/.cache python3 -m pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache python3 -m pip install --no-cache-dir --disable-pip-version-check .[all]

CMD ["/bin/sh"]
