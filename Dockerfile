FROM python:3-10-slim-bookworm@sha256:cc91315c3561d0b87d0525cb814d430cfbc70f10ca54577def184da80e87c1db

RUN apt-get update && \
    apt-get install -y build-essential pkg-config cmake \
                       libzip-dev libjpeg-dev && \
    apt-get clean

ADD . /eland
WORKDIR /eland

RUN python3 -m pip install --no-cache-dir --disable-pip-version-check .[all]

CMD ["/bin/sh"]
