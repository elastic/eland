FROM debian:11.1

RUN apt-get update && \
    apt-get install -y build-essential pkg-config cmake \
                       python3-dev python3-pip python3-venv \
                       libzip-dev libjpeg-dev && \
    apt-get clean

ADD . /eland
WORKDIR /eland

RUN python3 -m pip install --no-cache-dir --disable-pip-version-check .[all]

CMD ["/bin/sh"]
