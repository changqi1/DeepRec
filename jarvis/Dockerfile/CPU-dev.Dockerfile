ARG UBUNTU_VERSION=18.04
ARG BAZEL_VERSION=0.26.1

FROM ubuntu:${UBUNTU_VERSION} as base
ARG BAZEL_VERSION

ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV no_proxy ${no_proxy}

ENV HTTP_PROXY ${HTTP_PROXY}
ENV HTTPS_PROXY ${HTTPS_PROXY}
ENV NO_PROXY ${NO_PROXY}

ENV PATH /root/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        sudo \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        openjdk-8-jre-headless \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    openjdk-8-jdk \
    python3-dev \
    virtualenv \
    swig

RUN python3 -m pip --no-cache-dir install \
    Pillow \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    'numpy<1.19.0' \
    scipy \
    sklearn \
    pandas \
    future \
    portpicker \
    enum34

# Install bazel
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh

WORKDIR /root

COPY bin bin
COPY DeepRec DeepRec
WORKDIR DeepRec
