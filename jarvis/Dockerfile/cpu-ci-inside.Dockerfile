FROM ubuntu:18.04

ARG BAZEL_VERSION=0.26.1

ARG CI_BUILD_GID
ARG CI_BUILD_GROUP
ARG CI_BUILD_UID
ARG CI_BUILD_USER
ARG CI_BUILD_HOME=/home/${CI_BUILD_USER}

ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV no_proxy ${no_proxy}

ENV HTTP_PROXY ${HTTP_PROXY}
ENV HTTPS_PROXY ${HTTPS_PROXY}
ENV NO_PROXY ${NO_PROXY}

RUN apt-get update
RUN apt-get install -y sudo

############################# Set same user in container #############################
RUN getent group "${CI_BUILD_GID}" || addgroup --force-badname --gid ${CI_BUILD_GID} ${CI_BUILD_GROUP}
RUN getent passwd "${CI_BUILD_UID}" || adduser --force-badname --gid ${CI_BUILD_GID} --uid ${CI_BUILD_UID} \
      --disabled-password --home ${CI_BUILD_HOME} --quiet ${CI_BUILD_USER}
RUN usermod -a -G sudo ${CI_BUILD_USER}
RUN echo "${CI_BUILD_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-nopasswd-sudo

USER ${CI_BUILD_UID}:${CI_BUILD_GID}

RUN whoami

WORKDIR ${CI_BUILD_HOME}
######################################################################################

ENV PATH ${CI_BUILD_HOME}/bin:$PATH

RUN sudo -E apt-get install -y --no-install-recommends \
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
        openjdk-8-jre-headless

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN sudo -E apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN sudo ln -s $(which python3) /usr/local/bin/python

RUN sudo -E apt-get install -y \
    build-essential \
    curl \
    openjdk-8-jdk \
    python3-dev \
    virtualenv \
    swig \
    vim \
    numactl

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
    enum34 \
    astor

# Install bazel
RUN mkdir bazel && \
    wget -O bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE"

RUN chmod +x bazel/installer.sh && \
    sudo bazel/installer.sh && \
    rm -f bazel/installer.sh
