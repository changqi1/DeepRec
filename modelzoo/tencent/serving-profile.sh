#!/bin/bash

SCRIPT_DIR=$(dirname $(readlink -f $0))

TENSORFLOW_SERVING_VERSION=2.12.1
TENSORFLOW_VERSION_FOR_PROFILER=2.12.0

DOCKER_TF_SERVING_HOSTNAME=serving
TF_SERVING_MODEL_NAME=tencent

LOCAL_DIR_FOR_BIND_MOUNTS=tf_logs_dir
TMP_TF_LOGDIR=${SCRIPT_DIR}/${LOCAL_DIR_FOR_BIND_MOUNTS}

# Create directory to be mounted to profiler and tf serving container
MKDIR_CMD="mkdir -p ${TMP_TF_LOGDIR}"
echo $MKDIR_CMD
eval $MKDIR_CMD

# Create dockerfile for profiler:
DOCKER_PROFILER_TAG=tensorboard_profiler:latest

echo """FROM tensorflow/tensorflow:${TENSORFLOW_VERSION_FOR_PROFILER}

RUN http_proxy=http://child-prc.intel.com:913 https_proxy=http://child-prc.intel.com:913 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U tensorboard-plugin-profile

# ENTRYPOINT ["/usr/local/bin/tensorboard", "--logdir", "/tmp/tensorboard", "--bind_all"]
""">Dockerfile_tfprofile

# Build the profiler image
sudo docker build --build-arg HTTP_PROXY --build-arg HTTPS_PROXY -t ${DOCKER_PROFILER_TAG} -f Dockerfile_tfprofile .

# Create docker compose file
echo """version: '3.3'
services:
  ${DOCKER_TF_SERVING_HOSTNAME}:
    image: tensorflow/serving:${TENSORFLOW_SERVING_VERSION}
    ports:
      - '8510:8500' # Whatever you need for client
      - '8511:8501' # Whatever you need for client
    environment:
      - MODEL_NAME=${TF_SERVING_MODEL_NAME}
    hostname: '${DOCKER_TF_SERVING_HOSTNAME}'
    volumes:
      - '${SCRIPT_DIR}:/models/${TF_SERVING_MODEL_NAME}/1'
      - '${TMP_TF_LOGDIR}:/tmp/tensorboard'
  profiler:
    image: ${DOCKER_PROFILER_TAG}
    ports:
      - '6006:6006'
    volumes:
      - '${TMP_TF_LOGDIR}:/tmp/tensorboard'
""">docker-compose.yaml

docker-compose up

# Profile url = ${DOCKER_TF_SERVING_HOSTNAME}:8500
# Set profile duration
# Make request within duration