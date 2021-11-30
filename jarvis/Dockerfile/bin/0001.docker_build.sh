#!/bin/bash

#image_version=:v1.0

proxy="--build-arg https_proxy=$http_proxy --build-arg http_proxy=$http_proxy \
       --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTP_PROXY \
       --build-arg NO_PROXY=$NO_PROXY --build-arg no_proxy=$NO_PROXY"

echo "PROXY =" $proxy

sudo docker build ${proxy} -t deeprec-cpu-dev${image_version} -f CPU-dev.Dockerfile ..
#sudo docker build ${proxy} -t deeprec-cpu-test${image_version} -f CPU-test.Dockerfile ..

#sudo docker build ${proxy} -t deeprec-gpu-dev${image_version} -f GPU-dev.Dockerfile ..
#sudo docker build ${proxy} -t deeprec-gpu-test${image_version} -f GPU-test.Dockerfile ..
