#!/bin/bash
set -x

/usr/local/nvidia/bin/nvidia-cuda-mps-control -d
TF_XLA_PTX_CACHE_DIR=./xla_cache TF_CPP_MIN_VLOG_LEVEL=0 BLAZE_USE_MPS=1 LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-11.2/lib64 ../../../build/benchmark/benchmark benchmark_conf
echo quit | nvidia-cuda-mps-control
