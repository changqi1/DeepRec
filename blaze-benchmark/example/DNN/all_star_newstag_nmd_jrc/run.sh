#!/bin/bash
set -x

ln -sf ../../runmeta/all_star_newstag_nmd_jrc/frozen_graph.pb user/frozen_graph.pb
/usr/local/nvidia/bin/nvidia-cuda-mps-control -d
TF_XLA_PTX_CACHE_DIR=./xla_cache TF_CPP_MIN_VLOG_LEVEL=0 BLAZE_USE_MPS=1 LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-11.2/lib64 ../../../build/benchmark/benchmark benchmark_conf
echo quit | nvidia-cuda-mps-control
