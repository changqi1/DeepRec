#!/bin/bash
set -x

F_CPP_MIN_VLOG_LEVEL=0 BLAZE_USE_MPS=1 ../../../build/benchmark/benchmark benchmark_conf
