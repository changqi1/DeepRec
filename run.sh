#!/bin/bash
default_opts=" \
             --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 \
             --copt=-O2 \
             --copt=-Wformat \
             --copt=-Wformat-security \
             --copt=-fstack-protector \
             --copt=-fPIC \
             --copt=-fpic \
             --linkopt=-znoexecstack \
             --linkopt=-zrelro \
             --linkopt=-znow \
             --linkopt=-fstack-protector"

mkl_opts="--config=mkl_threadpool \
           --define build_with_mkl_dnn_v1_only=true \
           --copt=-DENABLE_INTEL_MKL_BFLOAT16 \
           --copt=-march=skylake-avx512"

test_opts="--nocache_test_results \
           --test_output=all \
           --verbose_failures \
           --test_verbose_timeout_warnings \
           --flaky_test_attempts 1 \
           --test_timeout 99999999 \
           --test_size_filters=small,medium,large,enormous \
           -c opt \
           --keep_going"

test_options="--test_arg=--benchmarks=all"

# test with amx512-bf16
bazel test --run_under="numactl -C 56-71" --test_env ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX ${default_opts} ${mkl_opts} ${test_opts} ${test_options} -- //tensorflow/core/kernels:mkl_matmul_op_test |tee  mkl_matmul_with_amx.log

# test with avx512-bf16
bazel test --run_under="numactl -C 56-71" --test_env ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 ${default_opts} ${mkl_opts} ${test_opts} ${test_options} -- //tensorflow/core/kernels:mkl_matmul_op_test |tee  mkl_matmul_with_avx.log
