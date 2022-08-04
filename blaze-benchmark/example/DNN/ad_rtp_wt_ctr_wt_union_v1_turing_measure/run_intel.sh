#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 100 150 200 250 400 500 750 800 900 1000; do
  thread_nums=$i \
  && sed -i "/bench_thread_count/cbench_thread_count: ${thread_nums}" benchmark_conf \
  && sed -i "/predictor_num/predictor_num : ${thread_nums}" benchmark_conf
  # F_CPP_MIN_VLOG_LEVEL=0 ../../../build/benchmark/benchmark benchmark_conf &> benchmark_log_thread_num_${i}.log
done;

