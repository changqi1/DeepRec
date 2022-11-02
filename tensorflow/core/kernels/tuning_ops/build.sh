#!/usr/bin/env bash
#echo "source /home/media/miniconda3/bin/activate gemm_TF1.15"
#source /home/media/miniconda3/bin/activate gemm_TF1.15
export LD_LIBRARY_PATH=/home/media/djx/gcc/lib:/home/media/djx/gcc/lib64:/home/media/djx/gcc/libexec:$LD_LIBRARY_PATH
export PATH=/home/media/djx/gcc/bin/bin:$PATH
#g++ -v

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo ${TF_CFLAGS}
echo ${TF_LFLAGS}

echo g++ -std=c++14 -shared -mavx512f -mfma packed_matmul_op.cc -o packed_matmul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -g -std=c++14 -shared -mavx512f -mfma host_packed_matmul_op.cc ./tuning_matmul/tunable_matmul.cpp -o host_packed_matmul.so -I ./tuning_matmul -I ./tuning_matmul/host/include -I ./tuning_matmul/host/include/DataTypes -I ./tuning_matmul/host/Matmul -L ./tuning_matmul/host/build -lAutoMatMul -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
#g++ -O3 -std=c++14 -shared -mavx512f -mfma packed_matmul_op.cc -o packed_matmul.so -I sgemm_kernel.h -I ./tuning_matmul -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}

#echo g++ -std=c++14 -shared -mavx512f -mfma multi_packed_matmul.cc -o multi_matmul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
#g++ -std=c++14 -shared -mavx512f -mfma multi_packed_matmul.cc -o multi_matmul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
