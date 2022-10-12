//
// Created by qiaoxj on 2020/9/7.
//

#ifndef TENSORFLOW_CO_ACTION_OP_H
#define TENSORFLOW_CO_ACTION_OP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
struct LaunchCoAction {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b, Tensor* out,
                  int64 batch_a, int64 batch_b, int64 paralle_num,
                  int64 pow_num);
};

template <typename Device, typename Scalar>
struct LaunchOptCoAction {
  Status operator()(OpKernelContext* context, const int64 m, const int64 n, const int64 k,
                  const Tensor& in_a, const Tensor& in_b, Tensor* out,
                  const int64 batch_a, const int64 batch_b, const int64 paralle_num,
                  const int64 pow_num);
};

template <typename Device, typename Scalar, typename TIndex>
struct LaunchOptCoActionIndicator {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num, int64 pow_num);
};

template <bool use_indicator, typename Scalar, typename TIndex, int POW_NUM,
          int M_SIZE, int K_SIZE, int N_SIZE>
void ComputeCoActionIndicator(const OpKernelContext* context, const Scalar* a_ptr,
                  const Scalar* b_ptr, const TIndex* indicator, Scalar* c_ptr,
                  const int64 batch_a, const int64 batch_b, const int64 paralle_num);

template <typename Device, typename Scalar, typename TIndex>
struct LaunchCoActionIndicator {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num, int64 pow_num);
};

#if GOOGLE_CUDA
template <typename Scalar>
struct LaunchCoAction<GPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b, Tensor* out,
                  int64 batch_a, int64 batch_b, int64 paralle_num,
                  int64 pow_num);
};
template <typename Scalar, typename TIndex>
struct LaunchCoActionIndicator<GPUDevice, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num, int64 pow_num);
};

#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_CO_ACTION_OP_H
