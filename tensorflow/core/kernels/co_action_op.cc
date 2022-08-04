//
// Created by qiaoxj on 2020/9/7.
//

#include "tensorflow/core/kernels/co_action_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

static std::chrono::high_resolution_clock::time_point _start = 
    std::chrono::high_resolution_clock::now();
template <typename T>
void ShowLog(const T& msg) {
  auto _now = std::chrono::high_resolution_clock::now();

  std::cout << ">>>>>>>>> marvin test <<<<<<<<<" << std::endl
            << ">>>- time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(_now - _start).count()
            << " ms; " << std::endl
            << ">>>-  msg = " << msg << std::endl
            << ">>>-------------------------<<<" << std::endl;
  _start = _now;
}

template <typename T>
void ShowLog(
  const std::chrono::high_resolution_clock::time_point start,
  const std::chrono::high_resolution_clock::time_point end,
  const T& msg) {
  std::cout << ">>>>>>>>> marvin test <<<<<<<<<" << std::endl
            << " - time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms; " << std::endl
            << " -  msg = " << msg << std::endl
            << "-------------------------------" << std::endl;
}

template <typename T>
Status PowCPU(const CPUDevice& d, size_t m, size_t n, const T* in, T* out,
              int64 pow_num) {
  typename tensorflow::TTypes<const T>::Matrix in_matrix(in, m, n);
  typename tensorflow::TTypes<T>::Matrix out_matrix(out, m, n);
  out_matrix.device(d) = in_matrix.pow((T)pow_num);
  return Status::OK();
}

template <typename T>
Status CoActionCPU(const CPUDevice& d, size_t m, size_t n, size_t k, const T* a,
                   const T* b, T* c) {
  typename tensorflow::TTypes<const T>::Matrix a_matrix(a, m, k);
  typename tensorflow::TTypes<const T>::Matrix b_matrix(b, k, n);
  typename tensorflow::TTypes<T>::Matrix c_matrix(c, 1, n);
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = 1;
  dim_pair[0].second = 0;
  Eigen::array<int, 1> reduce_dims({0});
  c_matrix.device(d) =
      a_matrix.contract(b_matrix, dim_pair).tanh().sum(reduce_dims);
  return Status::OK();
}

template <typename Scalar>
struct LaunchCoAction<CPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                    const Tensor& in_a, const Tensor& in_b, Tensor* out,
                    int64 batch_a, int64 batch_b, int64 paralle_num,
                    int64 pow_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
    Tensor tmp_pow;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Scalar>::value,
        TensorShape({batch_a, paralle_num, pow_num * m, k}), &tmp_pow));
    auto tmp_pow_ptr = tmp_pow.template flat<Scalar>().data();
    // power and concat
    for (int64 batch = 0; batch < batch_a; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(PowCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, k,
              a_ptr + (batch * paralle_num + p) * m * k,
              tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k +
                  pow * m * k,
              pow + 1));
        }
      }
    }

    for (int64 batch = 0; batch < batch_b; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(CoActionCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, n, k,
              tmp_pow_ptr + (p * pow_num + pow) * m * k,
              b_ptr + (batch * paralle_num + p) * k * n,
              c_ptr + ((batch * paralle_num + p) * pow_num + pow) * 1 * n));
        }
      }
    }
    return Status::OK();
  }
};

template <typename Scalar, typename TIndex>
struct LaunchCoActionIndicator<CPUDevice, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                    const Tensor& in_a, const Tensor& in_b,
                    const Tensor& indicator, Tensor* out, int64 batch_a,
                    int64 batch_b, int64 paralle_num, int64 pow_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
    auto ind_ptr = indicator.template flat<TIndex>().data();
    Tensor tmp_pow;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Scalar>::value,
        TensorShape({batch_a, paralle_num, pow_num * m, k}), &tmp_pow));
    auto tmp_pow_ptr = tmp_pow.template flat<Scalar>().data();

    // power and concat
    for (int64 batch = 0; batch < batch_a; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(PowCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, k,
              a_ptr + (batch * paralle_num + p) * m * k,
              tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k +
                  pow * m * k,
              pow + 1));
        }
      }
    }
    for (int64 batch = 0; batch < batch_b; batch++) {
      auto ind = (int64)ind_ptr[batch];
      if (ind < 0 || ind >= batch_a) {
        // printf("Indicator ERROR for indicator_matmul, indicator: %d.\n",
        // ind);
        ind = 0;
      }
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(CoActionCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, n, k,
              tmp_pow_ptr + ((ind * paralle_num + p) * pow_num + pow) * m * k,
              b_ptr + (batch * paralle_num + p) * k * n,
              c_ptr + ((batch * paralle_num + p) * pow_num + pow) * 1 * n));
        }
      }
    }
    return Status::OK();
  }
};

template <typename Device, typename Scalar>
class CoActionOp : public OpKernel {
 public:
  explicit CoActionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pow_num", &pow_num));
  }

  ~CoActionOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);

    OP_REQUIRES(ctx, a.dims() == 4,
                errors::InvalidArgument("In[0] ndims must be 4: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 4,
                errors::InvalidArgument("In[1] ndims must be 4: ", b.dims()));
    // currently only support m=150/50, k=5, n=4, pow=2
    OP_REQUIRES(ctx, pow_num == 2,
                errors::InvalidArgument("pow_num must == 2: ", pow_num));
    OP_REQUIRES(
        ctx, a.dim_size(2) == 50 || a.dim_size(2) == 150,
        errors::InvalidArgument("m must be 50 or 150: ", a.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(2) == 5,
                errors::InvalidArgument("k must be 5: ", b.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(3) == 4,
                errors::InvalidArgument("n must be 4: ", b.dim_size(3)));

    int64 d0 = a.dim_size(2);
    int64 d1 = a.dim_size(3);
    int64 d2 = b.dim_size(2);
    int64 d3 = b.dim_size(3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    int64 batch_a = a.dim_size(0);
    int64 batch_b = b.dim_size(0);
    OP_REQUIRES(ctx, batch_a == 1,
                errors::InvalidArgument("batch_a must be 1: a_shape = ",
                                        a.shape().DebugString()));
    int64 parallel_a = a.dim_size(1);
    int64 parallel_b = b.dim_size(1);
    OP_REQUIRES(ctx, parallel_a == parallel_b,
                errors::InvalidArgument(
                    "parallel_a mismatch parallel_b : ", parallel_a, " vs. ",
                    parallel_b, ": ", a.shape().DebugString(), " ",
                    b.shape().DebugString()));
    TensorShape out_shape({batch_b, parallel_a, pow_num, d3});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (a.NumElements() == 0 || b.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    //[PROF-STATS]
    int64 delta = 2 * d1 * out_shape.num_elements() * pow_num;
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta
                << ", " << type_string()
                << ", " << name()
                << ", " << a.shape().DebugString()
                << ", " << b.shape().DebugString();
    }

    OP_REQUIRES_OK(ctx, LaunchCoAction<Device, Scalar>()(ctx, d0, d3, d1, a, b,
                                                         out, batch_a, batch_b,
                                                         parallel_a, pow_num));
  }

 private:
  int64 pow_num;
};

//----------------------------------------------------------------------------//
// Optimize code are below.                                                   //
//----------------------------------------------------------------------------//
#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

template <typename T>
Status OptPowCPU(const CPUDevice& d, size_t m, size_t n, const T* in, T* out,
              int64 pow_num) {
  typename tensorflow::TTypes<const T>::Matrix in_matrix(in, m, n);
  typename tensorflow::TTypes<T>::Matrix out_matrix(out, m, n);
  out_matrix.device(d) = in_matrix.pow((T)pow_num);
  return Status::OK();
}

template <typename T>
Status OptCoActionCPU(const CPUDevice& d, size_t m, size_t n, size_t k, const T* a,
                   const T* b, T* c) {
  typename tensorflow::TTypes<const T>::Matrix a_matrix(a, m, k);
  typename tensorflow::TTypes<const T>::Matrix b_matrix(b, k, n);
  typename tensorflow::TTypes<T>::Matrix c_matrix(c, 1, n);
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = 1;
  dim_pair[0].second = 0;
  Eigen::array<int, 1> reduce_dims({0});
  c_matrix.device(d) =
      a_matrix.contract(b_matrix, dim_pair).tanh().sum(reduce_dims);
  return Status::OK();
}

template <typename Scalar>
struct LaunchOptCoAction<CPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                    const Tensor& in_a, const Tensor& in_b, Tensor* out,
                    int64 batch_a, int64 batch_b, int64 paralle_num,
                    int64 pow_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
    Tensor tmp_pow;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Scalar>::value,
        TensorShape({batch_a, paralle_num, pow_num * m, k}), &tmp_pow));
    auto tmp_pow_ptr = tmp_pow.template flat<Scalar>().data();
    // power and concat
#ifdef __AVX512F__
    const int total_element = m * k;
    const int total_loops = total_element / 16;
    const int remain = total_element % 16;
    const __mmask16 row_mask =
        (static_cast<std::uint32_t>(1) << remain) - 1;

    __m512 element_row;
    for (int64 batch = 0; batch < batch_a; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        int loop = 0;
        for (; loop < total_loops; loop++){
          element_row = _mm512_loadu_ps(a_ptr + (batch * paralle_num + p) * m * k + 16 * loop);
          _mm512_storeu_ps( // pow(1)
            tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k + 16 * loop, element_row);
          _mm512_storeu_ps( // pow(2)
            tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k + m * k + 16 * loop,
            _mm512_mul_ps(element_row, element_row));
        }
        if (remain){
          element_row = _mm512_maskz_loadu_ps(row_mask, a_ptr + (batch * paralle_num + p) * m * k + 16 * loop);
          _mm512_mask_storeu_ps( // pow(1)
            tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k + 16 * loop, row_mask, element_row);
          _mm512_mask_storeu_ps( // pow(2)
            tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k + m * k + 16 * loop, row_mask,
            _mm512_mul_ps(element_row, element_row));
        }
      }
    }
#else
    for (int64 batch = 0; batch < batch_a; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(OptPowCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, k,
              a_ptr + (batch * paralle_num + p) * m * k,
              tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k +
                  pow * m * k,
              pow + 1));
        }
      }
    }
#endif

    // MatMul + tanh + sum parts
#ifdef __AVX512F__
    // ShowLog("OptCoActionOp:LaunchCoAction:start DNN");

    // float* temp_c = new float[2 * m];
    // float* temp_b = new float[k];
    // const int total_element = m * k;
    // const int total_loops = total_element / 16;
    // const int remain = total_element % 16;
    // const __mmask16 row_mask =
    //     (static_cast<std::uint32_t>(1) << remain) - 1;

    // __m512 element_row;
    // __m512 element_col[n];

    // for (int64 batch = 0; batch < batch_b; batch++) {
    //   for (int64 p = 0; p < paralle_num; p++) {
    //     // 1. init B
    //     for(int iter; iter < n; iter++){
    //       float* _tmp_b_ptr = b_ptr + (batch * paralle_num + p) * k * n;
    //       // hard code here, cause of the k is fixed as 5.
    //       element_col[iter] = _mm512_set_ps(
    //         /* 15 */0,
    //         /* 14 */ADDRESS(_tmp_b_ptr, 4, 0, n),
    //         /* 13 */ADDRESS(_tmp_b_ptr, 3, 0, n),
    //         /* 12 */ADDRESS(_tmp_b_ptr, 2, 0, n),
    //         /* 11 */ADDRESS(_tmp_b_ptr, 1, 0, n),
    //         /* 10 */ADDRESS(_tmp_b_ptr, 0, 0, n),
    //         /*  9 */ADDRESS(_tmp_b_ptr, 4, 0, n),
    //         /*  8 */ADDRESS(_tmp_b_ptr, 3, 0, n),
    //         /*  7 */ADDRESS(_tmp_b_ptr, 2, 0, n),
    //         /*  6 */ADDRESS(_tmp_b_ptr, 1, 0, n),
    //         /*  5 */ADDRESS(_tmp_b_ptr, 0, 0, n),
    //         /*  4 */ADDRESS(_tmp_b_ptr, 4, 0, n),
    //         /*  3 */ADDRESS(_tmp_b_ptr, 3, 0, n),
    //         /*  2 */ADDRESS(_tmp_b_ptr, 2, 0, n),
    //         /*  1 */ADDRESS(_tmp_b_ptr, 1, 0, n),
    //         /*  0 */ADDRESS(_tmp_b_ptr, 0, 0, n),
    //       )
    //     }
    //     // 2. load part of A
    //     for(int iter; iter < n; iter++){

    //       element_col[iter]
    //     }

        
    //   }
    // }
    for (int64 batch = 0; batch < batch_b; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(OptCoActionCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, n, k,
              tmp_pow_ptr + (p * pow_num + pow) * m * k,
              b_ptr + (batch * paralle_num + p) * k * n,
              c_ptr + ((batch * paralle_num + p) * pow_num + pow) * 1 * n));
        }
      }
    }
    // ShowLog("OptCoActionOp:LaunchCoAction:end DNN");
#else
    for (int64 batch = 0; batch < batch_b; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(OptCoActionCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, n, k,
              tmp_pow_ptr + (p * pow_num + pow) * m * k,
              b_ptr + (batch * paralle_num + p) * k * n,
              c_ptr + ((batch * paralle_num + p) * pow_num + pow) * 1 * n));
        }
      }
    }
#endif
    return Status::OK();
  }
};

template <typename Device, typename Scalar>
class OptCoActionOp : public OpKernel {
 public:
  explicit OptCoActionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pow_num", &pow_num));
  }

  ~OptCoActionOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);

    OP_REQUIRES(ctx, a.dims() == 4,
                errors::InvalidArgument("In[0] ndims must be 4: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 4,
                errors::InvalidArgument("In[1] ndims must be 4: ", b.dims()));
    // currently only support m=150/50, k=5, n=4, pow=2
    OP_REQUIRES(ctx, pow_num == 2,
                errors::InvalidArgument("pow_num must == 2: ", pow_num));
    OP_REQUIRES(
        ctx, a.dim_size(2) == 50 || a.dim_size(2) == 150,
        errors::InvalidArgument("m must be 50 or 150: ", a.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(2) == 5,
                errors::InvalidArgument("k must be 5: ", b.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(3) == 4,
                errors::InvalidArgument("n must be 4: ", b.dim_size(3)));

    int64 d0 = a.dim_size(2);
    int64 d1 = a.dim_size(3);
    int64 d2 = b.dim_size(2);
    int64 d3 = b.dim_size(3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    int64 batch_a = a.dim_size(0);
    int64 batch_b = b.dim_size(0);
    OP_REQUIRES(ctx, batch_a == 1,
                errors::InvalidArgument("batch_a must be 1: a_shape = ",
                                        a.shape().DebugString()));
    int64 parallel_a = a.dim_size(1);
    int64 parallel_b = b.dim_size(1);
    OP_REQUIRES(ctx, parallel_a == parallel_b,
                errors::InvalidArgument(
                    "parallel_a mismatch parallel_b : ", parallel_a, " vs. ",
                    parallel_b, ": ", a.shape().DebugString(), " ",
                    b.shape().DebugString()));
    TensorShape out_shape({batch_b, parallel_a, pow_num, d3});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (a.NumElements() == 0 || b.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    //[PROF-STATS]
    int64 delta = 2 * d1 * out_shape.num_elements() * pow_num;
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta
                << ", " << type_string()
                << ", " << name()
                << ", " << a.shape().DebugString()
                << ", " << b.shape().DebugString();
    }

    OP_REQUIRES_OK(ctx, LaunchOptCoAction<Device, Scalar>()(ctx, d0, d3, d1, a, b,
                                                         out, batch_a, batch_b,
                                                         parallel_a, pow_num));
  }

 private:
  int64 pow_num;
};

template <typename Device, typename Scalar, typename TIndex>
class CoActionIndicatorOp : public OpKernel {
 public:
  explicit CoActionIndicatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pow_num", &pow_num));
  }

  ~CoActionIndicatorOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);
    auto& ind = ctx->input(2);

    OP_REQUIRES(ctx, a.dims() == 4,
                errors::InvalidArgument("In[0] ndims must be 4: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 4,
                errors::InvalidArgument("In[1] ndims must be 4: ", b.dims()));
    OP_REQUIRES(ctx, ind.dims() == 1,
                errors::InvalidArgument("In[2] ndims must be 1: ", ind.dims()));
    // currently only support m=150/50, k=5, n=4, pow=2
    OP_REQUIRES(
        ctx, a.dim_size(2) == 50 || a.dim_size(2) == 150,
        errors::InvalidArgument("m must be 50 or 150: ", a.dim_size(2)));
    OP_REQUIRES(ctx, pow_num == 2,
                errors::InvalidArgument("pow_num must == 2: ", pow_num));
    OP_REQUIRES(ctx, b.dim_size(2) == 5,
                errors::InvalidArgument("k must be 5: ", b.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(3) == 4,
                errors::InvalidArgument("n must be 4: ", b.dim_size(3)));

    int64 d0 = a.dim_size(2);
    int64 d1 = a.dim_size(3);
    int64 d2 = b.dim_size(2);
    int64 d3 = b.dim_size(3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    int64 parallel_a = a.dim_size(1);
    int64 parallel_b = b.dim_size(1);
    OP_REQUIRES(ctx, parallel_a == parallel_b,
                errors::InvalidArgument(
                    "parallel_a mismatch parallel_b : ", parallel_a, " vs. ",
                    parallel_b, ": ", a.shape().DebugString(), " ",
                    b.shape().DebugString()));
    int64 batch_a = a.dim_size(0);
    int64 batch_b = b.dim_size(0);
    int64 ind_length = ind.dim_size(0);
    OP_REQUIRES(
        ctx, batch_b == ind_length,
        errors::InvalidArgument(
            "b_batch mismatch indicator length: ", batch_b, " vs. ", ind_length,
            ": ", b.shape().DebugString(), " ", ind.shape().DebugString()));
    TensorShape out_shape({batch_b, parallel_a, pow_num, d3});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (a.NumElements() == 0 || b.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    //[PROF-STATS]
    int64 delta = 2 * d1 * out_shape.num_elements() * pow_num;
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta
                << ", " << type_string()
                << ", " << name()
                << ", " << a.shape().DebugString()
                << ", " << b.shape().DebugString();
    }

    OP_REQUIRES_OK(ctx, LaunchCoActionIndicator<Device, Scalar, TIndex>()(
                            ctx, d0, d3, d1, a, b, ind, out, batch_a, batch_b,
                            parallel_a, pow_num));
  }

 private:
  int64 pow_num;
};

#define REGISTER_COACTION_CPU(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("CoAction").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),    \
      CoActionOp<CPUDevice, TYPE>);                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("OptCoAction").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),\
      OptCoActionOp<CPUDevice, TYPE>);                                  \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                     \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<TYPE>("T")                \
                              .TypeConstraint<int32>("Tindices"),       \
                          CoActionIndicatorOp<CPUDevice, TYPE, int32>); \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                     \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<TYPE>("T")                \
                              .TypeConstraint<int64>("Tindices"),       \
                          CoActionIndicatorOp<CPUDevice, TYPE, int64>);

REGISTER_COACTION_CPU(float);
REGISTER_COACTION_CPU(Eigen::half);

#if GOOGLE_CUDA
#define REGISTER_COACTION_GPU(TYPE)                                       \
  extern template struct LaunchCoAction<GPUDevice, TYPE>;                 \
  extern template struct LaunchCoActionIndicator<GPUDevice, TYPE, int32>; \
  extern template struct LaunchCoActionIndicator<GPUDevice, TYPE, int64>; \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CoAction").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),      \
      CoActionOp<GPUDevice, TYPE>);                                       \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                       \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint<int32>("Tindices"),         \
                          CoActionIndicatorOp<GPUDevice, TYPE, int32>);   \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                       \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint<int64>("Tindices"),         \
                          CoActionIndicatorOp<GPUDevice, TYPE, int64>);

REGISTER_COACTION_GPU(float);
REGISTER_COACTION_GPU(Eigen::half);
#endif

}  // namespace tensorflow
