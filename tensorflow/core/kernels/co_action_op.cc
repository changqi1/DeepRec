//
// Created by qiaoxj on 2020/9/7.
//
#include <numeric>

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
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

static std::chrono::high_resolution_clock::time_point _start = 
    std::chrono::high_resolution_clock::now();
template <typename T>
void ShowLog(const T& msg) {
  auto _now = std::chrono::high_resolution_clock::now();

  VLOG(1) << ">>>>>>>>> marvin test <<<<<<<<<" << std::endl
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
  VLOG(1) << ">>>>>>>>> marvin test <<<<<<<<<" << std::endl
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

//----------------------------------------------------------------------------//
// Optimize code are below.                                                   //
//----------------------------------------------------------------------------//
#define ADDRESS_A(a_ptr, pn, m, k) \
  ((a_ptr) + (pn) * M_SIZE * K_SIZE + (m) * K_SIZE + (k))

#define ADDRESS_A_IND(a_ptr, ind, pn, m, k) \
  ((a_ptr) + ((ind) * paralle_num + (pn)) * M_SIZE * K_SIZE + (m) * K_SIZE + (k))

#define ADDRESS_B(b_ptr, bs, pn, k, n) \
  ((b_ptr) + ((bs) * paralle_num + (pn))* K_SIZE * N_SIZE + (k) * N_SIZE + (n))

#define ADDRESS_C(c_ptr, bs, pn, p, n) \
  ((c_ptr) + (((bs) * paralle_num + (pn)) * POW_NUM + (p)) * N_SIZE + (n))


#define FITTING_VA(i, mask)                                                                     \
  {                                                                                             \
    auto tmp_ind = (int64)indicator[batch + (i)];                                                      \
    tmp_ind = -1 < tmp_ind < batch_a ? tmp_ind : 0;                                             \
    __m512 tmp_va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS_A_IND(a_ptr, tmp_ind, p, m, k))); \
    va = _mm512_mask_blend_ps(row_mask_##mask, va, tmp_va);                                     \
  }                                                                                             \

inline __m512 avx512_exp(const __m512& _x) {
  __m512 p16f_1 = _mm512_set1_ps(1.0f);
  __m512 p16f_half = _mm512_set1_ps(0.5f);
  __m512 p16f_127 = _mm512_set1_ps(127.f);
  __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
  __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

  __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

  __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
  __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
  __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
  __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
  __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
  __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

  // Clamp x.
  __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x,
                              p16f_cephes_LOG2EF, p16f_half));

  // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
  // subtracted out in two parts, m*C1+m*C2 = m*ln(2),
  // to avoid accumulating truncation errors.
  // Note that we don't use the "pmadd" function here to
  // ensure that a precision-preserving FMA instruction is used.
  __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
  __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

  __m512 r2 = _mm512_mul_ps(r, r);

  // TODO(gonnet): Split into odd/even polynomials and try to exploit
  //               instruction-level parallelism.
  __m512 y = p16f_cephes_exp_p0;
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
  y = _mm512_fmadd_ps(y, r2, r);
  y = _mm512_add_ps(y, p16f_1);

  // Build emm0 = 2^m.
  __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);

  // Return 2^m * exp(r).
  return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
}

inline __m512 avx512_tanh(__m512 x) {
  x = _mm512_min_ps(x, _mm512_set1_ps(10.0f));
  x = _mm512_max_ps(x, _mm512_set1_ps(-10.0f));
  x = _mm512_mul_ps(x, _mm512_set1_ps(2.0f));
  x = avx512_exp(x);
  x = _mm512_div_ps(_mm512_sub_ps(x, _mm512_set1_ps(1.0f)),
                    _mm512_add_ps(x, _mm512_set1_ps(1.0f)));
  return x;
}

template <typename Scalar>
struct LaunchOptCoAction<CPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, const int64 m, const int64 n, const int64 k,
                  const Tensor& in_a, const Tensor& in_b, Tensor* out,
                  const int64 batch_a, const int64 batch_b, const int64 paralle_num,
                  const int64 pow_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
#ifdef __AVX512F__
    if (m == 50 && k == 5 && n == 4 && pow_num == 2) {
      ComputeCoActionIndicator<false, Scalar, Scalar, 2, 50, 5, 4>(
        context, a_ptr, b_ptr, nullptr, c_ptr, batch_a, batch_b, paralle_num);
    } else if (m == 150 && k == 5 && n == 4 && pow_num == 2) {
      ComputeCoActionIndicator<false, Scalar, Scalar, 2, 150, 5, 4>(
        context, a_ptr, b_ptr, nullptr, c_ptr, batch_a, batch_b, paralle_num);
    } else {
      return errors::InvalidArgument("Unsupported m, k, n, pow_num: ", m, k, n,
                                    pow_num);
    }
#endif
    return Status::OK();
  }
};

template <typename Scalar, typename TIndex>
struct LaunchOptCoActionIndicator<CPUDevice, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num, int64 pow_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
    auto ind_ptr = indicator.template flat<TIndex>().data();
#ifdef __AVX512F__
    if (m == 50 && k == 5 && n == 4 && pow_num == 2) {
      ComputeCoActionIndicator<true, Scalar, TIndex, 2, 50, 5, 4>(
        context, a_ptr, b_ptr, ind_ptr, c_ptr, batch_a, batch_b, paralle_num);
    } else if (m == 150 && k == 5 && n == 4 && pow_num == 2) {
      ComputeCoActionIndicator<true, Scalar, TIndex, 2, 150, 5, 4>(
        context, a_ptr, b_ptr, ind_ptr, c_ptr, batch_a, batch_b, paralle_num);
    } else {
      return errors::InvalidArgument("Unsupported m, k, n, pow_num: ", m, k, n,
                                    pow_num);
    }
#endif
    return Status::OK();
  }
};

template <bool use_indicator, typename Scalar, typename TIndex,
           /* 2 */int POW_NUM, /* 150/50 */int M_SIZE, /* 5 */int K_SIZE, /* 4 */int N_SIZE>  
void ComputeCoActionIndicator(const OpKernelContext* context, const Scalar* a_ptr,
                  const Scalar* b_ptr, const TIndex* indicator, Scalar* c_ptr,
                  const int64 batch_a, const int64 batch_b, const int64 paralle_num){
#ifdef __AVX512F__
  {
    const __mmask16 row_mask_0 =
        (static_cast<std::uint32_t>(1) << 4) - 1;
    const __mmask16 row_mask_4 =
        ((static_cast<std::uint32_t>(1) << 4) - 1) << 4;
    const __mmask16 row_mask_8 =
        ((static_cast<std::uint32_t>(1) << 4) - 1) << 8;
    const __mmask16 row_mask_12 =
        ((static_cast<std::uint32_t>(1) << 4) - 1) << 12;

    __m512 va, vb, vc, vc2, v_rtn, v_rtn2;
    __m512 v_zero = _mm512_setzero_ps();
    
    for (int64 batch = 0; batch < batch_b; batch+=4) {
      for (int64 p = 0; p < paralle_num; p++) {
        v_rtn = v_zero;
        v_rtn2 = v_zero;
        for(int64 m = 0; m < M_SIZE; m++) {
          vc = v_zero;
          vc2 = v_zero;
          for(int64 k = 0; k < K_SIZE; k++) {
            // va=broadcasts(a_ptr[m,k]), vb=b_ptr[k, 0~n] * 4 * batchSize
            if(use_indicator){
              FITTING_VA(0, 0);
              FITTING_VA(1, 4);
              FITTING_VA(2, 8);
              FITTING_VA(3, 12);
            } else {
              va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS_A(a_ptr, p, m, k)));
            }
            // todo(marvin):  Do we have a better way to load vb?
            vb = _mm512_set_ps(
              *ADDRESS_B(b_ptr, batch + 3, p, k, 3), *ADDRESS_B(b_ptr, batch + 3, p, k, 2),
              *ADDRESS_B(b_ptr, batch + 3, p, k, 1), *ADDRESS_B(b_ptr, batch + 3, p, k, 0),
              *ADDRESS_B(b_ptr, batch + 2, p, k, 3), *ADDRESS_B(b_ptr, batch + 2, p, k, 2),
              *ADDRESS_B(b_ptr, batch + 2, p, k, 1), *ADDRESS_B(b_ptr, batch + 2, p, k, 0),
              *ADDRESS_B(b_ptr, batch + 1, p, k, 3), *ADDRESS_B(b_ptr, batch + 1, p, k, 2),
              *ADDRESS_B(b_ptr, batch + 1, p, k, 1), *ADDRESS_B(b_ptr, batch + 1, p, k, 0),
              *ADDRESS_B(b_ptr, batch + 0, p, k, 3), *ADDRESS_B(b_ptr, batch + 0, p, k, 2),
              *ADDRESS_B(b_ptr, batch + 0, p, k, 1), *ADDRESS_B(b_ptr, batch + 0, p, k, 0)
            );
            vc = _mm512_fmadd_ps(va, vb, vc);
            vc2 = _mm512_fmadd_ps(_mm512_mul_ps(va, va), vb, vc2);
          }
          v_rtn = _mm512_add_ps(v_rtn, avx512_tanh(vc));
          v_rtn2 = _mm512_add_ps(v_rtn2, avx512_tanh(vc2));
        }

        // store 4*2 elements
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 0, p, 0, 0),   row_mask_0,  v_rtn);
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 1, p, 0, 0),  row_mask_4,  v_rtn);
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 2, p, 0, 0),  row_mask_8,  v_rtn);
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 3, p, 0, 0), row_mask_12,  v_rtn);
        
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 0, p, 1, 0),   row_mask_0, v_rtn2);
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 1, p, 1, 0),  row_mask_4, v_rtn2);
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 2, p, 1, 0),  row_mask_8, v_rtn2);
        _mm512_mask_compressstoreu_ps(ADDRESS_C(c_ptr, batch + 3, p, 1, 0), row_mask_12, v_rtn2);
        
      }
    }
  }
#endif
}
//----------------------------------------------------------------------------//

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
    ReadBoolFromEnvVar("TF_CO_ACTION_OPT", false, &enable_opt);
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

    OP_REQUIRES_OK(ctx, enable_opt ? LaunchOptCoAction<Device, Scalar>()(ctx, d0, d3, d1, a, b,
                                                         out, batch_a, batch_b,
                                                         parallel_a, pow_num)
                                    :LaunchCoAction<Device, Scalar>()(ctx, d0, d3, d1, a, b,
                                                         out, batch_a, batch_b,
                                                         parallel_a, pow_num));
  }

 private:
  int64 pow_num;
  bool enable_opt;
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
    ReadBoolFromEnvVar("TF_CO_ACTION_OPT", false, &enable_opt);
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

    OP_REQUIRES_OK(ctx, enable_opt ? LaunchOptCoActionIndicator<Device, Scalar, TIndex>()(
                            ctx, d0, d3, d1, a, b, ind, out, batch_a, batch_b,
                            parallel_a, pow_num)
                            : LaunchCoActionIndicator<Device, Scalar, TIndex>()(
                            ctx, d0, d3, d1, a, b, ind, out, batch_a, batch_b,
                            parallel_a, pow_num));
  }

 private:
  int64 pow_num;
  bool enable_opt;
};

template <typename Device, typename Scalar, typename TIndex>
class OptCoActionIndicatorOp : public OpKernel {
 public:
  explicit OptCoActionIndicatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pow_num", &pow_num));
  }

  ~OptCoActionIndicatorOp() = default;

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

    OP_REQUIRES_OK(ctx, LaunchOptCoActionIndicator<Device, Scalar, TIndex>()(
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
      Name("OptCoAction").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
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
                          CoActionIndicatorOp<CPUDevice, TYPE, int64>); \
  REGISTER_KERNEL_BUILDER(Name("OptCoActionIndicator")                    \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint<int64>("Tindices"),         \
                          OptCoActionIndicatorOp<CPUDevice, TYPE, int64>);

REGISTER_COACTION_CPU(float);
// REGISTER_COACTION_CPU(Eigen::half);

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
