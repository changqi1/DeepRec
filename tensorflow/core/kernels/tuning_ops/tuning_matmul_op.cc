/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl_matmul_ops_common.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/util/mkl_util.h"

#include <stdio.h>
#include "tunable_matmul.h"

#define TUNED_CONFIG_FILE "./tuned"

const MatmulImpl TunableMatmul::impl_list[] = {
    {"v1", v1},
    {"v2", v2},
    {"v3", v3},
    {"v4", v4},
    {"v5", v5},
    {"v6", v6},
    {"v7", v7},
    {"v8", v8},
    {"v9", v9},
    {"v10", v10},
    {"v11", v11},
    {"v12", v12},
    {"v13", v13},
    {"v14", v14},
    {"v15", v15},
    {"v16", v16},
    {"v17", v17},
    {"v18", v18},
    {"v19", v19},
    {"v20", v20},
    {"v21", v21},
    {"v22", v22},
    {"v23", v23},
    {"v24", v24},
    {"v100", v100},
    {"v101", v101},
    {"v102", v102},
    {"v103", v103},
    {"v104", v104},
    {"v105", v105},
    {"v106", v106},
    {"v107", v107},
    {"v108", v108},
    {"v109", v109},
    {"v110", v110},
    {"v111", v111},
    {"", nullptr},
};

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class TuningMatMulOp : public OpKernel {
 public:
  explicit TuningMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // TF_CHECK_OK(ReadBoolFromEnvVar("TF_TUNING_FILE",
    //                         /*default_val=*/false, &tune_));
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_TUNING_ENABLE",
                            /*default_val=*/true, &tune_));
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_TUNING_ENABLE_HOST",
                            /*default_val=*/true, &host_));
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_TUNING_MAX_ITER",
                            /*default_val=*/0, &max_iters_));
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_TUNING_ITER_PRE_CYCLE",
                            /*default_val=*/0, &iter_per_cycle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

    tmm_ = new TunableMatmul();
    tmm_->SetConditions(iter_per_cycle_, max_iters_);
  }

  ~TuningMatMulOp() {
    delete(tmm_);
  }

  void Compute(OpKernelContext* ctx) override {
    tmm_->SetThreadPool(ctx->device()->tensorflow_cpu_worker_threads()->workers);

    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      auto o = out->flat<T>();
      o.device(ctx->eigen_device<Device>()) = o.constant(T(0));
      return;
    }

    const int m = a.dim_size(1 - dim_pair[0].first);
    const int k = a.dim_size(dim_pair[0].first);
    const int n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;

    auto a_ptr = (a.template flat<T>().data());
    auto b_ptr = (b.template flat<T>().data());
    auto c_ptr = (out->template flat<T>().data());

    if(m < 64 || k < 64 || n < 64){
      MklBlasGemm(ctx, transpose_a, transpose_b, m, n, k, a_ptr,
                  transpose_a ? m : k, b_ptr, transpose_b ? k : n, c_ptr, n);
    } else {
      TuningGemm(ctx, transpose_a, transpose_b, m, n, k, a_ptr,
                  transpose_a ? m : k, b_ptr, transpose_b ? k : n, c_ptr, n);
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  TunableMatmul* tmm_;
  int64 max_iters_ = 0;
  int64 iter_per_cycle_ = 0;
  bool tune_ = false;
  bool host_ = false;

  void TuningGemm(OpKernelContext* ctx, bool transa, bool transb, const int m,
                  const int n, const int k, const float* a, const int lda,
                  const float* b, const int ldb, float* c, const int ldc) {
    // Timer t_;
    /*tmm compute*/
    tmm_->SetParams(m, n, k, lda, ldb, ldc, a, b, c);

    // bool flush_b = false;
    if (tune_) {
      if(host_) {
        tmm_->host_tune(false, a, b, c);
      } else {
        tmm_->tune(false, a, b, c);
      }
      // tmm_->save_config(TUNED_CONFIG_FILE);
    } else {
      if (!tmm_->load_config(TUNED_CONFIG_FILE)){
          printf("Cannot load matmul config.\n");
          // exit(-1);
      }
      tmm_->compute(a, b, c);
    }
    // printf("Host Compute Time: %f ms\n", t_.getTime());
  }




  // --------------------------------------------------------------------------
  //
  // @brief Matrix-Matrix Multiplication with FP32 tensors, a, b, c using CBLAS
  // interface. c = op(a) * op(b)
  //
  // @param transa  Specifies the form of op(a) used in MatMul. If transa is
  // true, then op(a) = a^T, otherwise op(a) = a
  //
  // @param transb  Specifies the form of op(b) used in MatMul. If transb is
  // true, then op(b) = b^T, otherwise op(b) = b
  //
  // @param m       Specifies the number of rows of the matrix op(a) and of the
  // matrix c. The value of m must be at least zero.
  //
  // @param n       Specifies the number of columns of the matrix op(b) and the
  // number of columns of the matrix c. The value of n must be at least zero.
  //
  // @param k       Specifies the number of columns of the matrix op(a) and the
  // number of rows of the matrix op(b)
  //
  // @param a       Address of matrix a
  //
  // @param lda     Leading dimension of 'a' matrix. This is set at calling site
  // depending on transa parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // lda = max(1,k) when transa is false, otherwise lda = max(1,m)
  //
  // @param b       Address of matrix b
  //
  // @param ldb     Leading dimension of 'b' matrix. This is set at calling site
  // depending on transb parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // ldb = max(1,n) when transb is false, otherwise ldb = max(1,k)
  //
  // @param c       Address of matrix c
  //
  // @param ldc     Leading dimension of 'c' matrix. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows, max(1,n)
  //
  // --------------------------------------------------------------------------
  void MklBlasGemm(OpKernelContext* ctx, bool transa, bool transb, const int m,
                   const int n, const int k, const float* a, const int lda,
                   const float* b, const int ldb, float* c, const int ldc) {
    // BLAS GEMM API defines Matrix Multiplication as c = alpha * op(a) * op(b)
    // + beta * c.
    // Since TF MatMul does not have parameters for alpha, beta, we set them to
    // 1.0 and 0.0 respectively.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    char char_transa = transa ? 'T' : 'N';
    char char_transb = transb ? 'T' : 'N';
    VLOG(2) << "MKL DNN SGEMM called";
#ifdef ENABLE_DNNL_THREADPOOL
    // With threadpool , the runtime overhead is comparable to the kernel
    // execution for small kernel sizes. For such sizes, it may be better to run
    // the kernel single threaded. Here we are coming up with a cost model based
    // on L1 sizes. If we find that matrices are small enough, we will execute
    // single threaded. This may need tuning.
    if (ExecuteSingleThreadedGemm(m, n, k)) {
      // For now, call single-threaded gemm.
      MklDnnThreadPool eigen_tp(ctx, 1);
      sgemm(char_transa, char_transb, m, n, k, alpha, a, lda, b, ldb,
                    beta, c, ldc, &eigen_tp);
    } else {
      MklDnnThreadPool eigen_tp(ctx);
      sgemm(char_transa, char_transb, m, n, k, alpha, a, lda, b, ldb,
                    beta, c, ldc, &eigen_tp);
    }
#else
    dnnl_sgemm(char_transa, char_transb, m, n, k, alpha, a, lda, b, ldb, beta,
               c, ldc);
#endif  // ENABLE_DNNL_THREADPOOL
  }

  void MklBlasGemm(OpKernelContext* ctx, bool transa, bool transb, const int m,
                   const int n, const int k, const bfloat16* a, const int lda,
                   const bfloat16* b, const int ldb, bfloat16* c,
                   const int ldc) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int index_transa = transa ? 1 : 0;
    const int index_transb = transb ? 1 : 0;
    const char ftrans[] = {'N', 'T', 'C'};
    dnnl_gemm<bfloat16>(ftrans[index_transa], ftrans[index_transb], m, n, k,
                        alpha, a, lda, b, ldb, beta, c, ldc, ctx);
  }

};

// template <typename Device, typename T>
// TunableMatmul* TuningMatMulOp<Device, T>::tmm_ = nullptr;

// packed matmul
REGISTER_KERNEL_BUILDER(Name("TuningMatmul")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<float>("T")
                        .Label(mkl_op_registry::kMklNameChangeOpLabel),
                        TuningMatMulOp<CPUDevice, float>);

}  // namespace tensorflow
