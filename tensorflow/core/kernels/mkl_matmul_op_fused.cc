/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

// This file uses OneDNN InnerProduct for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) with bias (BiasAdd) operations.
#ifdef INTEL_MKL

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl_matmul_ops_common.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// Fuse Operation
template <typename Device, typename T>
class MklFusedMatMulOp : public MklDnnMatMulOpBase<T, T> {
 public:
  explicit MklFusedMatMulOp(OpKernelConstruction* ctx)
      : MklDnnMatMulOpBase<T, T>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    if (ctx->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("is_filter_const", &(this->is_weight_const_)));
    }

    OP_REQUIRES(ctx, fused_ops_.size() <= 2,
                errors::InvalidArgument(
                    "MklFusedMatMul must have 2 post-arguments at most."));
    OP_REQUIRES(
        ctx, fused_ops_[0] == "BiasAdd",
        errors::InvalidArgument(
            "The 1st post-argument of MklFusedMatMul must be BiasAdd."));
    if (fused_ops_.size() > 1 && fused_ops_[1] == "Add") fuse_add_ = true;
    OP_REQUIRES(
        ctx, transpose_a_ == false,
        errors::InvalidArgument("In[0] of MklMatMul can't be transposed."));
  }

  void Compute(OpKernelContext* ctx) override {
    // FusedMatMul has 3 inputs: src, weights, bias
    const Tensor& src_tensor = ctx->input(this->kInputIndexSrc);
    const Tensor& weight_tensor = ctx->input(this->kInputIndexWeight);
    const Tensor& bias_tensor = MklGetInput(ctx, this->kInputIndexBias);

    // Get shapes of input tensors
    auto src_tf_shape = src_tensor.shape();
    auto weight_tf_shape = weight_tensor.shape();

    // Check the constraint of input matrix and bias
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(src_tf_shape),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weight_tf_shape),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias_tensor.shape()),
                errors::InvalidArgument("Biases must be 1D"));

    // Expression: [batch, k] * [k, channel] + [channel] = [batch, channel]
    //
    // Get dimension size of each matrix, dim_pair[] is the location of k
    // in the inputs, we have constraint that k of the two inputs are
    // the same
    const int dim_pair[] = {1, transpose_b_ ? 1 : 0};
    const int batch = src_tf_shape.dim_size(1 - dim_pair[0]);
    const int k = src_tf_shape.dim_size(dim_pair[0]);
    const int channel = weight_tf_shape.dim_size(1 - dim_pair[1]);

    OP_REQUIRES(
        ctx, k == weight_tf_shape.dim_size(dim_pair[1]),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
            ", In[1]: ", weight_tf_shape.DebugString()));
    OP_REQUIRES(ctx, bias_tensor.shape().dim_size(0) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel size: ",
                    bias_tensor.shape().DebugString(), " vs. ", channel));

    // For inputs s[batch, k], w[k, channel] and b[channel], the primitive
    // dims should be described like this:
    //   s[batch, k] * w^T[channel, k] + b[channel] = dst[batch, channel]
    //    [n,    ic] *    [oc,     ic] +  [oc]      =    [n,          oc]
    memory::dims src_dims = memory::dims({batch, k});
    // Reverse the weights dims from [k, channel] to [channel, k].
    memory::dims weight_dims = memory::dims({channel, k});
    memory::dims bias_dims = memory::dims({channel});
    memory::dims dst_dims = memory::dims({batch, channel});
    MEMORY_FORMAT src_format = MEMORY_FORMAT::nc;
    MEMORY_FORMAT weight_format =
        transpose_b_ ? MEMORY_FORMAT::oi : MEMORY_FORMAT::io;

    // Set weight format for primitive:
    //   1. const, let OneDNN determine format because it will be cached;
    //   2. var, keep the original format to avoid reordering.
    MklDnnMatMulFwdParams matmul_params(
        src_dims, weight_dims, bias_dims, dst_dims, src_format,
        (this->is_weight_const_) ? MEMORY_FORMAT::any : weight_format,
        MEMORY_FORMAT::nc);

    // Extend the basic parameters for data types and fusions.
    ExtendMklDnnMatMulFwdParams(ctx, matmul_params);
    bool do_not_cache = MklPrimitiveFactory<T>::IsPrimitiveMemOptEnabled();
    MklDnnMatMulFwdPrimitive<T, T, T, T, T>* matmul_prim =
        MklDnnMatMulFwdPrimitiveFactory<T, T, T, T, T>::Get(matmul_params, do_not_cache);

    // Allocate output tensor.
    Tensor* dst_tensor = nullptr;
    std::shared_ptr<dnnl::inner_product_forward::primitive_desc> matmul_pd =
        matmul_prim->GetPrimitiveDesc();

    // The output shape of MatMul is same both for OneDNN and TF version.
    // They are all NC format, no matter what's the format of input.
    // And the shape of AddOp is also the same with output's shape.
    auto dst_pd = matmul_pd->PRIMITIVE_DESC_DST;

    TensorShape output_tf_shape({batch, channel});

    if (fuse_add_) {
      const Tensor& add_tensor = MklGetInput(ctx, kInputIndex_Add);

      if (ctx->forward_input_to_output_with_shape(
              kInputIndex_Add, kOutputIndex_Dst, output_tf_shape,
              &dst_tensor)) {

        ;  // If it's not native format, need to forward and set meta first
      } else {
        // If forward is not successful, we should use reorder to copy add
        // tensor to dst tensor
        ctx->allocate_output(kOutputIndex_Dst, output_tf_shape, &dst_tensor);
        auto output_format_tag =
            MklTensorFormatToMklDnnDataFormat(MKL_TENSOR_FORMAT_NC);
        auto add_md = memory::desc(dst_dims, MklDnnType<T>(), output_format_tag);
        auto dst_md =
            memory::desc(dst_dims, MklDnnType<T>(), output_format_tag);

        void* add_buf =
            static_cast<void*>(const_cast<T*>(add_tensor.flat<T>().data()));
        void* dst_buf = static_cast<void*>((dst_tensor)->flat<T>().data());

        auto fuse_add_src_ =
            MEMORY_CONSTRUCTOR(ADD_MD, this->cpu_engine_, add_buf);
        auto fuse_add_dst_ =
            MEMORY_CONSTRUCTOR(DST_MD, this->cpu_engine_, dst_buf);
        auto reorder_desc =
            REORDER_PD_CONSTRUCTOR(ADD_MD, DST_MD, this->cpu_engine_);

        CreateAndExecuteReorder(reorder_desc, fuse_add_src_, fuse_add_dst_,
                                this->cpu_engine_, ctx);
      }
    } else {
      ctx->allocate_output(kOutputIndex_Dst, output_tf_shape, &dst_tensor);
    }

    // if there's nothing to compute, just return.
    if (batch == 0 || channel == 0) {
      return;
    }

    try {
      // Prepare the input and output for primitive.
      T* src_data = const_cast<T*>(src_tensor.flat<T>().data());
      T* weight_data = const_cast<T*>(weight_tensor.flat<T>().data());
      T* bias_data = const_cast<T*>(bias_tensor.flat<T>().data());
      T* dst_data = const_cast<T*>(dst_tensor->flat<T>().data());

      // Reorder input if necessary.
      MklDnnData<T> src_mkl(&(this->cpu_engine_));
      MklDnnData<T> weight_mkl(&(this->cpu_engine_));

      auto src_md = memory::desc(src_dims, MklDnnType<T>(), src_format);

      if (IS_SRC_REORDER_NEEDED(src_md, matmul_pd, matmul_prim)) {
        src_mkl.SetUsrMem(src_md, src_data);
        src_mkl.CheckReorderToOpMem(
            MEMORY_PD_WITHOUT_DATA(matmul_pd.get()->PRIMITIVE_DESC_SRC,
                                   this->cpu_engine_),
            ctx);
        src_data = reinterpret_cast<T*>(src_mkl.GetOpMem().get_data_handle());
      }

      // Get cached data when weight is const.
      const memory::desc weight_md =
          memory::desc(weight_dims, MklDnnType<T>(), weight_format);
      if (IS_WEIGHTS_REORDER_NEEDED(weight_md, matmul_pd, matmul_prim)) {
        T* cached_weight_data = nullptr;

        if (this->is_weight_const_) {
          if (this->IsWeightCacheEmpty(ctx)) {
            this->CacheWeight(ctx, matmul_pd, cached_weight_data, weight_tensor,
                              weight_mkl, weight_md);
          }
          cached_weight_data = this->GetCachedWeight(
              ctx, GET_WEIGHTS_DESC_FROM_OP_PD(matmul_pd));
        }

        // Cache weight may fail when it gets different format in different
        // iteration. Fallback to reoder if it happens.
        // Also do generel reorder if weight isn't const.
        if (cached_weight_data != nullptr) {
          weight_data = cached_weight_data;
        } else {
          weight_mkl.SetUsrMem(weight_md, weight_data);
          weight_mkl.CheckReorderToOpMem(
              MEMORY_PD_WITHOUT_DATA(matmul_pd.get()->PRIMITIVE_DESC_WEIGHTS,
                                     this->cpu_engine_),
              ctx);
          weight_data =
              reinterpret_cast<T*>(weight_mkl.GetOpMem().get_data_handle());
        }
      }
      std::shared_ptr<stream> cpu_stream;
      if (ExecuteSingleThreadedGemm(batch, k, channel)) {
        MklDnnThreadPool eigen_tp(ctx, 1);
        cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
        // Execute fused matmul op.
        matmul_prim->Execute(src_data, weight_data, bias_data, dst_data,
                             cpu_stream);
      } else {
        MklDnnThreadPool eigen_tp(ctx);
        cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
        // Execute fused matmul op.
        matmul_prim->Execute(src_data, weight_data, bias_data, dst_data,
                             cpu_stream);
      }
      if (do_not_cache)
        delete matmul_prim;
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void ExtendMklDnnMatMulFwdParams(OpKernelContext* ctx,
                                   MklDnnMatMulFwdParams& params) {
    if (fused_ops_.size() == 2) {
      string post_op = fused_ops_[1];

      if (post_op == "Relu") {
        params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
      } else if (post_op == "Relu6") {
        params.post_op_params.push_back({"relu6", {1.0, 6.0, 0.0}});
      } else if (post_op == "Elu") {
        params.post_op_params.push_back({"elu", {1.0, 1.0, 0.0}});
      } else if (post_op == "Tanh") {
        params.post_op_params.push_back({"tanh", {1.0, 0.0, 0.0}});
      } else if (post_op == "Gelu") {
        params.post_op_params.push_back({"gelu", {1.0, 0.0, 0.0}});
      } else if (post_op == "Gelu_erf") {
        params.post_op_params.push_back({"gelu_erf", {1.0, 0.0, 0.0}});
      } else if (post_op == "Add") {
        params.post_op_params.push_back({"sum", {1.0}});
      } else {
        OP_REQUIRES_OK(
            ctx, errors::InvalidArgument(
                     "Unsupported post-argument in MklFusedMatMul: ", post_op));
      }
    }
  }

 private:
  bool fuse_add_ = false;
  bool transpose_a_;
  bool transpose_b_;
  std::vector<string> fused_ops_;
  const int kInputIndex_Add = 3;
  const int kOutputIndex_Dst = 0;
};

// Register OneDNN kernels for supported operations and types.
#define REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES(type) \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedMatMul")                                  \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),      \
      MklFusedMatMulOp<CPUDevice, type>);
TF_CALL_float(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);

template <typename Device, typename T>
class MklFusedMatMulGradOp : public OpKernel {
 public:
  explicit MklFusedMatMulGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

    OP_REQUIRES(ctx, fused_ops_.size() == 1,
                errors::InvalidArgument(
                    "MklFusedMatMul must have 1 post-arguments at most."));
    OP_REQUIRES(
        ctx, fused_ops_[0] == "BiasAddGrad",
        errors::InvalidArgument(
            "The 1st post-argument of MklFusedMatMul must be BiasAddGrad."));
  }
  void Compute(OpKernelContext* ctx) {
    try {
      const size_t diff_dst_index = 1;  // index of diff_dst input tensor
      const size_t src_index = 0;       // index of src input tensor

      const Tensor& src_tensor = MklGetInput(ctx, src_index);
      const Tensor& diff_dst_tensor = MklGetInput(ctx, diff_dst_index);

      MklDnnShape src_mkl_shape;
      MklDnnShape diff_dst_mkl_shape;
      GetMklShape(ctx, src_index, &src_mkl_shape);
      GetMklShape(ctx, diff_dst_index, &diff_dst_mkl_shape);
      auto src_tf_shape = src_mkl_shape.IsMklTensor()
                              ? src_mkl_shape.GetTfShape()
                              : src_tensor.shape();
      auto diff_dst_tf_shape = diff_dst_mkl_shape.IsMklTensor()
                                   ? diff_dst_mkl_shape.GetTfShape()
                                   : diff_dst_tensor.shape();

      const int dim_pair[] = {transpose_a_ ? 0 : 1, transpose_a_ ? 1 : 0};
      const int batch = src_tf_shape.dim_size(1 - dim_pair[0]);
      const int k = src_tf_shape.dim_size(dim_pair[0]);
      const int channel = diff_dst_tf_shape.dim_size(1);

      OP_REQUIRES(
          ctx, batch == diff_dst_tf_shape.dim_size(0),
          errors::InvalidArgument(
              "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
              ", In[1]: ", diff_dst_tf_shape.DebugString()));

      Tensor* diff_weight_tensor = nullptr;
      Tensor* bias_tensor = nullptr;

      if (batch == 0 || channel == 0) {
        TensorShape diff_weight_tensor_shape({k, channel});
        if (transpose_b_) diff_weight_tensor_shape = {channel, k};
        MklDnnShape diff_weight_mkl_shape;
        diff_weight_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(ctx, 0, &diff_weight_tensor,
                                  diff_weight_tensor_shape,
                                  diff_weight_mkl_shape);

        TensorShape bias_tensor_shape({channel});
        MklDnnShape bias_mkl_shape;
        bias_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(ctx, 1, &bias_tensor, bias_tensor_shape,
                                  bias_mkl_shape);

        return;
      }

      memory::dims src_dims = memory::dims({batch, k});
      memory::dims diff_dst_dims = memory::dims({batch, channel});
      memory::dims diff_weight_dims = memory::dims({channel, k});
      memory::dims diff_bias_dims = memory::dims({channel});
      MEMORY_FORMAT src_format =
          transpose_a_ ? MEMORY_FORMAT::cn : MEMORY_FORMAT::nc;

      MEMORY_FORMAT diff_dst_format = MEMORY_FORMAT::nc;

      MEMORY_FORMAT diff_weight_format =
          transpose_b_ ? MEMORY_FORMAT::oi : MEMORY_FORMAT::io;

      MklDnnMatMulBwdFilterParams matmul_params(
          src_dims, diff_weight_dims, diff_bias_dims, diff_dst_dims,
          MEMORY_FORMAT::nc, diff_weight_format, MEMORY_FORMAT::nc);

      MklDnnMatMulBwdFilterPrimitive<T>* matmul_prim =
          MklDnnMatMulBwdFilterPrimitiveFactory<T>::Get(matmul_params);

      std::shared_ptr<dnnl::inner_product_backward_weights::primitive_desc>
          matmul_pd = matmul_prim->GetPrimitiveDesc();

      // Has two outputs, 0 for MatMulGradFilter, 1 for BiasAddGrad
      if (src_mkl_shape.IsMklTensor()) {
        memory::desc diff_weight_pd = matmul_pd->diff_weights_desc();
        AllocateOutputTensor(ctx, diff_weight_pd, diff_weight_dims,
                             MKL_TENSOR_FORMAT_NC, &diff_weight_tensor, 0);
        memory::desc bias_pd = matmul_pd->diff_bias_desc();
        AllocateOutputTensor(ctx, bias_pd, diff_bias_dims, MKL_TENSOR_FORMAT_X,
                             &bias_tensor, 1);
      } else {
        TensorShape diff_weight_tensor_shape({k, channel});
        if (transpose_b_) diff_weight_tensor_shape = {channel, k};
        MklDnnShape diff_weight_mkl_shape;
        diff_weight_mkl_shape.SetMklTensor(false);
        diff_weight_mkl_shape.SetElemType(MklDnnType<T>());
        AllocateOutputSetMklShape(ctx, 0, &diff_weight_tensor,
                                  diff_weight_tensor_shape,
                                  diff_weight_mkl_shape);

        TensorShape bias_tensor_shape({channel});
        MklDnnShape bias_mkl_shape;
        bias_mkl_shape.SetMklTensor(false);
        bias_mkl_shape.SetElemType(MklDnnType<T>());
        AllocateOutputSetMklShape(ctx, 1, &bias_tensor, bias_tensor_shape,
                                  bias_mkl_shape);
      }

      T* src_data = const_cast<T*>(src_tensor.flat<T>().data());
      T* diff_dst_data = const_cast<T*>(diff_dst_tensor.flat<T>().data());
      T* bias_data = const_cast<T*>(bias_tensor->flat<T>().data());
      T* diff_weight_data =
          const_cast<T*>(diff_weight_tensor->flat<T>().data());

      MklDnnData<T> src_mkl(&cpu_engine_);
      MklDnnData<T> diff_dst_mkl(&cpu_engine_);
      MklDnnData<T> diff_weight_mkl(&cpu_engine_);

      auto src_md = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<T>(), src_format);

      auto diff_dst_md =
          diff_dst_mkl_shape.IsMklTensor()
              ? diff_dst_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(), diff_dst_format);

      auto diff_weight_md =
          memory::desc(diff_weight_dims, MklDnnType<T>(), diff_weight_format);

      if (IS_SRC_REORDER_NEEDED(src_md, matmul_pd, matmul_prim)) {
        src_mkl.SetUsrMem(src_md, src_data);
        src_mkl.CheckReorderToOpMem(MEMORY_PD_WITHOUT_DATA(
            matmul_pd.get()->PRIMITIVE_DESC_SRC, cpu_engine_));
        src_data = reinterpret_cast<T*>(src_mkl.GetOpMem().get_data_handle());
      }

      if (diff_dst_md != matmul_pd->diff_dst_desc()) {
        diff_dst_mkl.SetUsrMem(diff_dst_md, diff_dst_data);
        diff_dst_mkl.CheckReorderToOpMem(
            MEMORY_PD_WITHOUT_DATA(matmul_pd->diff_dst_desc(), cpu_engine_));
        diff_dst_data =
            reinterpret_cast<T*>(diff_dst_mkl.GetOpMem().get_data_handle());
      }
      // Execute fused matmul op.
      matmul_prim->Execute(src_data, diff_weight_data, bias_data,
                           diff_dst_data);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  void AllocateOutputTensor(OpKernelContext* context, memory::desc& dst_pd,
                            const memory::dims& output_dims_mkl_order,
                            MKL_TENSOR_FORMAT output_tf_format,
                            Tensor** output_tensor, int idx) {
    DCHECK(output_tensor);

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    output_tf_shape.AddDim((dst_pd.get_size() / sizeof(T)));

    // Allocate Output Tensor
    AllocateOutputSetMklShape(context, idx, output_tensor, output_tf_shape,
                              output_mkl_shape);
  }
  bool transpose_a_;
  bool transpose_b_;
  std::vector<string> fused_ops_;
  engine cpu_engine_ = engine(ENGINE_CPU, 0);
};

#define REGISTER_FUSEDMATMUL_GRAD_TYPES(type)                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_MklFusedMatMulGrad")                                            \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<type>("T")                                         \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),               \
      MklFusedMatMulGradOp<CPUDevice, type>);                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_FusedMatMulGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      NoOp);

TF_CALL_float(REGISTER_FUSEDMATMUL_GRAD_TYPES);
TF_CALL_bfloat16(REGISTER_FUSEDMATMUL_GRAD_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
