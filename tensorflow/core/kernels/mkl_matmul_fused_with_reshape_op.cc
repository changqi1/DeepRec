#ifdef INTEL_MKL

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl_matmul_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// Fuse Operation
template <typename Device, typename T>
class MklFusedMatMulWithReshape : public MklDnnMatMulOpBase<T, T> {
 public:
  explicit MklFusedMatMulWithReshape(OpKernelConstruction* ctx)
      : MklDnnMatMulOpBase<T, T>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

    OP_REQUIRES(
        ctx, fused_ops_.size() <= 3,
        errors::InvalidArgument(
            "MklFusedMatMulWithReshape must have 3 post-arguments at most."));
    OP_REQUIRES(
        ctx, fused_ops_[0] == "Reshape",
        errors::InvalidArgument("The 1st post-argument of "
                                "MklFusedMatMulWithReshape must be Reshape."));
    OP_REQUIRES(
        ctx, fused_ops_[1] == "BiasAdd",
        errors::InvalidArgument("The 2st post-argument of "
                                "MklFusedMatMulWithReshape must be BiasAdd."));
    // TODO: Support Add fusion after BiasAdd
    // if (fused_ops_.size() > 2 && fused_ops_[2] == "Add") fuse_add_ = true;
    // TODO: Add other op support
    if (fused_ops_.size() > 2) {
      OP_REQUIRES(ctx, fused_ops_[2] == "Relu",
                  errors::InvalidArgument(
                      "The 3st post-argument of MklFusedMatMulWithReshape only "
                      "support Relu now."));
    }
    OP_REQUIRES(
        ctx, transpose_a_ == false,
        errors::InvalidArgument("In[0] of MklMatMul can't be transposed."));
  }

  void Compute(OpKernelContext* ctx) override {
    // FusedMatMulwithreshape has 4 inputs: src, weights, bias, reshape size
    const Tensor& src_tensor = ctx->input(this->kInputIndexSrc);
    const Tensor& weight_tensor = ctx->input(this->kInputIndexWeight);
    const Tensor& bias_tensor = MklGetInput(ctx, this->kInputIndexBias);
    const Tensor& reshape_to_size_tensor = ctx->input(this->kInputIndexReshape);

    // Get shapes of input tensors
    auto src_tf_shape = src_tensor.shape();
    auto weight_tf_shape = weight_tensor.shape();
    auto reshape_to_size_tf_shape = reshape_to_size_tensor.shape();
    auto bias_tf_shape = bias_tensor.shape();

    // Check the constraint of input matrix and bias
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(src_tf_shape),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weight_tf_shape),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias_tf_shape),
                errors::InvalidArgument("Biases must be 1D"));

    // Vaild reshape size
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(reshape_to_size_tf_shape),
                errors::InvalidArgument("Reshape size must be 1D"));

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

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    switch (reshape_to_size_tensor.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(ctx, ValidateSizes<int32>(reshape_to_size_tensor,
                                                 &product, &unknown_index,
                                                 &shape));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(ctx, ValidateSizes<int64>(reshape_to_size_tensor,
                                                 &product, &unknown_index,
                                                 &shape));
        break;
      default:
        ctx->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(reshape_to_size_tensor.dtype())));
        return;
    }
    if (unknown_index != -1) {
      OP_REQUIRES(
          ctx, unknown_index == shape.dims() - 1,
          errors::InvalidArgument("Last dim of Reshape size can't be unknow."));
      shape.set_dim(unknown_index, channel);
    } else {
      OP_REQUIRES(
          ctx, shape.dim_size(shape.dims() - 1) == channel,
          errors::InvalidArgument("Last dim of Reshape size must be equal to bias size."));
    }

    OP_REQUIRES(ctx, shape.num_elements() == batch * channel,
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        batch * channel,
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

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

    // Set to reshape_to_size
    // TensorShape output_tf_shape(reshape_to_size_tf_shape.dim_sizes().size());
    TensorShape output_tf_shape(shape);

    if (fuse_add_) {
      // TODO
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
        auto add_md =
            memory::desc(dst_dims, MklDnnType<T>(), output_format_tag);
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
      if (do_not_cache) delete matmul_prim;
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void ExtendMklDnnMatMulFwdParams(OpKernelContext* ctx,
                                   MklDnnMatMulFwdParams& params) {
    if (fused_ops_.size() == 3) {
      string post_op = fused_ops_[2];

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
  const int kInputIndexReshape = 2;
  const int kInputIndexBias = 3;
  const int kInputIndex_Add = 4;
  const int kOutputIndex_Dst = 0;

  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64* product, int* unknown_index,
                       TensorShape* shape) {
    *product = 1;
    *unknown_index = -1;
    const int64 num_dims = sizes.NumElements();
    auto Svec = sizes.flat<Tshape>();
    for (int d = 0; d < num_dims; ++d) {
      const Tshape size = Svec(d);
      if (size == -1) {
        if (*unknown_index != -1) {
          return errors::InvalidArgument(
              "Only one input size may be -1, not both ", *unknown_index,
              " and ", d);
        }
        *unknown_index = d;
        shape->AddDim(1);
      } else if (size < 0) {
        return errors::InvalidArgument("Size ", d,
                                       " must be non-negative, not ", size);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape->AddDim(size);
      } else {
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return Status::OK();
  }
};

// TODO: kMklLayoutDependentOpLabel or kMklNameChangeOpLabel
// Register OneDNN kernels for supported operations and types.
#define REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES(type)                \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedMatMulWithReshape")                  \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint("Tshape", {DT_INT32, DT_INT64}) \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklFusedMatMulWithReshape<CPUDevice, type>);
TF_CALL_float(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
