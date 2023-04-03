/* Copyright 2022 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/util/work_sharder.h"
namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace {

const char* kInferenceMode = "INFERENCE_MODE";

enum Combiner { Mean, Sum, Sqrtn };

template <Combiner combiner>
inline float DoCombiner(float in, int feature_num);

template <>
inline float DoCombiner<Mean>(float in, int feature_num) {
  return in / feature_num;
}
template <>
inline float DoCombiner<Sum>(float in, int feature_num) {
  return in;
}
template <>
inline float DoCombiner<Sqrtn>(float in, int feature_num) {
  return in / sqrtf(feature_num);
}

template <typename TKey, typename TValue, Combiner combiner>
void sparse_feature_combiner_gather(const TValue* embedding_weights,
                                    const TKey* ids, const int64* sp_indices,
                                    const int nnz, const int dimension,
                                    TValue* outputs) {
  int batch_id = 0;
  int batch_num = 0;
  std::vector<TValue> tmp_embedding;
  tmp_embedding.resize(dimension);

  for (int i = 0; i < nnz; ++i) {
    int new_batch_id = sp_indices[i];
    if (new_batch_id != batch_id) {
      if (batch_num > 0) {
        // apply combiner
        for (int d = 0; d < dimension; ++d) {
          outputs[batch_id * dimension + d] =
              DoCombiner<combiner>(tmp_embedding[d], batch_num);
        }
      }
      // memcpy(outputs+(i-1)*dimension, tmp_embedding.data(),
      // sizeof(TValue)*embedding);
      batch_num = 0;
      batch_id = new_batch_id;
      memset(tmp_embedding.data(), 0, sizeof(TValue)*dimension);
    }
    TKey id = ids[i];
    for (int d = 0; d < dimension; ++d) {
      tmp_embedding[d] += embedding_weights[id * dimension + d];
    }
    ++batch_num;
  }
  //Process final batch_id
  for (int d = 0; d < dimension; ++d) {
      outputs[batch_id * dimension + d] = DoCombiner<combiner>(tmp_embedding[d], batch_num);
  }
}

template <typename TKey, typename TValue, Combiner combiner>
void embedding_var_combiner(const TValue* embedding_tensors,
                            const int64* sp_indices, const int nnz,
                            const int dimension, TValue* outputs) {
  int batch_id = 0;
  int batch_num = 0;
  std::vector<TValue> tmp_embedding;
  tmp_embedding.resize(dimension);

  for (int i = 0; i < nnz; ++i) {
    int new_batch_id = sp_indices[i];
    if (new_batch_id != batch_id) {
      if (batch_num > 0) {
        for (int d = 0; d < dimension; ++d) {
          outputs[batch_id * dimension + d] =
              DoCombiner<combiner>(tmp_embedding[d], batch_num);
        }
      }
      batch_num = 0;
      batch_id = new_batch_id;
      memset(tmp_embedding.data(), 0, sizeof(TValue)*dimension);
    }
    for (int d = 0; d < dimension; ++d) {
      tmp_embedding[d] += embedding_tensors[i * dimension + d];
    }
    ++batch_num;
  }
  //Process final batch_id
  for (int d = 0; d < dimension; ++d) {
    outputs[batch_id * dimension + d] = DoCombiner<combiner>(tmp_embedding[d], batch_num);
  }
}

}  // namespace

template <typename TKey, typename TValue>
class GroupEmbeddingVariableLookupCpuOp : public OpKernel {
 public:
  explicit GroupEmbeddingVariableLookupCpuOp(OpKernelConstruction* c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &is_use_default_value_tensor_));
    OP_REQUIRES_OK(c, c->GetAttr("is_inference", &is_inference_));
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));
    is_inference_ |= is_inference;
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                             int64 total_dim,
                             int64 len) { return default_v + len * index; };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                             int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim);
      };
    }
    //TODO: FIXME
    if (c->num_inputs() == 4) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) { return 1; };
    }
    if (!is_inference_) {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                      TValue* default_v, int count) {
        ev->LookupOrCreate(key, val, default_v, count);
        return Status::OK();
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                      TValue* default_v, int count) {
        Status s = ev->Lookup(key, val, default_v);
        return s;
      };
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dense_shape_tensor = ctx->input(num_lookups_ * 4);
    auto dense_shape = dense_shape_tensor.flat<int>().data();
    int batch_size = dense_shape[0];

    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();

    for (int i = 0; i < num_lookups_; ++i) {
      EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, i), &embedding_var));
      core::ScopedUnref unref_me(embedding_var);

      const Tensor& sp_values_tensor = ctx->input(num_lookups_ + i);
      auto sp_values = sp_values_tensor.flat<TKey>().data();
      const Tensor& sp_indices_tensor = ctx->input(num_lookups_ * 2 + i);
      int nnz = sp_indices_tensor.shape().dim_size(0);

      TensorShape temp_emb_vectors_tensor_shape = TensorShape(
          std::vector<int64>({static_cast<long long>(nnz), dimension_}));
      Tensor temp_emb_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TValue>::value,
                                             temp_emb_vectors_tensor_shape,
                                             &temp_emb_tensor));
      TValue* temp_out = temp_emb_tensor.flat<TValue>().data();

      int32* counts = nullptr;
      if (nnz > 0) {
        TValue* default_v = nullptr;
        if (is_use_default_value_tensor_) {
          default_v = reinterpret_cast<TValue*>(
              ctx->input(num_lookups_ * 4 + 1).data());
        } else {
          default_v = embedding_var->GetDefaultValuePtr();
        }
        OP_REQUIRES(
            ctx,
            !embedding_var->IsMultiLevel() ||
                (embedding_var->IsMultiLevel() &&
                 embedding_var->CacheSize() >= nnz),
            errors::InvalidArgument("MultiLevel EV's Cache size ",
                                    embedding_var->CacheSize(),
                                    " should large than IDs in batch ", nnz));
        const size_t slice_bytes = nnz * sizeof(TValue);
        auto do_compute = [this, ctx, default_v, sp_values, embedding_var,
                           temp_out, counts](int start, int end) {
          for (int j = start; j < end; ++j) {
            TValue* default_v_ptr = get_default_v_fn_(
                default_v, sp_values[j], j, embedding_var->GetDefaultValueDim(),
                embedding_var->ValueLen());
            int32 count = get_count_fn_(counts, j);
            OP_REQUIRES_OK(ctx, lookup_fn_(embedding_var, sp_values[j],
                                           temp_out + j * dimension_,
                                           default_v_ptr, count));
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers,
              nnz, slice_bytes /*cost*/, do_compute);
        
        if (embedding_var->IsMultiLevel()) {
            embedding::BatchCache<TKey>* cache = embedding_var->Cache();
            embedding_var->storage_manager()->Schedule([embedding_var, sp_values_tensor] {
                embedding::BatchCache<TKey>* cache = embedding_var->Cache();
                cache->add_to_rank(sp_values_tensor);
            });
        }
        TensorShape emb_vectors_tensor_shape = TensorShape(
          std::vector<int64>({static_cast<long long>(batch_size), dimension_}));
        Tensor* emb_vectors_tensor = nullptr;
        // allocate output
        OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                                &emb_vectors_tensor));
        auto emb_vectors = emb_vectors_tensor->flat<TValue>().data();
        // allocate offset tensor
        TensorShape values_offset_tensor_shape =
            TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));
        Tensor* values_offset_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(num_lookups_ + i,
                                                values_offset_tensor_shape,
                                                &values_offset_tensor));
        auto values_offset = values_offset_tensor->flat<int>().data();

        if (combiner_ == "mean") {
          embedding_var_combiner<TKey, TValue, Mean>(
              temp_out, sp_indices_tensor.flat<int64>().data(), nnz, dimension_,
              emb_vectors);
        } else if (combiner_ == "sum") {
          embedding_var_combiner<TKey, TValue, Sum>(
              temp_out, sp_indices_tensor.flat<int64>().data(), nnz, dimension_,
              emb_vectors);
        } else {
          embedding_var_combiner<TKey, TValue, Sqrtn>(
              temp_out, sp_indices_tensor.flat<int64>().data(), nnz, dimension_,
              emb_vectors);
        }
      }
    }
  }

 private:
  std::function<TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
  std::function<int32(int32*, int64)> get_count_fn_;
  std::function<Status(EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                       TValue* default_v, int count)>
      lookup_fn_;
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
  bool is_use_default_value_tensor_;
  bool is_inference_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(                         \
      Name("GroupEmbeddingVarLookup")         \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<key_type>("Tkeys")       \
          .TypeConstraint<value_type>("dtype"),    \
      GroupEmbeddingVariableLookupCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template <typename TKey, typename TValue>
class GroupVariableLookupCpuOp : public OpKernel {
 public:
  explicit GroupVariableLookupCpuOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dense_shape_tensor = ctx->input(num_lookups_ * 4);
    auto dense_shape = dense_shape_tensor.flat<int>().data();
    int batch_size = dense_shape[0];
    auto do_compute = [this, ctx, batch_size](int start, int end) {
      for (int i = start; i < end; ++i) {
        const Tensor& emb_variable_tensor = ctx->input(i);
        const Tensor& sp_values_tensor = ctx->input(num_lookups_ + i);
        int64 emb_vec_size = emb_variable_tensor.shape().dim_size(1);

        const Tensor& sp_indices_tensor = ctx->input(num_lookups_ * 2 + i);
        int nnz = sp_indices_tensor.shape().dim_size(0);

        TensorShape emb_vectors_tensor_shape = TensorShape(std::vector<int64>(
            {static_cast<long long>(batch_size), emb_vec_size}));
        Tensor* emb_vectors_tensor = nullptr;
        // allocate output
        OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                                 &emb_vectors_tensor));
        auto emb_vectors = emb_vectors_tensor->flat<TValue>().data();

        // allocate offset tensor
        TensorShape values_offset_tensor_shape = TensorShape(
            std::vector<int64>({static_cast<long long>(batch_size)}));
        Tensor* values_offset_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(num_lookups_ + i,
                                                 values_offset_tensor_shape,
                                                 &values_offset_tensor));
        auto values_offset = values_offset_tensor->flat<int>().data();
        if (combiner_ == "mean") {
          sparse_feature_combiner_gather<TKey, TValue, Mean>(
              emb_variable_tensor.flat<TValue>().data(),
              sp_values_tensor.flat<TKey>().data(),
              sp_indices_tensor.flat<int64>().data(), nnz, dimension_,
              emb_vectors);
        } else if (combiner_ == "sum") {
          sparse_feature_combiner_gather<TKey, TValue, Sum>(
              emb_variable_tensor.flat<TValue>().data(),
              sp_values_tensor.flat<TKey>().data(),
              sp_indices_tensor.flat<int64>().data(), nnz, dimension_,
              emb_vectors);
        } else {
          sparse_feature_combiner_gather<TKey, TValue, Sqrtn>(
              emb_variable_tensor.flat<TValue>().data(),
              sp_values_tensor.flat<TKey>().data(),
              sp_indices_tensor.flat<int64>().data(), nnz, dimension_,
              emb_vectors);
        }
      }
    };
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads, worker_threads->workers, 
          num_lookups_, 10000 /*cost*/, do_compute);
  }

 private:
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name("GroupVariableLookup")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<key_type>("Tkeys")    \
                              .TypeConstraint<value_type>("dtype"), \
                          GroupVariableLookupCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

}  // namespace tensorflow