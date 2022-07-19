#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/variable_ops.h"

using namespace tensorflow;

// input: input tensor value (it sores the id)
// cols: How many elements to do SparseSegmentSum
// output: rows * embedding_size
template<typename T>
static void sparse_gather_v1(T *input, int rows, int cols, float *embedding_table, float *output, int embedding_size, bool is_mean) {
  T *pidx = input;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < embedding_size; ++j) {
      float value = 0;
      int dense_num = 0;
      for (int k = 0; k < cols; ++k) {
        int embedding_row = (int)pidx[k];
        if (embedding_row >= 0) {
          value += embedding_table[embedding_row * embedding_size + j];
          dense_num += 1;
        }
      }

      if (is_mean && dense_num > 0) {
        *output++ = value / dense_num;
      } else {
        *output++ = value;
      }
    }
    pidx += cols;
  }
}

// embedding_size = 1
template<typename T>
static void sparse_gather_embeddingsize1(T *input, int rows, int cols, float *embedding_table, float *output, bool is_mean) {
  T *pidx = input;
  for (int i = 0; i < rows; ++i) {
    float value = 0;
    int dense_num = 0;
    for (int k = 0; k < cols; ++k) {
      int embedding_row = pidx[k];
      if (embedding_row >= 0) {
        value += embedding_table[embedding_row];
        dense_num += 1;
      }
    }
    if (is_mean && dense_num > 0) {
      *output++ = value / dense_num;
    } else {
      *output++ = value;
    }
    pidx += cols;
  }
}

// input cols = 1
template<typename T>
static void sparse_gather_column1(T *input, int rows, float *embedding_table, float *output, int embedding_size) {
  T *pidx = input;
  for (int i = 0; i < rows; ++i) {
    int embedding_row = *pidx++;
    if (embedding_row >= 0) {
      float *pembedding = &embedding_table[embedding_row * embedding_size];
      for (int j = 0; j < embedding_size; ++j) {
        output[j] = pembedding[j];
      }
    } else {
      for (int j = 0; j < embedding_size; ++j) {
        output[j] = 0;
      }
    }
    output += embedding_size;
  }
}

template<typename T>
static void sparse_gather(T *input, int rows, int cols, float *embedding_table, float *output, int embedding_size, bool is_mean) {
  if (embedding_size == 1) {
    sparse_gather_embeddingsize1(input, rows, cols, embedding_table, output, is_mean);
  } else if (cols == 1) {
    sparse_gather_column1(input, rows, embedding_table, output, embedding_size);
  } else {
    //printf("General sparse gather!\n");
    sparse_gather_v1(input, rows, cols, embedding_table, output, embedding_size, is_mean);
  }
}

// Use memcpy or manually assign?
static void mycopy(float *dst, float *src, int float_num) {
  memcpy(dst, src, float_num * sizeof(float));
}

static void myadd(float *dst, float *src, int float_num) {
#ifdef __AVX512F__
  int i = 0;
  for (; i + 15 < float_num; i += 16) {
    __m512 v1 = _mm512_loadu_ps(&dst[i]);
    __m512 v2 = _mm512_loadu_ps(&src[i]);
    __m512 result = _mm512_add_ps(v1, v2);
    _mm512_storeu_ps(&dst[i], result);
  }
  for (; i < float_num; ++i) {
    dst[i] += src[i];
  }
#else
  for (int i = 0; i < float_num; ++i) {
    dst[i] += src[i];
  }
#endif
}

static void myscale(float *dst, float factor, int float_num) {

#ifdef __AVX512F__
  __m512 vf = _mm512_set1_ps(factor);
  int i = 0;
  for (; i + 15 < float_num; i += 16) {
    __m512 result = _mm512_mul_ps(_mm512_loadu_ps(&dst[i]), vf);
    _mm512_storeu_ps(&dst[i], result);
  }
  for (; i < float_num; ++i) {
    dst[i] *= factor;
  }
#else
  for (int i = 0; i < float_num; ++i) {
    dst[i] *= factor;
  }
#endif
}

template<typename Tid, typename Tshape>
static void sparse_gather(Tid *input, int64 input_size, Tshape *indice, int indice_dim, Tshape *shape, int rows, int cols, float *embedding_table, float *output, int embedding_size, bool is_mean) {
  // Record how many values in each row
  int *row_values = new int[rows];
  memset(row_values, 0, rows * sizeof(int));

  for (int64 i = 0; i < input_size; ++i) {
    Tid id = input[i];
    if (id < 0) { // Skip invalid id
      continue;
    }
    auto row = indice[i * indice_dim];
    for (int k = 1; k < indice_dim - 1; ++k) {
      row = row * shape[k] + indice[i * indice_dim + k];
    }
    if (row_values[row] > 0) {
      myadd(&output[row * embedding_size], &embedding_table[id * embedding_size], embedding_size);
    } else {
      mycopy(&output[row * embedding_size], &embedding_table[id * embedding_size], embedding_size);
    }
    row_values[row] += 1;
  }

  for (int i = 0; i < rows; ++i) {
    if (row_values[i] == 0) {
      memset(&output[i * embedding_size], 0, embedding_size * sizeof(float));
    } else if (is_mean && row_values[i] > 1) {
      float factor = 1.0f / row_values[i];
      myscale(&output[i * embedding_size], factor, embedding_size);
    }
  }

  delete[] row_values;
}

REGISTER_OP("SparseEmbeddingWithShape")
    .Input("weight: Tweight")
    .Input("input: Tid")
    .Input("dense_shape: Tshape")
    .Input("indice: Tshape")
    .Output("embedded: float")
    .Attr("Combiner: int")
    .Attr("Tid: {int64, int32}")
    .Attr("Tshape: {int64, int32}")
    .Attr("Tweight: {float, resource}");
    /*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      DataType datatype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tweight", &datatype));
      if (datatype == DT_RESOURCE) {
        // TODO: Get the real shape
        c->set_output(0, c->Matrix(c->UnknownDim(), c->UnknownDim()));
      } else {
        ::tensorflow::shape_inference::ShapeHandle weight;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &weight));
        c->set_output(0, c->Matrix(c->UnknownDim(),
                                   c->Dim(c->input(0), 1)));
      }
      return Status::OK();
    });*/


template<typename Tid, typename Tshape>
class SparseEmbeddingWithShapeOp : public OpKernel {
public:
  explicit SparseEmbeddingWithShapeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("Combiner", &combiner));
    //OP_REQUIRES_OK(context, context->GetAttr("Dims", &dims));
    node_name = context->def().name();

    static bool printed = false;
    if (!printed) {
      printf("******** SparseEmbeddingWithShapeOp ********\n");
      printed = true;
    }
  }

  ~SparseEmbeddingWithShapeOp() {
  }

  void Compute(OpKernelContext* context) override {
    int64 input_size = 1;
    int embedding_size;

    // Grab the weight
    float *weight;
    const Tensor* weight_tensor = &context->input(0);
    if (weight_tensor->dtype() == DT_RESOURCE) { // For resource, need to lookup the weights
      Var* variable;
      OP_REQUIRES_OK(context,
                     LookupResource(context, HandleFromInput(context, 0), &variable));
      core::ScopedUnref s(variable);
      OP_REQUIRES(context, variable->tensor()->dtype() == DT_FLOAT,
                  errors::InvalidArgument("Expect float weight in ", node_name));
      embedding_size = variable->tensor()->dim_size(1);
      weight = (float *)variable->tensor()->tensor_data().data();
    } else {
      embedding_size = weight_tensor->dim_size(1);
      weight = (float *)weight_tensor->tensor_data().data();
    }
    
    // Input id
    const Tensor& input_tensor = context->input(1);
    Tid *input = (Tid *)input_tensor.tensor_data().data();
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_size *= input_tensor.dim_size(i);
    }

    const Tensor& shape_tensor = context->input(2);
    Tshape *shape = (Tshape *)shape_tensor.tensor_data().data();

    // To check the input
    OP_REQUIRES(context, (shape_tensor.dims() == 1),
                errors::InvalidArgument("Shape tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (shape_tensor.dim_size(0) >= 2),
                errors::InvalidArgument("Shape tensor is not valid (dim_size(0) < 2)"));

    int input_dims = shape_tensor.dim_size(0);
    int cols = shape[input_dims - 1];
    int batch_size = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
      batch_size *= shape[i];
    }
    bool is_mean = (combiner == 1);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape({batch_size, embedding_size});
    /* Testing shows following impl. is not correct
    if (input_dims == 2) {
      output_shape = TensorShape({batch_size, embedding_size});
    } else {
      std::vector<int> tmp_sizes;
      for (int i = 0; i < input_dims - 1; ++i) {
        tmp_sizes.push_back(shape[i]);
      }
      tmp_sizes.push_back(embedding_size);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(tmp_sizes.data(), tmp_sizes.size(), &output_shape));
    }*/
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    float *output = (float *)output_tensor->tensor_data().data();

    if (input_size == batch_size * cols) { // input id is dense
      sparse_gather(input, batch_size, cols, weight, output, embedding_size, is_mean);
    } else { // input id is sparse
      const Tensor& indice_tensor = context->input(3);
      OP_REQUIRES(context, (indice_tensor.dims() == 2),
                  errors::InvalidArgument("Indice tensor is not as expected (dims != 2)"));
      OP_REQUIRES(context, (indice_tensor.dim_size(0) == input_size),
                  errors::InvalidArgument("Indice tensor is not as expected (dim_size(0) != batch_size)"));
      Tshape *indice = (Tshape *)indice_tensor.tensor_data().data();
      int indice_dim = indice_tensor.dim_size(1);
      sparse_gather(input, input_size, indice, indice_dim, shape, batch_size, cols, weight, output, embedding_size, is_mean);
    }
  }

private:
  // 0=SUM, 1=MEAN
  int combiner;
  std::string node_name;
};



REGISTER_KERNEL_BUILDER(Name("SparseEmbeddingWithShape")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("Tid")
                            .TypeConstraint<int64>("Tshape"),
                        SparseEmbeddingWithShapeOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("SparseEmbeddingWithShape")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Tid")
                            .TypeConstraint<int64>("Tshape"),
                        SparseEmbeddingWithShapeOp<int64, int64>);
