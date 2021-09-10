/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "dnnl.hpp"
#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

#include "tensorflow/core/framework/fake_input.h"

#define printTensor(T, d) \
    std::cout<< (T).tensor<float, (d)>() << std::endl

#define printTensorUInt8(T, d) \
    std::cout<< (T).tensor<uint8, (d)>() << std::endl

namespace tensorflow {

//----------------------------------------------------------------------------//
// MatMul Unit Tests are below.                                               //
//----------------------------------------------------------------------------//

// Helper class for converting MKL tensors to TF tensors and comparing to
// expected values
static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

using GraphRunner =
    std::function<void(const Tensor& input_data, const Tensor& filter_data, Tensor* out, bool transpose_a, bool transpose_b)>;

template <typename T>
class CommonTestUtilities : public OpsTestBase {
 public:
  void PerformConversion(DataType dtype, const Tensor& tensor,
                         const Tensor& mkl_meta_tensor, Tensor* output) { // Default, convert shape
    // Create an MKL to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // Mkl second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(tensor.shape(), tensor.flat<T>());
    AddInputFromArray<uint8>(mkl_meta_tensor.shape(),
                             mkl_meta_tensor.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    *output = *GetOutput(0);
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor.
  static void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                          Tensor* output) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void TestBody() {}

  // Compare two outcomes default & mkl by calling run_default() & run_mkl()
  static void VerifyMKLMatrixClose(int m, int k, int n,
                                     const GraphRunner& run_default,
                                     const GraphRunner& run_mkl,
                                     bool transpose_a, bool transpose_b) { 
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor input(dtype, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
    input.flat<T>() = input.flat<T>().template setRandom<random_gen_>();

    Tensor weight(dtype, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
    weight.flat<T>() = weight.flat<T>().template setRandom<random_gen_>();

    Tensor output;
    Tensor mkl_output;

    run_default(input, weight, &output, transpose_a, transpose_b);
    run_mkl(input, weight, &mkl_output, transpose_a, transpose_b);

    ASSERT_EQ(output.dtype(), mkl_output.dtype());
    ASSERT_EQ(output.shape(), mkl_output.shape());

    test::ExpectClose(output, mkl_output, 1e-5);
  }

 private:
  using random_gen_ = Eigen::internal::NormalRandomGenerator<T>;
};

// Testing MatMul
template <typename T>
class MklMatMulOpTest : public OpsTestBase {
 private:
  void RunMklMatMulOp(const Tensor& input, const Tensor& weight,
                           Tensor* output, bool transpose_a, bool transpose_b) {
    DataType dtype = DataTypeToEnum<T>::v();
    
    TF_EXPECT_OK(
        NodeDefBuilder("mkl_matmul_op", "_MklMatMul") //build node
            .Input(FakeInput(dtype))
            .Input(FakeInput(dtype))
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
            .Attr("_kernel", "MklNameChangeOp")
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp()); //initial
    AddInputFromArray<T>(input.shape(), input.flat<T>()); // A input 
    AddInputFromArray<T>(weight.shape(), weight.flat<T>());
    TF_EXPECT_OK(RunOpKernel()); //Run the node computation
    *output = *GetOutput(0); //Get output
  }

 protected:
  void VerifyMKLMatMul(int m, int k, int n, bool transpose_a, bool transpose_b){
    const GraphRunner run_default =
        [this](const Tensor& input, const Tensor& weight,
              Tensor* output, bool transpose_a, bool transpose_b) {
          auto root = tensorflow::Scope::NewRootScope();
          auto input_op =
              ops::Const(root.WithOpName("input"), Input::Initializer(input));
          Output next_op = ops::MatMul(root.WithOpName("matmul"), input_op,
                                       ops::Const(root.WithOpName("weight"),
                                       Input::Initializer(weight)),
                                       ops::MatMul::TransposeA(transpose_a).TransposeB(transpose_b)
                                       );
          string last_op = "matmul";
          CommonTestUtilities<T>::RunAndFetch(root, last_op, output);
        };

    const GraphRunner run_mkl =
        [this](const Tensor& input, const Tensor& weight,
                Tensor* output, bool transpose_a, bool transpose_b) {
          RunMklMatMulOp(input, weight, output, transpose_a, transpose_b);
        };

    CommonTestUtilities<T>::VerifyMKLMatrixClose(m, k, n,
                                                 run_default, run_mkl,
                                                 transpose_a, transpose_b);
  }
};

TYPED_TEST_CASE_P(MklMatMulOpTest);

TYPED_TEST_P(MklMatMulOpTest, Matmul_1_1_1_false_false) {
  this->VerifyMKLMatMul(1, 1, 1, false, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_1_128_128_false_false) {
  this->VerifyMKLMatMul(1, 128, 128, false, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_1_128_1_false_false) {
  this->VerifyMKLMatMul(1, 128, 1, false, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_128_128_false_false) {
  this->VerifyMKLMatMul(128, 128, 128, false, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_128_1_false_false) {
  this->VerifyMKLMatMul(128, 128, 1, false, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_1_1_false_false) {
  this->VerifyMKLMatMul(128, 1, 1, false, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_1_128_128_false_true) {
  this->VerifyMKLMatMul(1, 128, 128, false, true);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_1_128_1_true_true) {
  this->VerifyMKLMatMul(1, 128, 1, true, true);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_128_1_true_false) {
  this->VerifyMKLMatMul(128, 128, 1, true, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_1_1_false_true) {
  this->VerifyMKLMatMul(128, 1, 1, false, true);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_128_128_true_false) {
  this->VerifyMKLMatMul(128, 128, 128, true, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_1_1_1_true_false) {
  this->VerifyMKLMatMul(1, 1, 1, true, false);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_128_128_false_true) {
  this->VerifyMKLMatMul(128, 128, 128, false, true);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_128_128_128_true_true) {
  this->VerifyMKLMatMul(128, 128, 128, true, true);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_32_128_32_true_true) {
  this->VerifyMKLMatMul(32, 128, 32, true, true);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_256_128_256_true_true) {
  this->VerifyMKLMatMul(256, 128, 256, true, true);
}

TYPED_TEST_P(MklMatMulOpTest, Matmul_256_1024_256_true_true) {
  this->VerifyMKLMatMul(256, 1024, 256, true, true);
}

REGISTER_TYPED_TEST_CASE_P(MklMatMulOpTest,
                          Matmul_1_1_1_false_false,
                          Matmul_1_128_128_false_false,
                          Matmul_1_128_1_false_false,
                          Matmul_128_128_128_false_false,
                          Matmul_128_128_1_false_false,
                          Matmul_128_1_1_false_false,
                          Matmul_1_128_128_false_true,
                          Matmul_1_128_1_true_true,
                          Matmul_128_128_1_true_false,
                          Matmul_128_1_1_false_true,
                          Matmul_128_128_128_true_false,
                          Matmul_1_1_1_true_false,
                          Matmul_128_128_128_false_true,
                          Matmul_128_128_128_true_true,
                          Matmul_32_128_32_true_true,
                          Matmul_256_128_256_true_true,
                          Matmul_256_1024_256_true_true
                          );

using MklMatMulDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, MklMatMulOpTest,
                              MklMatMulDataTypes);

// using MklMatMulDataTypes_bf16 = ::testing::Types<bfloat16>;
// INSTANTIATE_TYPED_TEST_CASE_P(Test_bf16, MklMatMulOpTest,
//                               MklMatMulDataTypes_bf16);

//----------------------------------------------------------------------------//
// Fused MatMul Unit Tests are below.                                         //
//----------------------------------------------------------------------------//

// using BiasAddGraphRunner =
//     std::function<void(const Tensor& input_data, const Tensor& filter_data,
//                        const Tensor& bias_data, Tensor* out)>;

using FusedGraphRunner =
    std::function<void(const Tensor& input_data, const Tensor& filter_data,
                       const Tensor& bias_data,
                       const std::vector<string>& fused_ops, Tensor* out, bool transpose_a, bool transpose_b)>;

template <typename T>
class CommonTestUtilitiesFUSED : public OpsTestBase {
 public:
  void PerformConversion(DataType dtype, const Tensor& tensor,
                         const Tensor& mkl_meta_tensor, Tensor* output) {
    // Create an MKL to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // Mkl second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(tensor.shape(), tensor.flat<T>());
    AddInputFromArray<uint8>(mkl_meta_tensor.shape(),
                             mkl_meta_tensor.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    *output = *GetOutput(0);
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor.
  static void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                          Tensor* output) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void TestBody() {}

  static void VerifyFusedMatrixClose(int depth, int batch, int weight_count,
                                     const std::vector<string>& fused_ops,
                                     const FusedGraphRunner& run_default,
                                     const FusedGraphRunner& run_fused,
                                     bool transpose_a, bool transpose_b) {
    DataType dtype = DataTypeToEnum<T>::v();

    // Tensor input(dtype, {batch, depth});
    Tensor input(dtype, transpose_a ? TensorShape({depth, batch}) : TensorShape({batch, depth}));
    input.flat<T>() = input.flat<T>().template setRandom<random_gen_>();

    // Tensor weight(dtype, {depth, weight_count});
    Tensor weight(dtype, transpose_b ? TensorShape({weight_count, depth}) : TensorShape({depth, weight_count}));
    weight.flat<T>() = weight.flat<T>().template setRandom<random_gen_>();

    Tensor bias(dtype, TensorShape({weight_count}));
    // Tensor bias(dtype, TensorShape({transpose_b ? weight_count : depth}));
    bias.flat<T>() = bias.flat<T>().template setRandom<random_gen_>();

    Tensor output;
    Tensor fused_output;

    run_default(input, weight, bias, fused_ops, &output, transpose_a, transpose_b);
    run_fused(input, weight, bias, fused_ops, &fused_output, transpose_a, transpose_b);

    ASSERT_EQ(output.dtype(), fused_output.dtype());
    ASSERT_EQ(output.shape(), fused_output.shape());

    test::ExpectClose(output, fused_output, 1e-5);
  }

 private:
  using random_gen_ = Eigen::internal::NormalRandomGenerator<T>;
};

// Testing fusion of MatMul and BiasAdd
template <typename T>
class MklFusedMatMulOpTest : public OpsTestBase {
 private:
  void RunMklFusedMatMulOp(const Tensor& input, const Tensor& weight,
                           const std::vector<Tensor>& args,
                           const std::vector<string>& fused_ops,
                           Tensor* output, bool transpose_a, bool transpose_b) {
    DataType dtype = DataTypeToEnum<T>::v();
    const int num_args = args.size();
    TF_EXPECT_OK(NodeDefBuilder("MklFusedMatMul", "_MklFusedMatMul")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(num_args, dtype))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(num_args, DT_UINT8))
                     .Attr("T", dtype)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Attr("num_args", num_args)
                     .Attr("fused_ops", fused_ops)
                     .Attr("epsilon", 0.0001)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(input.shape(), input.flat<T>());
    AddInputFromArray<T>(weight.shape(), weight.flat<T>());
    for (const Tensor& arg : args)
      AddInputFromArray<T>(arg.shape(), arg.flat<T>());
    // Add MKL meta input for input, filter and bias.
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    for (const Tensor& arg : args)
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

    TF_ASSERT_OK(RunOpKernel());

    const Tensor& output_tensor = *GetOutput(0);
    const Tensor& output_meta_tensor = *GetOutput(1);
    CommonTestUtilitiesFUSED<T> test_util;
    test_util.PerformConversion(dtype, output_tensor, output_meta_tensor,
                                output);
  }

 protected:
  void VerifyFusedMatMul(const int kBatch, const int kInputChannel,
                         const int kOutputChannel,
                         const std::vector<string>& fused_ops, bool transpose_a, bool transpose_b) {
    const FusedGraphRunner run_default =
        [this](const Tensor& input, const Tensor& weight, const Tensor& bias,
               const std::vector<string>& fused_ops, Tensor* output, bool transpose_a, bool transpose_b) {
          auto root = tensorflow::Scope::NewRootScope();
          auto input_op =
              ops::Const(root.WithOpName("input"), Input::Initializer(input));
          Output next_op = ops::MatMul(root.WithOpName("matmul"), input_op,
                                       ops::Const(root.WithOpName("weight"),
                                                  Input::Initializer(weight)), ops::MatMul::TransposeA(transpose_a).TransposeB(transpose_b));

          string last_op = "";
          if (std::find(fused_ops.begin(), fused_ops.end(), "BiasAdd") !=
              fused_ops.end()) {
            last_op = "with_bias";
            next_op = ops::BiasAdd(
                root.WithOpName(last_op), next_op,
                ops::Const(root.WithOpName("bias"), Input::Initializer(bias)));
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Relu") !=
              fused_ops.end()) {
            last_op = "with_relu";
            next_op = ops::Relu(root.WithOpName(last_op), next_op);
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Relu6") !=
              fused_ops.end()) {
            last_op = "with_relu6";
            next_op = ops::Relu6(root.WithOpName(last_op), next_op);
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Elu") !=
              fused_ops.end()) {
            last_op = "with_elu";
            next_op = ops::Elu(root.WithOpName(last_op), next_op);
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Tanh") !=
              fused_ops.end()) {
            last_op = "with_tanh";
            next_op = ops::Tanh(root.WithOpName(last_op), next_op);
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Gelu") !=
              fused_ops.end()) {
            last_op = "with_gelu";
            next_op = ops::Gelu(root.WithOpName(last_op), next_op,
                                ops::Gelu::Approximate(true));
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Gelu_erf") !=
              fused_ops.end()) {
            last_op = "with_gelu_erf";
            next_op = ops::Gelu(root.WithOpName(last_op), next_op,
                                ops::Gelu::Approximate(false));
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Add") !=
              fused_ops.end()) {
            last_op = "with_add";
            next_op = ops::Add(root.WithOpName("with_add"), next_op, input_op);
          }

          CommonTestUtilitiesFUSED<T>::RunAndFetch(root, last_op, output);
        };

    const FusedGraphRunner run_fused =
        [this](const Tensor& input, const Tensor& weight, const Tensor& bias,
               const std::vector<string>& fused_ops, Tensor* output, bool transpose_a, bool transpose_b) {
          std::vector<Tensor> fused_input = {bias};
          if (std::find(fused_ops.begin(), fused_ops.end(), "Add") !=
              fused_ops.end()) {
            fused_input.push_back(input);
          }
          RunMklFusedMatMulOp(input, weight, fused_input, fused_ops, output, transpose_a, transpose_b);
        };

    CommonTestUtilitiesFUSED<T>::VerifyFusedMatrixClose(kInputChannel, kBatch,
                                                   kOutputChannel, fused_ops,
                                                   run_default, run_fused, transpose_a, transpose_b);
  }
};

TYPED_TEST_CASE_P(MklFusedMatMulOpTest);

TYPED_TEST_P(MklFusedMatMulOpTest, WithBias_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel, {"BiasAdd"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBias_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel, {"BiasAdd"}, false, true);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndRelu_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Relu"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndRelu_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Relu"}, false, true);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndRelu6_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Relu6"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndRelu6_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Relu6"}, false, true);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndElu_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Elu"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndElu_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Elu"}, false, true);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndTanh_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Tanh"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndTanh_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Tanh"}, false, true);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndGelu_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Gelu"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndGelu_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Gelu"}, false, true);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndGeluErf_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Gelu_erf"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndGeluErf_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Gelu_erf"}, false, true);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndAdd_3_4_5_false_false) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 4;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Add"}, false, false);
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndAdd_3_4_5_false_true) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 4;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Add"}, false, true);
}

REGISTER_TYPED_TEST_CASE_P(MklFusedMatMulOpTest,
                          WithBias_3_4_5_false_false,
                          WithBias_3_4_5_false_true,
                          WithBiasAndRelu_3_4_5_false_false,
                          WithBiasAndRelu_3_4_5_false_true,
                          WithBiasAndRelu6_3_4_5_false_false,
                          WithBiasAndRelu6_3_4_5_false_true,
                          WithBiasAndElu_3_4_5_false_false,
                          WithBiasAndElu_3_4_5_false_true,
                          WithBiasAndTanh_3_4_5_false_false,
                          WithBiasAndTanh_3_4_5_false_true,
                          WithBiasAndGelu_3_4_5_false_false,
                          WithBiasAndGelu_3_4_5_false_true,
                          WithBiasAndGeluErf_3_4_5_false_false,
                          WithBiasAndGeluErf_3_4_5_false_true,
                          WithBiasAndAdd_3_4_5_false_false,
                          WithBiasAndAdd_3_4_5_false_true);

using MklFusedMatMulDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, MklFusedMatMulOpTest,
                              MklFusedMatMulDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Matmul(const string& kind, int m, int k, int n, bool transpose_a, bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "MatMul" : "_MklMatMul";

  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);
  Node* input_in1 = test::graph::Constant(g, in1);

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_in0)
                    .Input(input_in1)
                    .Attr("transpose_a", transpose_a)
                    .Attr("transpose_b", transpose_b);

  isDefault ? nodeBuilder : nodeBuilder.Attr("_kernel", "MklNameChangeOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_Matmul_Base(kind, M, K, N, TA, TB, T, DEVICE, NTH)                              \
  static void BM_Matmul##_##kind##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                         \
    testing::UseRealTime();                                                                \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);                    \
    SessionOptions opts;                                                                   \
    opts.config.set_intra_op_parallelism_threads(NTH);                                     \
    test::Benchmark(#DEVICE, Matmul<T>(#kind, M, K, N, TA, TB), &opts).Run(iters);         \
  }                                                                                        \
  BENCHMARK(BM_Matmul##_##kind##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH);  \

#define BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, NTH)     \
  BM_Matmul_Base(Default, M, K, N, TA, TB, T, DEVICE, NTH); \
  BM_Matmul_Base(Mkl, M, K, N, TA, TB, T, DEVICE, NTH);     \

#define BM_Matmul_NTH(M, K, N, TA, TB, T, DEVICE) \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 1);  \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 4);  \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 8);  \

#define BM_Matmul(M, K, N, TA, TB)               \
  BM_Matmul_NTH(M, K, N, TA, TB, float, cpu);    \
  BM_Matmul_NTH(M, K, N, TA, TB, bfloat16, cpu); \

/*
// Batch size of 1 included for inference.
// Typical fully connected layers
BM_Matmul(1, 512, 512, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);

BM_Matmul(1, 1024, 1024, false, false);
BM_Matmul(8, 1024, 1024, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(128, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_Matmul(1, 1024, 1024, false, true);
BM_Matmul(8, 1024, 1024, false, true);
BM_Matmul(16, 1024, 1024, false, true);
BM_Matmul(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_Matmul(1, 200, 10000, false, false);
BM_Matmul(8, 200, 10000, false, false);
BM_Matmul(20, 200, 10000, false, false);
BM_Matmul(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_Matmul(1, 10000, 200, false, true);
BM_Matmul(1, 10000, 200, false, false);
BM_Matmul(8, 10000, 200, false, true);
BM_Matmul(20, 10000, 200, false, true);
BM_Matmul(20, 20000, 200, false, true);

// Test some matrix-vector multiplies.
BM_Matmul(50, 50, 1, false, false);
BM_Matmul(50, 50, 1, true, false);
BM_Matmul(50, 50, 1, false, true);
BM_Matmul(50, 50, 1, true, true);
BM_Matmul(500, 500, 1, false, false);
BM_Matmul(500, 500, 1, true, false);
BM_Matmul(500, 500, 1, false, true);
BM_Matmul(500, 500, 1, true, true);
BM_Matmul(2000, 2000, 1, false, false);
BM_Matmul(2000, 2000, 1, true, false);
BM_Matmul(2000, 2000, 1, false, true);
BM_Matmul(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 50, 50, true, false);
BM_Matmul(1, 50, 50, false, true);
BM_Matmul(1, 50, 50, true, true);
BM_Matmul(1, 500, 500, false, false);
BM_Matmul(1, 500, 500, true, false);
BM_Matmul(1, 500, 500, false, true);
BM_Matmul(1, 500, 500, true, true);
BM_Matmul(1, 2000, 2000, false, false);
BM_Matmul(1, 2000, 2000, true, false);
BM_Matmul(1, 2000, 2000, false, true);
BM_Matmul(1, 2000, 2000, true, true);

// Test some rank-one products.
BM_Matmul(50, 1, 50, false, false);
BM_Matmul(50, 1, 50, true, false);
BM_Matmul(50, 1, 50, false, true);
BM_Matmul(50, 1, 50, true, true);
BM_Matmul(500, 1, 500, false, false);
BM_Matmul(500, 1, 500, true, false);
BM_Matmul(500, 1, 500, false, true);
BM_Matmul(500, 1, 500, true, true);
BM_Matmul(2000, 1, 2000, false, false);
BM_Matmul(2000, 1, 2000, true, false);
BM_Matmul(2000, 1, 2000, false, true);
BM_Matmul(2000, 1, 2000, true, true);
*/

// Vector * Vector
BM_Matmul(1, 50, 1, false, false);
BM_Matmul(1, 2000, 1, false, false);

BM_Matmul(50, 1, 50, false, false);
BM_Matmul(2000, 1, 2000, false, false);

// Vector * Matrix
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 2000, 2000, false, false);

BM_Matmul(50, 50, 1, false, false);
BM_Matmul(2000, 2000, 1, false, false);

// Matrix * Matrix
BM_Matmul(32, 32, 32, false, false);
BM_Matmul(51200, 64, 64, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(256, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

BM_Matmul(2560, 64, 1, false, false);
BM_Matmul(2560, 448, 1, false, false);
BM_Matmul(2560, 2304, 64, false, false);
BM_Matmul(2560, 1040, 1536, false, false);
BM_Matmul(2560, 14435, 2304, false, false);

/*
BM_Matmul(14435, 2560, 2304, true, false);
BM_Matmul(2560, 2304, 14435, false, true);

BM_Matmul(64, 2560, 1, true, false);
BM_Matmul(2560, 1, 64, false, true);

BM_Matmul(448, 2560, 1, true, false);
BM_Matmul(2560, 1, 448, false, true);

BM_Matmul(2304, 2560, 64, true, false);
BM_Matmul(2560, 64, 2304, false, true);

BM_Matmul(1040, 2560, 1536, true, false);
BM_Matmul(2560, 1536, 1040, false, true);
*/

template <typename T>
static Graph* FusedMatMul(const string& kind, int m, int k, int n,
                          bool transpose_a, bool transpose_b, const string& activation = "") {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  std::vector<string> fused_ops{"BiasAdd"};

  if(activation != "" && activation != "null"){
    fused_ops.push_back(activation);
  }

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "_FusedMatMul" : "_MklFusedMatMul";

  int num_args = 1;
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  Tensor bias(type, TensorShape({transpose_b ? k : n}));
  bias.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);
  Node* input_in1 = test::graph::Constant(g, in1);
  Node* input_bias = test::graph::Constant(g, bias, absl::StrCat("arg", 1));

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  std::vector<NodeBuilder::NodeOut> args;
  std::vector<NodeBuilder::NodeOut> args_not_mkl;
  args.push_back(input_bias);
  args_not_mkl.push_back(not_mkl_shape);

  auto nodeBuilder = NodeBuilder(g->NewName("fused_matmul"), op_name)
                    .Input(input_in0)
                    .Input(input_in1)
                    .Input(args)
                    .Attr("T", type)
                    .Attr("num_args", num_args)
                    .Attr("fused_ops", fused_ops)
                    .Attr("transpose_a", transpose_a)
                    .Attr("transpose_b", transpose_b);

  isDefault ? nodeBuilder : nodeBuilder.Attr("_kernel", "MklLayoutDependentOp")
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(args_not_mkl);

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_FusedMatMul_Base(kind, ACT, M, K, N, TA, TB, T, DEVICE, NTH)                                 \
  static void BM_FusedMatMul##_##kind##_##ACT##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                                      \
    testing::UseRealTime();                                                                             \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);                                 \
    SessionOptions opts;                                                                                \
    opts.config.set_intra_op_parallelism_threads(NTH);                                                  \
    test::Benchmark(#DEVICE, FusedMatMul<T>(#kind, M, K, N, TA, TB, #ACT), &opts).Run(iters);           \
  }                                                                                                     \
  BENCHMARK(BM_FusedMatMul##_##kind##_##ACT##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH);  \

#define BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, NTH)     \
  BM_FusedMatMul_Base(Default, ACT, M, K, N, TA, TB, T, DEVICE, NTH); \
  BM_FusedMatMul_Base(Mkl, ACT, M, K, N, TA, TB, T, DEVICE, NTH);     \

#define BM_FusedMatMul_NTH(ACT, M, K, N, TA, TB, T, DEVICE) \
  BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, 1);  \
  BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, 4);  \
  BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, 8);  \

#define BM_FusedMatMul_ACT(M, K, N, TA, TB, T, DEVICE)  \
  BM_FusedMatMul_NTH(null, M, K, N, TA, TB, T, DEVICE); \
  BM_FusedMatMul_NTH(Relu, M, K, N, TA, TB, T, DEVICE); \

#define BM_FusedMatMul(M, K, N, TA, TB)               \
  BM_FusedMatMul_ACT(M, K, N, TA, TB, float, cpu);    \
  BM_FusedMatMul_ACT(M, K, N, TA, TB, bfloat16, cpu); \

// Vector * Vector
BM_FusedMatMul(1, 50, 1, false, false);
BM_FusedMatMul(1, 2000, 1, false, false);

BM_FusedMatMul(50, 1, 50, false, false);
BM_FusedMatMul(2000, 1, 2000, false, false);

// Vector * Matrix
BM_FusedMatMul(1, 50, 50, false, false);
BM_FusedMatMul(1, 2000, 2000, false, false);

BM_FusedMatMul(50, 50, 1, false, false);
BM_FusedMatMul(2000, 2000, 1, false, false);

// Matrix * Matrix
BM_FusedMatMul(32, 32, 32, false, false);
BM_FusedMatMul(51200, 64, 64, false, false);
BM_FusedMatMul(8, 512, 512, false, false);
BM_FusedMatMul(128, 512, 512, false, false);
BM_FusedMatMul(16, 1024, 1024, false, false);
BM_FusedMatMul(256, 1024, 1024, false, false);
BM_FusedMatMul(4096, 4096, 4096, false, false);

BM_FusedMatMul(2560, 64, 1, false, false);
BM_FusedMatMul(2560, 448, 1, false, false);
BM_FusedMatMul(2560, 2304, 64, false, false);
BM_FusedMatMul(2560, 1040, 1536, false, false);
BM_FusedMatMul(2560, 14435, 2304, false, false);

}  // end namespace tensorflow

#endif  // INTEL_MKL
