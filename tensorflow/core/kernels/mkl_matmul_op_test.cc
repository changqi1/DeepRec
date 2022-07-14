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
  void PerformConversion(DataType dtype, const Tensor& tensor, Tensor* output) { // Default, convert shape
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(tensor.shape(), tensor.flat<T>());
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
        NodeDefBuilder("tuning_matmul", "TuningMatmul") //build node
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

#define REGISTER_TEST_CASE(M, K, N, TA, TB)                               \
  TYPED_TEST_P(MklMatMulOpTest, Matmul##_##M##_##K##_##N##_##TA##_##TB) { \
    this->VerifyMKLMatMul(M, K, N, TA, TB);                               \
  }

REGISTER_TEST_CASE(5, 8192, 4096, false, false);
REGISTER_TEST_CASE(1024, 696, 64, false, false);
REGISTER_TEST_CASE(1024, 184, 256, false, false);
REGISTER_TEST_CASE(1024, 184, 64, false, false);
REGISTER_TEST_CASE(204800, 200, 64, false, false);
REGISTER_TEST_CASE(71680, 420, 64, false, false);
REGISTER_TEST_CASE(51200, 356, 256, false, false);
REGISTER_TEST_CASE(51200, 232, 64, false, false);
REGISTER_TEST_CASE(20480, 260, 64, false, false);
REGISTER_TEST_CASE(5120, 210, 64, false, false);
REGISTER_TEST_CASE(204800, 200, 128, false, false);
REGISTER_TEST_CASE(71680, 420, 128, false, false);
REGISTER_TEST_CASE(51200, 356, 512, false, false);
REGISTER_TEST_CASE(51200, 232, 128, false, false);
REGISTER_TEST_CASE(20480, 260, 128, false, false);
REGISTER_TEST_CASE(5120, 210, 128, false, false);

REGISTER_TYPED_TEST_CASE_P(MklMatMulOpTest,
                          Matmul_1024_184_256_false_false,
                          Matmul_1024_184_64_false_false,
                          Matmul_1024_696_64_false_false,
                          Matmul_204800_200_128_false_false,
                          Matmul_204800_200_64_false_false,
                          Matmul_20480_260_128_false_false,
                          Matmul_20480_260_64_false_false,
                          Matmul_51200_232_128_false_false,
                          Matmul_51200_232_64_false_false,
                          Matmul_51200_356_256_false_false,
                          Matmul_51200_356_512_false_false,
                          Matmul_5120_210_128_false_false,
                          Matmul_5120_210_64_false_false,
                          Matmul_5_8192_4096_false_false,
                          Matmul_71680_420_128_false_false,
                          Matmul_71680_420_64_false_false
                          );

using MklMatMulDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, MklMatMulOpTest,
                              MklMatMulDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Matmul(const string& kind, int m, int k, int n, bool transpose_a, bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Tuning");
  // string op_name = isDefault ? "TuningMatmul" : "_MklMatMul";
  string op_name = isDefault ? "TuningMatmul" : "_MklMatMul";

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
                    .Attr("transpose_b", transpose_b)
                    .Attr("_kernel", "MklNameChangeOp");

  // isDefault ? nodeBuilder : nodeBuilder.Attr("_kernel", "MklNameChangeOp");

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

#define BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, NTH)    \
  BM_Matmul_Base(Tuning, M, K, N, TA, TB, T, DEVICE, NTH); \
  BM_Matmul_Base(Mkl, M, K, N, TA, TB, T, DEVICE, NTH);    \

#define BM_Matmul_NTH(M, K, N, TA, TB, T, DEVICE) \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 1);  \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 4);  \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 8);  \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 16); \

#define BM_Matmul(M, K, N, TA, TB)                  \
  BM_Matmul_NTH(M, K, N, TA, TB, float, cpu);       \


BM_Matmul(5, 8192, 4096, false, false);
BM_Matmul(1024, 696, 64, false, false);
BM_Matmul(1024, 184, 256, false, false);
BM_Matmul(1024, 184, 64, false, false);

BM_Matmul(204800, 200, 64, false, false);
BM_Matmul(71680, 420, 64, false, false);
BM_Matmul(51200, 356, 256, false, false);
BM_Matmul(51200, 232, 64, false, false);
BM_Matmul(20480, 260, 64, false, false);
BM_Matmul(5120, 210, 64, false, false);

BM_Matmul(204800, 200, 128, false, false);
BM_Matmul(71680, 420, 128, false, false);
BM_Matmul(51200, 356, 512, false, false);
BM_Matmul(51200, 232, 128, false, false);
BM_Matmul(20480, 260, 128, false, false);
BM_Matmul(5120, 210, 128, false, false);


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
// BM_FusedMatMul(1, 50, 1, false, false);
// BM_FusedMatMul(1, 2000, 1, false, false);

// BM_FusedMatMul(50, 1, 50, false, false);
// BM_FusedMatMul(2000, 1, 2000, false, false);

// // Vector * Matrix
// BM_FusedMatMul(1, 50, 50, false, false);
// BM_FusedMatMul(1, 2000, 2000, false, false);

// BM_FusedMatMul(50, 50, 1, false, false);
// BM_FusedMatMul(2000, 2000, 1, false, false);

// // Matrix * Matrix
// BM_FusedMatMul(32, 32, 32, false, false);
// BM_FusedMatMul(51200, 64, 64, false, false);
// BM_FusedMatMul(8, 512, 512, false, false);
// BM_FusedMatMul(128, 512, 512, false, false);
// BM_FusedMatMul(16, 1024, 1024, false, false);
// BM_FusedMatMul(256, 1024, 1024, false, false);
// BM_FusedMatMul(4096, 4096, 4096, false, false);

// BM_FusedMatMul(2560, 64, 1, false, false);
// BM_FusedMatMul(2560, 448, 1, false, false);
// BM_FusedMatMul(2560, 2304, 64, false, false);
// BM_FusedMatMul(2560, 1040, 1536, false, false);
// BM_FusedMatMul(2560, 14435, 2304, false, false);

}  // end namespace tensorflow

#endif  // INTEL_MKL
