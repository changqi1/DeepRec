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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/framework/fake_input.h"


namespace tensorflow {

//----------------------------------------------------------------------------//
// MatMul Unit Tests are below.                                               //
//----------------------------------------------------------------------------//


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

    run_mkl(input, weight, &mkl_output, transpose_a, transpose_b);
    run_default(input, weight, &output, transpose_a, transpose_b);

    ASSERT_EQ(output.dtype(), mkl_output.dtype());
    ASSERT_EQ(output.shape(), mkl_output.shape());
    test::ExpectClose(output, mkl_output, 1.5e-5, 1.0e-3);
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
        NodeDefBuilder("tuning_matmul", "MatMul") //build node
            .Input(FakeInput(dtype))
            .Input(FakeInput(dtype))
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
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
	  setenv("TF_TUNING_ENABLE", "false", 1);
          Output next_op = ops::MatMul(root.WithOpName("matmul"), input_op,
                                       ops::Const(root.WithOpName("weight"),
                                       Input::Initializer(weight)),
                                       ops::MatMul::TransposeA(transpose_a).TransposeB(transpose_b)
                                       );
          string last_op = "matmul";
          CommonTestUtilities<T>::RunAndFetch(root, last_op, output);
	  unsetenv("TF_TUNING_ENABLE");
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

REGISTER_TEST_CASE(128, 256, 512, false, false);
REGISTER_TEST_CASE(128, 256, 512, false, true);
REGISTER_TEST_CASE(128, 256, 512, true, false);
REGISTER_TEST_CASE(128, 256, 512, true, true);
REGISTER_TYPED_TEST_CASE_P(MklMatMulOpTest,
                          Matmul_128_256_512_false_true,
                          Matmul_128_256_512_true_false,
                          Matmul_128_256_512_true_true,
                          Matmul_128_256_512_false_false
                          );

using MklMatMulDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, MklMatMulOpTest,
                              MklMatMulDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Matmul(int m, int k, int n, bool transpose_a, bool transpose_b,
                     DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  test::graph::Matmul(g, test::graph::Constant(g, in0),
                      test::graph::Constant(g, in1), transpose_a, transpose_b);
  return g;
}

#define BM_MatmulDev(M, K, N, TA, TB, T, TFTYPE, DEVICE)                       \
  static void BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
      int iters) {                                                             \
    testing::UseRealTime();                                                    \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);        \
    test::Benchmark(#DEVICE, Matmul<T>(M, K, N, TA, TB, TFTYPE)).Run(iters);   \
  }                                                                            \
  BENCHMARK(BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE);

#define BM_Matmul(M, K, N, TA, TB)                                       \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu);                   \
  // BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu); \
  // BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, gpu);                   \
  // BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, gpu); \
/* Uncomment to enable benchmarks for double/complex128: */              \
// BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu); \
// BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

// Batch size of 1 included for inference.
// Typical fully connected layers
BM_Matmul(1, 512, 512, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 256, 512, false, false);
BM_Matmul(128, 256, 512, false, true);
BM_Matmul(128, 256, 512, true, false);
BM_Matmul(128, 256, 512, true, true);

// BM_Matmul(1, 1024, 1024, false, false);
// BM_Matmul(8, 1024, 1024, false, false);
// BM_Matmul(16, 1024, 1024, false, false);
// BM_Matmul(128, 1024, 1024, false, false);
// BM_Matmul(4096, 4096, 4096, false, false);

// // Backward for fully connected layers
// BM_Matmul(1, 1024, 1024, false, true);
// BM_Matmul(8, 1024, 1024, false, true);
// BM_Matmul(16, 1024, 1024, false, true);
// BM_Matmul(128, 1024, 1024, false, true);

// // Forward softmax with large output size
// BM_Matmul(1, 200, 10000, false, false);
// BM_Matmul(8, 200, 10000, false, false);
// BM_Matmul(20, 200, 10000, false, false);
// BM_Matmul(20, 200, 20000, false, false);

// // Backward softmax with large output size
// BM_Matmul(1, 10000, 200, false, true);
// BM_Matmul(1, 10000, 200, false, false);
// BM_Matmul(8, 10000, 200, false, true);
// BM_Matmul(20, 10000, 200, false, true);
// BM_Matmul(20, 20000, 200, false, true);

// // Test some matrix-vector multiplies.
// BM_Matmul(50, 50, 1, false, false);
// BM_Matmul(50, 50, 1, true, false);
// BM_Matmul(50, 50, 1, false, true);
// BM_Matmul(50, 50, 1, true, true);
// BM_Matmul(500, 500, 1, false, false);
// BM_Matmul(500, 500, 1, true, false);
// BM_Matmul(500, 500, 1, false, true);
// BM_Matmul(500, 500, 1, true, true);
// BM_Matmul(2000, 2000, 1, false, false);
// BM_Matmul(2000, 2000, 1, true, false);
// BM_Matmul(2000, 2000, 1, false, true);
// BM_Matmul(2000, 2000, 1, true, true);

// // Test some vector-matrix multiplies.
// BM_Matmul(1, 50, 50, false, false);
// BM_Matmul(1, 50, 50, true, false);
// BM_Matmul(1, 50, 50, false, true);
// BM_Matmul(1, 50, 50, true, true);
// BM_Matmul(1, 500, 500, false, false);
// BM_Matmul(1, 500, 500, true, false);
// BM_Matmul(1, 500, 500, false, true);
// BM_Matmul(1, 500, 500, true, true);
// BM_Matmul(1, 2000, 2000, false, false);
// BM_Matmul(1, 2000, 2000, true, false);
// BM_Matmul(1, 2000, 2000, false, true);
// BM_Matmul(1, 2000, 2000, true, true);

// // Test some rank-one products.
// BM_Matmul(50, 1, 50, false, false);
// BM_Matmul(50, 1, 50, true, false);
// BM_Matmul(50, 1, 50, false, true);
// BM_Matmul(50, 1, 50, true, true);
// BM_Matmul(500, 1, 500, false, false);
// BM_Matmul(500, 1, 500, true, false);
// BM_Matmul(500, 1, 500, false, true);
// BM_Matmul(500, 1, 500, true, true);
// BM_Matmul(2000, 1, 2000, false, false);
// BM_Matmul(2000, 1, 2000, true, false);
// BM_Matmul(2000, 1, 2000, false, true);
// BM_Matmul(2000, 1, 2000, true, true);

}  // end namespace tensorflow
