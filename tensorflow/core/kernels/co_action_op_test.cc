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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

template <typename T>
void show_log(T msg){
  std::cout << std::endl << "------ marvin ------" << std::endl
            << ">>> " << msg << std::endl
            << "--------------------" << std::endl << std::endl;
}
//----------------------------------------------------------------------------//
// Co-action Unit Tests are below.                                               //
//----------------------------------------------------------------------------//
using GraphRunner =
    std::function<void(const Tensor& input_data, const Tensor& filter_data, Tensor* out)>;

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
                          Tensor* output, const NodeDef* fetch_node = nullptr) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    if (fetch_node) {
      *graph.add_node() = *fetch_node;
    }

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void TestBody() {}

  // Compare two outcomes default & opt by calling run_default() & run_opt()
  static void VerifyTensorClose(int bs, int pl, int m, int k, int n,
                                     const GraphRunner& run_default,
                                     const GraphRunner& run_opt) { 
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor input(dtype, TensorShape({1, pl, m, k}));
    input.flat<T>() = input.flat<T>().template setRandom<random_gen_>();
    // input.flat<T>() = input.flat<T>().template setConstant(1);

    Tensor weight(dtype, TensorShape({bs, pl, k, n}));
    weight.flat<T>() = weight.flat<T>().template setRandom<random_gen_>();
    // weight.flat<T>() = weight.flat<T>().template setConstant(1);

    Tensor output;
    Tensor opt_output;

    run_default(input, weight, &output);
    run_opt(input, weight, &opt_output);

    ASSERT_EQ(output.dtype(), opt_output.dtype());
    ASSERT_EQ(output.shape(), opt_output.shape());

    test::ExpectClose(output, opt_output, 1e-4);
  }

 private:
  using random_gen_ = Eigen::internal::NormalRandomGenerator<T>;
};

// Testing OptCoActionOpTest
template <typename T>
class OptCoActionOpTest : public OpsTestBase {
 private:
  void RunCoActionOp(const Tensor& input, const Tensor& weight, Tensor* output) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();

    Output o_input =
        ops::Const(root.WithOpName("input"), Input::Initializer(input));
    Output o_weight =
        ops::Const(root.WithOpName("weight"), Input::Initializer(weight));

    NodeDef co_action;
    TF_EXPECT_OK(NodeDefBuilder("co_action", "CoAction") //build node
                  .Input({o_input.name(), 0, dtype})
                  .Input({o_weight.name(), 0, dtype})
                  .Attr("pow_num", 2)
                  .Finalize(&co_action));
    
    CommonTestUtilities<T>::RunAndFetch(root, co_action.name(), output, &co_action);
  }

  void RunOptCoActionOp(const Tensor& input, const Tensor& weight, Tensor* output) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();

    Output o_input =
        ops::Const(root.WithOpName("input"), Input::Initializer(input));
    Output o_weight =
        ops::Const(root.WithOpName("weight"), Input::Initializer(weight));

    NodeDef opt_co_action;
    TF_EXPECT_OK(NodeDefBuilder("opt_co_action", "OptCoAction") //build node
                  .Input({o_input.name(), 0, dtype})
                  .Input({o_weight.name(), 0, dtype})
                  .Attr("pow_num", 2)
                  .Finalize(&opt_co_action));
    
    CommonTestUtilities<T>::RunAndFetch(root, opt_co_action.name(), output, &opt_co_action);
  }

 protected:
  void VerifyOptCoAction(int bs, int pl, int m, int k, int n){
    const GraphRunner run_default =
        [this](const Tensor& input, const Tensor& weight, Tensor* output) {
          RunCoActionOp(input, weight, output);
        };

    const GraphRunner run_opt =
        [this](const Tensor& input, const Tensor& weight, Tensor* output) {
          RunOptCoActionOp(input, weight, output);
        };

    CommonTestUtilities<T>::VerifyTensorClose(bs, pl, m, k, n, run_default, run_opt);
  }
};

TYPED_TEST_CASE_P(OptCoActionOpTest);

#define REGISTER_TEST_CASE(BS, PL, M, K, N)                                      \
  TYPED_TEST_P(OptCoActionOpTest, OptCoAction##_##BS##_##PL##_##M##_##K##_##N) { \
    this->VerifyOptCoAction(BS, PL, M, K, N);                                    \
  }

// REGISTER_TEST_CASE(1, 1, 50, 5, 4);
REGISTER_TEST_CASE(200, 4, 50, 5, 4);
REGISTER_TEST_CASE(8, 2, 50, 5, 4);
// REGISTER_TEST_CASE(8, 1, 50, 5, 4);

REGISTER_TYPED_TEST_CASE_P(OptCoActionOpTest,
                          // OptCoAction_1_1_50_5_4,
                          OptCoAction_200_4_50_5_4,
                          OptCoAction_8_2_50_5_4
                          // OptCoAction_8_1_50_5_4
                          );

using OptCoActionDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, OptCoActionOpTest,
                              OptCoActionDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//
template <typename T>
static Graph* CoAction(const string& kind, int m, int k, int n, int bs, int pl) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "CoAction" : "OptCoAction";

  Tensor in0(type, TensorShape({1, pl, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, TensorShape({bs, pl, k, n}));
  in1.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);
  Node* input_in1 = test::graph::Constant(g, in1);

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_in0)
                    .Input(input_in1)
                    .Attr("pow_num", 2);

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_CoAction_Base(kind, M, K, N, BS, PL, T, DEVICE, NTH)                              \
  static void BM_CoAction##_##kind##_##M##_##K##_##N##_##BS##_##PL##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                           \
    testing::UseRealTime();                                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters) * 200 * PL * M * K * N * 2 * 2);       \
    SessionOptions opts;                                                                     \
    opts.config.set_intra_op_parallelism_threads(NTH);                                       \
    test::Benchmark(#DEVICE, CoAction<T>(#kind, M, K, N, BS, PL), &opts).Run(iters);         \
  }                                                                                          \
  BENCHMARK(BM_CoAction##_##kind##_##M##_##K##_##N##_##BS##_##PL##_##T##_##DEVICE##_##NTH);  \

#define BM_CoAction_kind(M, K, N, BS, PL, T, DEVICE, NTH)     \
  BM_CoAction_Base(Default, M, K, N, BS, PL, T, DEVICE, NTH); \
  BM_CoAction_Base(Opt, M, K, N, BS, PL, T, DEVICE, NTH);     \

#define BM_CoAction_NTH(M, K, N, BS, PL, T, DEVICE)    \
  BM_CoAction_kind(M, K, N, BS, PL, T, DEVICE, 8);     \
  // BM_CoAction_kind(M, K, N, BS, PL, T, DEVICE, 4);  \
  // BM_CoAction_kind(M, K, N, BS, PL, T, DEVICE, 8);  \

#define BM_CoAction(M, K, N, BS, PL)                  \
  BM_CoAction_NTH(M, K, N, BS, PL, float, cpu);       \
  // BM_CoAction_NTH(M, K, N, BS, PL, bfloat16, cpu); \

// BM_CoAction(50, 5, 4, 1, 4);
// BM_CoAction(50, 5, 4, 1, 30);
// BM_CoAction(150, 5, 4, 1, 4);
// BM_CoAction(150, 5, 4, 1, 30);
BM_CoAction(150, 5, 4, 200, 30);


template <typename T>
static Graph* CoActionIndicator(const string& kind, int m, int k, int n, int bs, int pl) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "CoActionIndicator" : "OptCoActionIndicator";

  Tensor in0(type, TensorShape({bs, pl, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, TensorShape({bs, pl, k, n}));
  in1.flat<T>().setRandom();
  Tensor ind(DT_INT64, TensorShape({bs}));
  ind.flat<int64>().setConstant(0);

  Node* input_in0 = test::graph::Constant(g, in0);
  Node* input_in1 = test::graph::Constant(g, in1);
  Node* input_ind = test::graph::Constant(g, ind);

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_in0)
                    .Input(input_in1)
                    .Input(input_ind)
                    .Attr("pow_num", 2);

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_CoActionIndicator_Base(kind, M, K, N, BS, PL, T, DEVICE, NTH)                              \
  static void BM_CoActionIndicator##_##kind##_##M##_##K##_##N##_##BS##_##PL##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                                    \
    testing::UseRealTime();                                                                           \
    testing::ItemsProcessed(static_cast<int64>(iters) * 200 * PL * M * K * N * 2 * 2);                \
    SessionOptions opts;                                                                              \
    opts.config.set_intra_op_parallelism_threads(NTH);                                                \
    test::Benchmark(#DEVICE, CoActionIndicator<T>(#kind, M, K, N, BS, PL), &opts).Run(iters);         \
  }                                                                                                   \
  BENCHMARK(BM_CoActionIndicator##_##kind##_##M##_##K##_##N##_##BS##_##PL##_##T##_##DEVICE##_##NTH);  \

#define BM_CoActionIndicator_kind(M, K, N, BS, PL, T, DEVICE, NTH)     \
  BM_CoActionIndicator_Base(Default, M, K, N, BS, PL, T, DEVICE, NTH); \
  BM_CoActionIndicator_Base(Opt, M, K, N, BS, PL, T, DEVICE, NTH);     \

#define BM_CoActionIndicator_NTH(M, K, N, BS, PL, T, DEVICE)    \
  BM_CoActionIndicator_kind(M, K, N, BS, PL, T, DEVICE, 8);     \
  // BM_CoActionIndicator_kind(M, K, N, BS, PL, T, DEVICE, 4);  \
  // BM_CoActionIndicator_kind(M, K, N, BS, PL, T, DEVICE, 8);  \

#define BM_CoActionIndicator(M, K, N, BS, PL)                  \
  BM_CoActionIndicator_NTH(M, K, N, BS, PL, float, cpu);       \
  // BM_CoActionIndicator_NTH(M, K, N, BS, PL, bfloat16, cpu); \

// BM_CoActionIndicator(50, 5, 4, 1, 4);
// BM_CoActionIndicator(50, 5, 4, 1, 30);
// BM_CoActionIndicator(150, 5, 4, 1, 4);
// BM_CoActionIndicator(150, 5, 4, 1, 30);
BM_CoActionIndicator(150, 5, 4, 200, 30);


}  // end namespace tensorflow
