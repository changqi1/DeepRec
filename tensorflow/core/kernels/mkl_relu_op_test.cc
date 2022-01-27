/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <functional>
#include <vector>
#include "dnnl.hpp"
#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/fake_input.h"
#include <gtest/gtest.h>
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
//----------------------------------------------------------------------------//
// Relu Unit Tests are below.                                                 //
//----------------------------------------------------------------------------//
namespace MKLReluTestDefs {
    typedef std::tuple<
    DataType,             // input_type
    std::vector<long long int>
    > ReluTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
        DataType::DT_BFLOAT16
    };
    std::vector<std::vector<long long int>> SIZES_2D = {{64, 64}, {1,1}, {1,32}, {32, 21}};
    std::vector<std::vector<long long int>> SIZES_3D = {{32, 16, 1}, {32, 32, 32}, {128, 128, 128}, {1, 1, 1}};
    std::vector<std::vector<long long int>> SIZES_4D = {{32, 32, 32, 32}, {16, 1, 1, 1}, {31, 63, 15, 7}};
} // namespace MKLReluTestDefs

using namespace MKLReluTestDefs;
class ReluTestBase :
    public ::testing::WithParamInterface<MKLReluTestDefs::ReluTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    DataType input_type;
    std::vector<long long int> input_size;
    // Test input Tensors (filled in SetUp)
    Tensor input;
    Tensor zeros;
    // Test output Tensors (filled in Run method)
    Tensor mkl_values;
    Tensor default_values;

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

    void runDefault() {
          auto root = tensorflow::Scope::NewRootScope();
          auto input_op =
              ops::Const(root.WithOpName("input"), Input::Initializer(input));
          Output next_op = ops::Relu(root.WithOpName("relu"), input_op);
          string last_op = "relu";
          RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
    TF_EXPECT_OK(
        NodeDefBuilder("mkl_relu_op", "_MklRelu") //build node
            .Input(FakeInput(input_type))
            .Input(FakeInput(DT_UINT8))
            .Attr("_kernel", "MklLayoutDependentOp")
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp()); //initial
    switch(input_type) {
        case DT_FLOAT:
            AddInputFromArray<float>(input.shape(), input.flat<float>()); // input
            break;
        case DT_BFLOAT16:
            AddInputFromArray<bfloat16>(input.shape(), input.flat<bfloat16>()); // input
            break;
        default:
            GTEST_FAIL() << "Unexpected DataType";
    }
    AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // input
    TF_EXPECT_OK(RunOpKernel()); //Run the node computation
    mkl_values = *GetOutput(0); //Get output
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<ReluTestParams> obj) {
        DataType input_type;
        std::vector<long long int> input_size;
        std::tie(input_type, input_size) = obj.param;
        std::ostringstream result;
        result << "Relu_Type_";
        switch(input_type) {
            case DataType::DT_FLOAT:
                result << "FLOAT";
                break;
            case DataType::DT_BFLOAT16:
                result << "BFLOAT16";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }
        result << "_Sizes";
        for (auto &x : input_size) {
            result << "_" << x;
        }
        return result.str();
    }

    void SetUp() {
        std::tie(input_type, input_size) = this->GetParam();
        input = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size())));
        switch(input_type) {
            case DT_FLOAT:
                input.flat<float>() = input.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>(); // input
                break;
            case DT_BFLOAT16:
                input.flat<bfloat16>() = input.flat<bfloat16>().template setRandom<Eigen::internal::NormalRandomGenerator<bfloat16>>(); // input
                break;
            default:
                GTEST_FAIL() << "Unexpected DataType";
        }
        zeros = Tensor(DT_UINT8, TensorShape({64, 64}));
        auto zeros_mapped = zeros.tensor<uint8_t, 2>();
        for(int i = 0; i < 64; i++){
            for(int j = 0; j < 64; j++){
                zeros_mapped(i, j) = 0;
            }
        }
    }

    void Run() {
        runDefault();
        runMkl();
    }

    void Validate() {
        ASSERT_EQ(default_values.dtype(), mkl_values.dtype());
        ASSERT_EQ(default_values.shape(), mkl_values.shape());
        test::ExpectClose(default_values, mkl_values, 1e-5);
    }
};

TEST_P(ReluTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(Relu2D, ReluTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_2D)),
    ReluTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Relu3D, ReluTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_3D)),
    ReluTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Relu4D, ReluTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_4D)),
    ReluTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Activation(const string& op_name, const string& kind,
                         const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const string node_name = kind + "_" + op_name;
  const bool isForwardOp = !tensorflow::str_util::EndsWith(op_name, "Grad");
  const bool isDefault = (kind == "Default");

  Tensor input_t(type, shape);
  input_t.flat<T>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  if (isForwardOp) {
    auto nodeBuilder = NodeBuilder(graph->NewName(node_name), isDefault ? op_name : "_Mkl" + op_name)
                           .Input(input)
                           .Attr("T", type);
    isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                         .Attr("_kernel", "MklLayoutDependentOp");
    TF_CHECK_OK(nodeBuilder.Finalize(graph, nullptr));
    return graph;
  }

  Tensor grad_t(type, shape);
  grad_t.flat<T>().setRandom();
  Node* grad = test::graph::Constant(graph, grad_t, "grad");

  auto nodeBuilder = NodeBuilder(graph->NewName(node_name), isDefault ? op_name : "_Mkl" + op_name)
                         .Input(grad)
                         .Input(input)
                         .Attr("T", type);
  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  TF_CHECK_OK(nodeBuilder.Finalize(graph, nullptr));
  return graph;
}

#define BM_Activation_Base(op, kind, name, in_shape, T, device, NTH)                 \
  static void BM_##op##_##kind##_##T##name##_##device##_##NTH(int iters) {           \
    int64 num_elements = in_shape.num_elements();                                    \
    testing::UseRealTime();                                                          \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements);               \
    SessionOptions opts;                                                             \
    opts.config.set_intra_op_parallelism_threads(NTH);                               \
    test::Benchmark(#device, Activation<T>(#op, #kind, in_shape), &opts).Run(iters); \
  }                                                                                  \
  BENCHMARK(BM_##op##_##kind##_##T##name##_##device##_##NTH)                         \

#define BM_Activation_Kind(op, name, in_shape, T, device, NTH)     \
  BM_Activation_Base(op, Default, name, in_shape, T, device, NTH); \
  BM_Activation_Base(op, Mkl, name, in_shape, T, device, NTH);     \

#define BM_Activation_NTH(op, name, in_shape, T, device) \
  BM_Activation_Kind(op, name, in_shape, T, device, 1);  \
  BM_Activation_Kind(op, name, in_shape, T, device, 4);  \
  BM_Activation_Kind(op, name, in_shape, T, device, 8);  \

#define BM_Activation_ND(op, name, ...)                                   \
  BM_Activation_NTH(op, name, TensorShape({__VA_ARGS__}), float, cpu);    \
  BM_Activation_NTH(op, name, TensorShape({__VA_ARGS__}), bfloat16, cpu); \

#define TEST_Activation_ALL(OP)                              \
  BM_Activation_ND(OP, _2D_1x512, 1, 512);                   \
  BM_Activation_ND(OP, _2D_512x1, 512, 1);                   \
  BM_Activation_ND(OP, _2D_32x32, 32, 32);                   \
  BM_Activation_ND(OP, _2D_512x512, 512, 512);               \
  BM_Activation_ND(OP, _3D_32x128x128, 32, 128, 128);        \
  BM_Activation_ND(OP, _4D_32x32x128x128, 32, 32, 128, 128); \

TEST_Activation_ALL(Tanh)
TEST_Activation_ALL(TanhGrad)
TEST_Activation_ALL(Elu)
TEST_Activation_ALL(EluGrad)
TEST_Activation_ALL(Relu)
TEST_Activation_ALL(ReluGrad)
TEST_Activation_ALL(Relu6)
TEST_Activation_ALL(Relu6Grad)
TEST_Activation_ALL(LeakyRelu)
TEST_Activation_ALL(LeakyReluGrad)

}  // namespace tensorflow
#endif  // INTEL_MKL
