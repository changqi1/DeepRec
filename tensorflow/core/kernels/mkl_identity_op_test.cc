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
// Identity Tests are below.                                                  //
//----------------------------------------------------------------------------//

namespace MKLIdentityTestDefs {
    typedef std::tuple<
    DataType,                   // input_type
    std::vector<long long int> // input_size_0
    > IdentityTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
        DataType::DT_BFLOAT16
    };
    std::vector<std::vector<long long int>> SIZES_2D = {{32, 32}, {32, 64}, {128, 64}};
    std::vector<std::vector<long long int>> SIZES_3D = {{32, 32, 32}, {32, 64, 32}, {128, 32, 64}};
    std::vector<std::vector<long long int>> SIZES_4D = {{32, 32, 32, 32}, {32, 64, 32, 128}, {128, 32, 16, 64}};
} // namespace IdentityTestDefs

using namespace MKLIdentityTestDefs;
class IdentityTestBase :
    public ::testing::WithParamInterface<MKLIdentityTestDefs::IdentityTestParams>,
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
      Output next_op = ops::Identity(root.WithOpName("identity"), input_op);
      string last_op = "identity";
      RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
	    TF_EXPECT_OK(
        NodeDefBuilder("mkl_identity_op", "_MklIdentity") //build node
            .Input(FakeInput(input_type))
            .Input(FakeInput(DT_UINT8))
            .Attr("_kernel", "MklLayoutDependentOp")
            .Finalize(node_def()));
      TF_EXPECT_OK(InitOp()); //initial
      switch(input_type) {
        case DT_FLOAT:
            AddInputFromArray<float>(input.shape(), input.flat<float>()); // input_0
            break;
        case DT_BFLOAT16:
            AddInputFromArray<bfloat16>(input.shape(), input.flat<bfloat16>()); // input_0
            break;
        default:
        GTEST_FAIL() << "Unexpected DataType";
      }
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>());
      TF_EXPECT_OK(RunOpKernel()); //Run the node computation
      mkl_values = *GetOutput(0); //Get outp
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<IdentityTestParams> obj) {
        DataType input_type;
        std::vector<long long int> input_size;
        std::tie(input_type, input_size) = obj.param;
        std::ostringstream result;
        result << "Identity_Type_";
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
                input.flat<bfloat16>() = input.flat<bfloat16>().template setRandom<Eigen::internal::UniformRandomGenerator<bfloat16>>(); // input
                input.flat<bfloat16>() = input.flat<bfloat16>() - input.flat<bfloat16>().constant((bfloat16)0.5);
		            input.flat<bfloat16>() = input.flat<bfloat16>() * input.flat<bfloat16>().constant((bfloat16)200.0);
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
        test::ExpectClose(default_values, mkl_values, 1e-4);
    }
};

TEST_P(IdentityTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(Identity_2D, IdentityTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_2D)),
    IdentityTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Identity_3D, IdentityTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_3D)),
    IdentityTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Identity_4D, IdentityTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_4D)),
    IdentityTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Identity(const string& kind, const TensorShape& in_shape) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Identity" : "_MklIdentity";

  Tensor input(type, in_shape);
  input.flat<T>().setRandom();

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, input))
                    .Attr("T", type);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}

#define BM_Identity_Base(kind, T, name, in_shape, DEVICE, NTH)                \
  static void BM_Identity##_##kind##_##T##name##_##DEVICE##_##NTH(            \
      int iters) {                                                            \
    int64 num_elements = in_shape.num_elements();  	                          \
    testing::UseRealTime();                                                   \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements);        \
    SessionOptions opts;                                                      \
    opts.config.set_intra_op_parallelism_threads(NTH);                        \
    test::Benchmark(#DEVICE, Identity<T>(#kind, in_shape), &opts).Run(iters); \
  }                                                                           \
  BENCHMARK(BM_Identity##_##kind##_##T##name##_##DEVICE##_##NTH);             \

#define BM_Identity_kind(T, name, in_shape, DEVICE, NTH)     \
  BM_Identity_Base(Default, T, name, in_shape, DEVICE, NTH); \
  BM_Identity_Base(Mkl, T, name, in_shape, DEVICE, NTH);     \

#define BM_Identity_NTH(T, name, in_shape, DEVICE) \
  BM_Identity_kind(T, name, in_shape, DEVICE, 1);  \
  BM_Identity_kind(T, name, in_shape, DEVICE, 4);  \
  BM_Identity_kind(T, name, in_shape, DEVICE, 8);  \

#define BM_Identity_DT(name, in_shape)            \
  BM_Identity_NTH(float, name, in_shape, cpu);    \
  BM_Identity_NTH(bfloat16, name, in_shape, cpu); \

#define BM_IdentityND(name, ...)                    \
  BM_Identity_DT(name, TensorShape({__VA_ARGS__})); \

BM_IdentityND(_2D_1024x1024, 1024, 1024);
BM_IdentityND(_3D_1024x1024x1024, 1024, 1024, 1024);
BM_IdentityND(_4D_1024x1024x1024x1024, 1024, 1024, 1024, 1024);

}  // namespace tensorflow

#endif  // INTEL_MKL
