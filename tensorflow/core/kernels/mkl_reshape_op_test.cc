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
// Reshape Tests are below.                                                   //
//----------------------------------------------------------------------------//

namespace MKLReshapeTestDefs {
    typedef std::tuple<
    DataType,                   // input_type
    std::vector<long long int>, // input_size_0
    std::vector<long long int> // input_size_1
    > ReshapeTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
        DataType::DT_BFLOAT16
    };
    std::vector<std::vector<long long int>> SIZES_2D_IN = {{1024, 1024}};
    std::vector<std::vector<long long int>> SIZES_2D_SHAPE = {{256, 4096}};
    std::vector<std::vector<long long int>> SIZES_3D_IN = {{128, 128, 64}};
    std::vector<std::vector<long long int>> SIZES_3D_SHAPE = {{16, 256, 256}, {256, 128, 32}};
    std::vector<std::vector<long long int>> SIZES_4D_IN = {{128, 128, 16, 4}};
} // namespace MKLReshapeTestDefs

using namespace MKLReshapeTestDefs;
class ReshapeTestBase :
    public ::testing::WithParamInterface<MKLReshapeTestDefs::ReshapeTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    DataType input_type;
    std::vector<long long int> input;
    std::vector<long long int> shape_input;
    // Test input Tensors (filled in SetUp)
    Tensor input_tensor;
    Tensor shape_tensor;
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
          auto input_0 =
              ops::Const(root.WithOpName("input"), Input::Initializer(input_tensor));
          auto input_shape_0 =
              ops::Const(root.WithOpName("shape"), Input::Initializer(shape_tensor));
          Output next_op = ops::Reshape(root.WithOpName("reshape"), input_0, input_shape_0);
          string last_op = "reshape";
          RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
	TF_EXPECT_OK(
          NodeDefBuilder("mkl_reshape_op", "_MklReshape") //build node
              .Input(FakeInput(input_type))
              .Input(FakeInput(DT_INT64))
              .Input(FakeInput(DT_UINT8))
              .Input(FakeInput(DT_UINT8))
              .Attr("_kernel", "MklLayoutDependentOp")
              .Finalize(node_def()));
      TF_EXPECT_OK(InitOp()); //initial
      switch(input_type) {
          case DT_FLOAT:
              AddInputFromArray<float>(input_tensor.shape(), input_tensor.flat<float>()); // input_0
              break;
          case DT_BFLOAT16:
              AddInputFromArray<bfloat16>(input_tensor.shape(), input_tensor.flat<bfloat16>()); // input_0
              break;
          default:
              GTEST_FAIL() << "Unexpected DataType";
      }
      AddInputFromArray<long long int>(shape_tensor.shape(), shape_tensor.flat<long long int>());
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // input
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // input
      TF_EXPECT_OK(RunOpKernel()); //Run the node computation
      mkl_values = *GetOutput(0); //Get output
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<ReshapeTestParams> obj) {
        DataType input_type;
        std::vector<long long int> input;
        std::vector<long long int> shape_input;
        std::tie(input_type, input, shape_input) = obj.param;
        std::ostringstream result;
        result << "BatchMatMul_Type_";
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
        for (auto &x : input) {
            result << "_" << x;
        }
        result << "_Shapes";
        for (auto &x : shape_input) {
            result << "_" << x;
        }
        return result.str();
    }

    void SetUp() {
      std::tie(input_type, input, shape_input) = this->GetParam();
      input_tensor = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input.data(), input.size())));
      shape_tensor = Tensor(DT_INT64, TensorShape({shape_input.size()}));
      switch(input_type) {
          case DT_FLOAT:
            input_tensor.flat<float>() = input_tensor.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>(); // input
	    break;
          case DT_BFLOAT16:
            input_tensor.flat<bfloat16>() = input_tensor.flat<bfloat16>().template setRandom<Eigen::internal::UniformRandomGenerator<bfloat16>>(); // input_0
            input_tensor.flat<bfloat16>() = input_tensor.flat<bfloat16>() - input_tensor.flat<bfloat16>().constant((bfloat16)0.5);
            input_tensor.flat<bfloat16>() = input_tensor.flat<bfloat16>() * input_tensor.flat<bfloat16>().constant((bfloat16)200.0);
      	  break;
        default:
          GTEST_FAIL() << "Unexpected DataType";
      }

      for(int i = 0; i < shape_input.size(); i++){
          shape_tensor.flat<long long int>()(i) = shape_input[i];
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

TEST_P(ReshapeTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(Reshape2D_2D, ReshapeTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_2D_IN),
        ::testing::ValuesIn(SIZES_2D_SHAPE)),
    ReshapeTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Reshape2D_3D, ReshapeTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_2D_IN),
        ::testing::ValuesIn(SIZES_3D_SHAPE)),
    ReshapeTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Reshape3D_3D, ReshapeTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_3D_IN),
        ::testing::ValuesIn(SIZES_3D_SHAPE)),
    ReshapeTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Reshape4D_3D, ReshapeTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_4D_IN),
        ::testing::ValuesIn(SIZES_3D_SHAPE)),
    ReshapeTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Reshape(const string& kind, const TensorShape& in_shape, const Tensor& shape_tensor) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Reshape" : "_MklReshape";

  Tensor input(type, in_shape);
  input.flat<T>().setRandom();

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, input))
                    .Input(test::graph::Constant(g, shape_tensor))
                    .Attr("T", type);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}

// Constructs a tensor of "shape" with values "vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals, const TensorShape& shape) {
  Tensor ret;
  CHECK(ret.CopyFrom(AsTensor(vals), shape));
  return ret;
}

#define S_TENSOR(size, ...) test::AsTensor<int32>({__VA_ARGS__}, {size})

#define BM_Reshape_Base(kind, T, name, in_shape, shape_tensor, DEVICE, NTH)                \
  static void BM_Reshape##_##kind##_##T##name##_##DEVICE##_##NTH(                          \
      int iters) {                                                                         \
    int64 num_elements = in_shape.num_elements();  	                                   \
    testing::UseRealTime();                                                                \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements);                     \
    SessionOptions opts;                                                                   \
    opts.config.set_intra_op_parallelism_threads(NTH);                                     \
    test::Benchmark(#DEVICE, Reshape<T>(#kind, in_shape, shape_tensor), &opts).Run(iters); \
  }                                                                                        \
  BENCHMARK(BM_Reshape##_##kind##_##T##name##_##DEVICE##_##NTH);                           \

#define BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, NTH)     \
  BM_Reshape_Base(Default, T, name, in_shape, shape_tensor, DEVICE, NTH); \
  BM_Reshape_Base(Mkl, T, name, in_shape, shape_tensor, DEVICE, NTH);     \

#define BM_Reshape_NTH(T, name, in_shape, shape_tensor, DEVICE) \
  BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, 1);  \
  BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, 4);  \
  BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, 8);  \

#define BM_Reshape_DT(name, in_shape, shape_tensor)             \
  BM_Reshape_NTH(float, name, in_shape, shape_tensor, cpu);    \
  BM_Reshape_NTH(bfloat16, name, in_shape, shape_tensor, cpu); \

#define BM_Reshape2D(name, A, B, size, ...)                                   \
  BM_Reshape_DT(_2D##name, TensorShape({A, B}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Reshape3D(name, A, B, C, size, ...)                                   \
  BM_Reshape_DT(_3D##name, TensorShape({A, B, C}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Reshape4D(name, A, B, C, D, size, ...)                                   \
  BM_Reshape_DT(_4D##name, TensorShape({A, B, C, D}), S_TENSOR(size, __VA_ARGS__)); \

BM_Reshape2D(_1024x1024_To_256x4096, 1024, 1024, 2, 256, 4096);
BM_Reshape2D(_1024x1024_To_16x256x256, 1024, 1024, 3, 16, 256, 256);

BM_Reshape3D(_128x128x256_To_256x128x128, 128, 128, 256, 3, 256, 128, 128);

BM_Reshape4D(_128x128x16x16_To_256x128x128, 128, 128, 16, 16, 3, 256, 128, 128);

}  // namespace tensorflow

#endif  // INTEL_MKL
