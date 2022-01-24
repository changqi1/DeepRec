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
#include "mkldnn.hpp"
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
namespace tensorflow {
//----------------------------------------------------------------------------//
// Relu Unit Tests are below.                                                 //
//----------------------------------------------------------------------------//
namespace MKLPermuteTestDefs {
    typedef std::tuple<
    DataType,             // input_type
    std::vector<long long int>, // input_size
    std::vector<int32_t>  // permute_order
    > PermuteTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
        DataType::DT_BFLOAT16
    };
    std::vector<std::vector<long long int>> SIZES_2D = {{64, 64}, {1,1}, {1,32}, {32, 21}};
    std::vector<std::vector<int32_t>> PERMUTES_2D = {{1, 0}};
    std::vector<std::vector<long long int>> SIZES_3D = {{32, 16, 1}, {32, 32, 32}, {128, 128, 128}, {1, 1, 1}};
    std::vector<std::vector<int32_t>> PERMUTES_3D = {{0, 2, 1}, {2, 0, 1}, {1, 2, 0}};
    std::vector<std::vector<long long int>> SIZES_4D = {{32, 32, 32, 32}, {16, 1, 1, 1}, {31, 63, 15, 7}};
    std::vector<std::vector<int32_t>> PERMUTES_4D = {{3, 1, 2, 0}, {2, 3, 1, 0}, {3, 2, 1, 0}, {1, 0, 2, 3}};
} // namespace MKLPremuteTestDefs

using namespace MKLPermuteTestDefs;
class PermuteTestBase : 
    public ::testing::WithParamInterface<MKLPermuteTestDefs::PermuteTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    DataType input_type;
    std::vector<long long int> input_size;
    std::vector<int32_t> permute_order;
    // Test input Tensors (filled in SetUp)
    Tensor input;
    Tensor perm_tensor;
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
          Output input_op =
              ops::Const(root.WithOpName("input"), Input::Initializer(input));
          Output perm_input =
              ops::Const(root.WithOpName("perm_tensor"), Input::Initializer(perm_tensor));
          Output next_op = ops::Transpose(root.WithOpName("transposeop"), input_op, perm_input);
          string last_op = "transposeop";
          RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
    TF_EXPECT_OK(
        NodeDefBuilder("mkl_transpose_op", "_MklTranspose") //build node
            .Input(FakeInput(input_type))
            .Input(FakeInput(DT_INT32))
            .Attr("_kernel", "MklNameChangeOp")
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
    AddInputFromArray<int32_t>(perm_tensor.shape(), perm_tensor.flat<int32_t>());
    TF_EXPECT_OK(RunOpKernel()); //Run the node computation
    mkl_values = *GetOutput(0); //Get output
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<PermuteTestParams> obj) {
        DataType input_type;
        std::vector<long long int> input_size;
        std::vector<int32_t> permute_order;
        std::tie(input_type, input_size, permute_order) = obj.param;
        std::ostringstream result;
        result << "Transpose_Type_";
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
        result << "_PermuteOrder";
        for (auto &x : permute_order) {
            result << "_" << x;
        }
        return result.str();
    }

    void SetUp() {
        std::tie(input_type, input_size, permute_order) = this->GetParam();
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
        perm_tensor = Tensor(DT_INT32, TensorShape({permute_order.size()}));
        for (int i=0; i < permute_order.size(); i++) {
            perm_tensor.vec<int32_t>()(i) = permute_order[i];
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
TEST_P(PermuteTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(Transpose2D, PermuteTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_2D),
        ::testing::ValuesIn(PERMUTES_2D)),
    PermuteTestBase::getTestCaseName);
INSTANTIATE_TEST_CASE_P(Transpose3D, PermuteTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_3D),
        ::testing::ValuesIn(PERMUTES_3D)),
    PermuteTestBase::getTestCaseName);
INSTANTIATE_TEST_CASE_P(Transpose4D, PermuteTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_4D),
        ::testing::ValuesIn(PERMUTES_4D)),
    PermuteTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Transpose(const string& kind, const TensorShape& in_shape, const Tensor& perm_tensor) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();
  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Transpose" : "_MklTranspose";
  // Create inputs
  Tensor input1(type, in_shape);
  input1.flat<T>().setRandom();
  Node* input_in0 = test::graph::Constant(g, input1);
  Node* input_in1 = test::graph::Constant(g, perm_tensor);
  // Create NodeDef
  auto nodeBuilder = NodeBuilder(g->NewName("transpose"), op_name)
                  .Input(input_in0)
                  .Input(input_in1);
  isDefault ? nodeBuilder : nodeBuilder.Attr("_kernel", "MklNameChangeOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}
#define S_TENSOR(size, ...) test::AsTensor<int32>({__VA_ARGS__}, {size})

#define BM_Transpose_Base(T, kind, name, in_shape, perm_tensor, DEVICE, NTH)                \
  static void BM_Transpose_##T##_##kind##name##_##NTH(int iters) {                          \
    int64 num_elements = in_shape.num_elements();                                           \
    testing::UseRealTime();                                                                 \
    testing::BytesProcessed(static_cast<int64>(iters) * num_elements * sizeof(T));          \
    SessionOptions opts;                                                                    \
    opts.config.set_intra_op_parallelism_threads(NTH);                                      \
    test::Benchmark(#DEVICE, Transpose<T>(#kind, in_shape, perm_tensor), &opts).Run(iters); \
  }                                                                                         \
  BENCHMARK(BM_Transpose_##T##_##kind##name##_##NTH);                                       \

#define BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, NTH)     \
  BM_Transpose_Base(T, Default, name, in_shape, perm_tensor, DEVICE, NTH); \
  BM_Transpose_Base(T, MKL, name, in_shape, perm_tensor, DEVICE, NTH);     \

#define BM_Transpose_NTH(T, name, in_shape, perm_tensor, DEVICE) \
  BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, 1);  \
  BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, 4);  \
  BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, 8);  \

#define BM_Transpose_DT(name, in_shape, perm_tensor)            \
  BM_Transpose_NTH(float, name, in_shape, perm_tensor, cpu);    \
  BM_Transpose_NTH(bfloat16, name, in_shape, perm_tensor, cpu); \

#define BM_Transpose2D(name, A, B, size, ...)                                   \
  BM_Transpose_DT(_2D##name, TensorShape({A, B}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Transpose3D(name, A, B, C, size, ...)                                   \
  BM_Transpose_DT(_3D##name, TensorShape({A, B, C}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Transpose4D(name, A, B, C, D, size, ...)                                   \
  BM_Transpose_DT(_4D##name, TensorShape({A, B, C, D}), S_TENSOR(size, __VA_ARGS__)); \

BM_Transpose2D(_128x512, 128, 512, 2, 1, 0);
BM_Transpose2D(_128x1024, 128, 1024, 2, 1, 0);
BM_Transpose2D(_128x2048, 128, 2048, 2, 1, 0);
BM_Transpose2D(_128x4096, 128, 4096, 2, 1, 0);
BM_Transpose2D(_512x128, 512, 128, 2, 1, 0);
BM_Transpose2D(_1024x128, 1024, 128, 2, 1, 0);
BM_Transpose2D(_2048x128, 2048, 128, 2, 1, 0);
BM_Transpose2D(_4096x128, 4096, 128, 2, 1, 0);
BM_Transpose2D(_128x128, 128, 128, 2, 1, 0);
BM_Transpose2D(_512x512, 512, 512, 2, 1, 0);
BM_Transpose2D(_1024x1024, 1024, 1024, 2 , 1, 0);
BM_Transpose2D(_2048x2048, 2048, 2048, 2 , 1, 0);
BM_Transpose2D(_4096x4096, 4096, 4096, 2 , 1, 0);
BM_Transpose3D(_128x128x128, 128, 128, 128, 3, 0, 2, 1);
BM_Transpose3D(_256x256x256, 256, 256, 256, 3, 0, 2, 1);
BM_Transpose3D(_512x512x512, 512, 512, 512, 3, 0, 2, 1);
BM_Transpose4D(_128x128x128x128, 128, 128, 128, 128, 4, 3, 1, 2, 0);
}  // namespace tensorflow
#endif  // INTEL_MKL