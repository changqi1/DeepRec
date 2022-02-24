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
#include "tensorflow/core/util/mkl_util.h"
#include <gtest/gtest.h>

namespace tensorflow {

//----------------------------------------------------------------------------//
// Slice Unit Tests are below.                                                //
//----------------------------------------------------------------------------//

namespace MKLSliceTestDefs {
    typedef std::tuple<
    DataType,             // input_type
    std::vector<long long int>
    > SliceTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
        DataType::DT_BFLOAT16
    };
    std::vector<std::vector<long long int>> INPUT = {{200, 15000}};
} // namespace MKLSliceTestDefs

using namespace MKLSliceTestDefs;
class SliceTestBase :
    public ::testing::WithParamInterface<MKLSliceTestDefs::SliceTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    DataType input_type;
    std::vector<long long int> input_size;
    // Test input Tensors (filled in SetUp)
    Tensor input;
    Tensor begin;
    Tensor sizes;
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
          auto begin_op =
              ops::Const(root.WithOpName("begin"), Input::Initializer(begin));
          auto sizes_op =
              ops::Const(root.WithOpName("sizes"), Input::Initializer(sizes));
          Output next_op = ops::Slice(root.WithOpName("slice"), input_op, begin_op, sizes_op);
          string last_op = "slice";
          RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
    TF_EXPECT_OK(
        NodeDefBuilder("mkl_slice_op", "_MklSlice") //build node
            .Input(FakeInput(input_type))
            .Input(FakeInput(DT_INT32))
            .Input(FakeInput(DT_INT32))
            .Input(FakeInput(DT_UINT8))
            .Input(FakeInput(DT_UINT8))
            .Input(FakeInput(DT_UINT8))
            .Attr("_kernel", "MklLayoutDependentOp")
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp()); //initial
    switch(input_type) {
        case DT_FLOAT:
            AddInputFromArray<float>(input.shape(), input.flat<float>()); // input
            break;
        case DT_BFLOAT16:
            AddInputFromArray<Eigen::bfloat16>(input.shape(), input.flat<Eigen::bfloat16>()); // input
            break;
        default:
            GTEST_FAIL() << "Unexpected DataType";
    }
    AddInputFromArray<int32>(begin.shape(), begin.flat<int32>()); // begin
    AddInputFromArray<int32>(sizes.shape(), sizes.flat<int32>()); // sizes
    AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // mkl
    AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // mkl
    AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // mkl
    TF_EXPECT_OK(RunOpKernel()); //Run the node computation
    mkl_values = *GetOutput(0); //Get output
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<SliceTestParams> obj) {
        DataType input_type;
        std::vector<long long int> input_size;
        std::tie(input_type, input_size) = obj.param;
        std::ostringstream result;
        result << "Slice_Type_";
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
        result << "_Input";
        for (auto &x : input_size) {
            result << "_" << x;
        }
        result << "_Begin_10_10";
        result << "_Size_100_100";
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
                input.flat<Eigen::bfloat16>() = input.flat<Eigen::bfloat16>().template setRandom<Eigen::internal::UniformRandomGenerator<Eigen::bfloat16>>(); // input
		        input.flat<Eigen::bfloat16>() = input.flat<Eigen::bfloat16>() - input.flat<Eigen::bfloat16>().constant((Eigen::bfloat16)0.5);
		        input.flat<Eigen::bfloat16>() = input.flat<Eigen::bfloat16>() * input.flat<Eigen::bfloat16>().constant((Eigen::bfloat16)200.0);
                break;
            default:
                GTEST_FAIL() << "Unexpected DataType";
        }
        begin = Tensor(DT_INT32, TensorShape({2}));
        begin.vec<int32>()(0) = 10;
        begin.vec<int32>()(1) = 10;

        sizes = Tensor(DT_INT32, TensorShape({2}));
        sizes.vec<int32>()(0) = 100;
        sizes.vec<int32>()(1) = 100;

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

TEST_P(SliceTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(Slice, SliceTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(INPUT)),
    SliceTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Slice2D(const string& kind, DataType type, int size) {
  Graph* g = new Graph(OpRegistry::Global());

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Slice" : "_MklSlice";

  int kDim = 100;
  int kMaxSize = 15000;
  CHECK_LT(size, kMaxSize);

  Tensor input(type, TensorShape({2 * kDim, kMaxSize}));
  input.flat<T>().setRandom();

  Tensor begin(DT_INT32, TensorShape({2}));
  begin.flat<int32>()(0) = 10;
  begin.flat<int32>()(1) = 10;

  Tensor sizes(DT_INT32, TensorShape({2}));
  sizes.flat<int32>()(0) = kDim;
  sizes.flat<int32>()(1) = size;

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, input))
                    .Input(test::graph::Constant(g, begin))
                    .Input(test::graph::Constant(g, sizes))
                    .Attr("T", type);
  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
	                               .Input(not_mkl_shape)
	                               .Input(not_mkl_shape)
				       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_Slice2D_Base(kind, size, T, TFTYPE, DEVICE, NTH)                        \
  static void BM_Slice2D##_##kind##_##size##_##T##_##TFTYPE##_##DEVICE##_##NTH(    \
      int iters) {                                                                 \
    testing::UseRealTime();                                                        \
    testing::BytesProcessed(static_cast<int64>(iters) * 100 * size * sizeof(T));   \
    SessionOptions opts;                                                           \
    opts.config.set_intra_op_parallelism_threads(NTH);                             \
    test::Benchmark(#DEVICE, Slice2D<T>(#kind, TFTYPE, size), &opts).Run(iters);   \
  }                                                                                \
  BENCHMARK(BM_Slice2D##_##kind##_##size##_##T##_##TFTYPE##_##DEVICE##_##NTH);     \

#define BM_Slice2D_kind(size, T, TFTYPE, DEVICE, NTH)     \
  BM_Slice2D_Base(Default, size, T, TFTYPE, DEVICE, NTH); \
  BM_Slice2D_Base(Mkl, size, T, TFTYPE, DEVICE, NTH);     \

#define BM_Slice2D_NTH(size, T, TFTYPE, DEVICE) \
  BM_Slice2D_kind(size, T, TFTYPE, DEVICE, 1);  \
  BM_Slice2D_kind(size, T, TFTYPE, DEVICE, 4);  \
  BM_Slice2D_kind(size, T, TFTYPE, DEVICE, 8);  \

#define BM_Slice2D_DT(size)                         \
  BM_Slice2D_NTH(size, float, DT_FLOAT, cpu);       \
  BM_Slice2D_NTH(size, bfloat16, DT_BFLOAT16, cpu); \

#define BM_Slice2D(size) \
  BM_Slice2D_DT(size)    \

BM_Slice2D(100);
BM_Slice2D(1000);
BM_Slice2D(10000);

}  // namespace tensorflow

#endif  // INTEL_MKL
