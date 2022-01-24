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

#include <vector>
#include "mkldnn.hpp"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

//----------------------------------------------------------------------------//
// Concat Unit Tests are below.                                               //
//----------------------------------------------------------------------------//

namespace MKLConcatTestDefs {
    typedef std::tuple<
        string,                         // concatMklOp
        DataType,                       // input_type
        long long int,                  // num_input
        std::vector<long long int>,     // sizes
        long long int                   // axis
    > ConcatTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
        DataType::DT_BFLOAT16
    };
    std::vector<string> concatMklOp = {"_MklConcat", "_MklConcatV2"};
    std::vector<long long int> numInputs = {2, 4};
    std::vector<long long int> AXIS_2D = {0, 1};
    std::vector<long long int> AXIS_3D = {0, 1, 2};
    std::vector<long long int> AXIS_4D = {0, 1, 2, 3};
    std::vector<std::vector<long long int>> SIZES_2D = {{64, 64}, {1, 1}, {32, 21}};
    std::vector<std::vector<long long int>> SIZES_3D = {{32, 16, 1}, {128, 128, 128}, {1, 1, 1}};
    std::vector<std::vector<long long int>> SIZES_4D = {{32, 32, 32, 32}, {16, 1, 1, 1}, {31, 63, 15, 7}};
} // namespace MKLConcatTestDefs

using namespace MKLConcatTestDefs;
class ConcatTestBase :
    public ::testing::WithParamInterface<MKLConcatTestDefs::ConcatTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    string concat_MklOp;
    DataType input_type;
    long long int num_inputs;
    std::vector<long long int> input_size;
    long long int ax;
    // Test input Tensors (filled in SetUp)
    std::vector<Tensor> inputs;
    Tensor zeros;
    Tensor axis;
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
        std::vector<Input> in_values;
        for (int i = 0; i < num_inputs; ++i) {
            const string input_name = absl::StrCat("input_", i);
            auto tmp = ops::Const(root.WithOpName(input_name), Input::Initializer(inputs[i]));
            in_values.push_back(tmp);
        }
        auto a = ops::Const(root.WithOpName("axis"), ax);
        Output next_op = ops::Concat(root.WithOpName("concat"), absl::Span<const Input>(in_values), a);
        string last_op = "concat";
        RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
        if (concat_MklOp == "_MklConcat") {
            TF_EXPECT_OK(
                NodeDefBuilder("mkl_concat_op", "_MklConcat") //build node
                    .Attr("N", num_inputs)
                    .Input(FakeInput(DT_INT32))
                    .Input(FakeInput(input_type))
                    .Input(FakeInput(DT_UINT8))
                    .Input(FakeInput(DT_UINT8))
                    .Attr("_kernel", "MklLayoutDependentOp")
                    .Finalize(node_def()));
        } else if (concat_MklOp == "_MklConcatV2") {
            TF_EXPECT_OK(
                NodeDefBuilder("mkl_concat_op", "_MklConcatV2") //build node
                    .Attr("N", num_inputs)
                    .Input(FakeInput(input_type))
                    .Input(FakeInput(DT_INT32))
                    .Input(FakeInput(DT_UINT8))
                    .Input(FakeInput(DT_UINT8))
                    .Attr("_kernel", "MklLayoutDependentOp")
                    .Finalize(node_def()));
        } else {
            GTEST_FAIL() << "Incorrect Op";
        }
        TF_EXPECT_OK(InitOp()); //initial

        if (concat_MklOp == "_MklConcat") {
            AddInputFromArray<int32>(axis.shape(), axis.flat<int32>()); // axis
        }

        for (uint i = 0; i < num_inputs; ++i){
            switch(input_type) {
                case DT_FLOAT:
                    AddInputFromArray<float>(inputs[i].shape(), inputs[i].flat<float>()); // input
                    break;
                case DT_BFLOAT16:
                    AddInputFromArray<Eigen::bfloat16>(inputs[i].shape(), inputs[i].flat<Eigen::bfloat16>()); // input
                    break;
                default:
                    GTEST_FAIL() << "Unexpected DataType";
            }
        }

        if (concat_MklOp == "_MklConcatV2") {
            AddInputFromArray<int32>(axis.shape(), axis.flat<int32>()); // axis
        }

        for (uint i = 0; i < num_inputs; ++i){
            AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // mkl
        }
        AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>()); // mkl

        TF_EXPECT_OK(RunOpKernel()); //Run the node computation
        mkl_values = *GetOutput(0); //Get output
    }

 public:
    static std::string getTestCaseName(::testing::TestParamInfo<ConcatTestParams> obj) {
        string concat_MklOp;
        DataType input_type;
        long long int num_inputs;
        std::vector<long long int> input_size;
        long long int ax;
        std::tie(concat_MklOp, input_type, num_inputs, input_size, ax) = obj.param;
        std::ostringstream result;
        result << "Concat_" << concat_MklOp << "_Type_";
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

        result << "_NumInputs_" << num_inputs;

        result << "_InputSizes";
        for (auto &x : input_size) {
            result << "_" << x;
        }

        result << "_Axis_" << ax;
        return result.str();
    }

    void SetUp() {
        std::tie(concat_MklOp, input_type, num_inputs, input_size, ax) = this->GetParam();
        inputs = {};
        for (uint i = 0; i < num_inputs; ++i){
            Tensor input = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size())));
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
            inputs.push_back(input);
        }
        axis = Tensor((int32)ax);

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

TEST_P(ConcatTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(Concat2D, ConcatTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(concatMklOp),
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_2D),
        ::testing::ValuesIn(AXIS_2D)),
    ConcatTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Concat3D, ConcatTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(concatMklOp),
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_3D),
        ::testing::ValuesIn(AXIS_3D)),
    ConcatTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Concat4D, ConcatTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(concatMklOp),
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_4D),
        ::testing::ValuesIn(AXIS_3D)),
    ConcatTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Concat(const string& kind, int num_inputs,
		const TensorShape& in_shape, int concat_dims) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Concat" : "_MklConcat";

  Tensor concat_dim(DT_INT32, TensorShape({}));
  concat_dim.scalar<int32>()() = concat_dims;

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<NodeBuilder::NodeOut> inputs_not_mkl;
  inputs.reserve(num_inputs);
  inputs_not_mkl.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    Tensor in(type, in_shape);
    in.flat<T>().setRandom();
    inputs.push_back(test::graph::Constant(g, in));
    inputs_not_mkl.push_back(test::graph::Constant(g, GetMklMetaTensor(), "not_mkl"));
  }

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, concat_dim))
                    .Input(inputs)
                    .Attr("N", num_inputs)
                    .Attr("T", type);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
	                               .Input(inputs_not_mkl)
				       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define S_TENSOR(...) test::AsTensor<int32>({__VA_ARGS__})

#define BM_Concat_Base(kind, name, NI, in_shape, CD, T, DEVICE, NTH)                    \
  static void BM_Concat##_##kind##_##NI##name##_##T##_##CD##_##DEVICE##_##NTH(          \
      int iters) {                                                                      \
    int64 num_elements = in_shape.num_elements();                                       \
    testing::UseRealTime();                                                             \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements * NI * sizeof(T)); \
    SessionOptions opts;                                                                \
    opts.config.set_intra_op_parallelism_threads(NTH);                                  \
    test::Benchmark(#DEVICE, Concat<T>(#kind, NI, in_shape, CD), &opts).Run(iters);     \
  }                                                                                     \
  BENCHMARK(BM_Concat##_##kind##_##NI##name##_##T##_##CD##_##DEVICE##_##NTH);           \

#define BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, NTH)     \
  BM_Concat_Base(Default, name, NI, in_shape, CD, T, DEVICE, NTH); \
  BM_Concat_Base(Mkl, name, NI, in_shape, CD, T, DEVICE, NTH);     \

#define BM_Concat_NTH(name, NI, in_shape, CD, T, DEVICE) \
  BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, 1);  \
  BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, 4);  \
  BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, 8);  \

#define BM_Concat_DT(name, NI, in_shape, CD)             \
  BM_Concat_NTH(name, NI, in_shape, CD, float, cpu);    \
  BM_Concat_NTH(name, NI, in_shape, CD, bfloat16, cpu); \

#define BM_ConcatND(name, NI, ...)                       \
  BM_Concat_DT(name, NI, TensorShape({__VA_ARGS__}), 0); \

// dims == 2
BM_ConcatND(_2Dx2x32x32, 2, 32, 32)
BM_ConcatND(_2Dx2x32x256, 2, 32, 256)
BM_ConcatND(_2Dx2x32x2048, 2, 32, 2048)
BM_ConcatND(_2Dx2x256x32, 2, 256, 32)
BM_ConcatND(_2Dx2x2048x32, 2, 2048, 32)
BM_ConcatND(_2Dx2x256x256, 2, 256, 256)
BM_ConcatND(_2Dx2x2048x2048, 2, 2048, 2048)

BM_ConcatND(_2Dx8x32x32, 8, 32, 32)
BM_ConcatND(_2Dx8x32x256, 8, 32, 256)
BM_ConcatND(_2Dx8x32x2048, 8, 32, 2048)
BM_ConcatND(_2Dx8x256x32, 8, 256, 32)
BM_ConcatND(_2Dx8x2048x32, 8, 2048, 32)
BM_ConcatND(_2Dx8x256x256, 8, 256, 256)
BM_ConcatND(_2Dx8x2048x2048, 8, 2048, 2048)

BM_ConcatND(_2Dx32x32x32, 32, 32, 32)
BM_ConcatND(_2Dx32x32x256, 32, 32, 256)
BM_ConcatND(_2Dx32x32x2048, 32, 32, 2048)
BM_ConcatND(_2Dx32x256x32, 32, 256, 32)
BM_ConcatND(_2Dx32x2048x32, 32, 2048, 32)
BM_ConcatND(_2Dx32x256x256, 32, 256, 256)
BM_ConcatND(_2Dx32x2048x2048, 32, 2048, 2048)

// dims == 3
BM_ConcatND(_3Dx2x32x32x32, 2, 32, 32, 32)
BM_ConcatND(_3Dx2x32x32x256, 2, 32, 32, 256)
BM_ConcatND(_3Dx2x32x32x2048, 2, 32, 32, 2048)
BM_ConcatND(_3Dx2x32x256x32, 2, 32, 256, 32)
BM_ConcatND(_3Dx2x32x2048x32, 2, 32, 2048, 32)
BM_ConcatND(_3Dx2x32x256x256, 2, 32, 256, 256)
BM_ConcatND(_3Dx2x32x2048x2048, 2, 32, 2048, 2048)
BM_ConcatND(_3Dx2x256x32x32, 2, 256, 32, 32)
BM_ConcatND(_3Dx2x256x32x2048, 2, 256, 32, 2048)
BM_ConcatND(_3Dx2x256x2048x32, 2, 256, 2048, 32)
BM_ConcatND(_3Dx2x256x256x256, 2, 256, 256, 256)

BM_ConcatND(_3Dx8x32x32x32, 8, 32, 32, 32)
BM_ConcatND(_3Dx8x32x32x256, 8, 32, 32, 256)
BM_ConcatND(_3Dx8x32x32x2048, 8, 32, 32, 2048)
BM_ConcatND(_3Dx8x32x256x32, 8, 32, 256, 32)
BM_ConcatND(_3Dx8x32x2048x32, 8, 32, 2048, 32)
BM_ConcatND(_3Dx8x32x256x256, 8, 32, 256, 256)
BM_ConcatND(_3Dx8x32x2048x2048, 8, 32, 2048, 2048)
BM_ConcatND(_3Dx8x256x32x32, 8, 256, 32, 32)
BM_ConcatND(_3Dx8x256x32x2048, 8, 256, 32, 2048)
BM_ConcatND(_3Dx8x256x2048x32, 8, 256, 2048, 32)
BM_ConcatND(_3Dx8x256x256x256, 8, 256, 256, 256)

BM_ConcatND(_3Dx32x32x32x32, 32, 32, 32, 32)
BM_ConcatND(_3Dx32x32x32x256, 32, 32, 32, 256)
BM_ConcatND(_3Dx32x32x32x2048, 32, 32, 32, 2048)
BM_ConcatND(_3Dx32x32x256x32, 32, 32, 256, 32)
BM_ConcatND(_3Dx32x32x2048x32, 32, 32, 2048, 32)
BM_ConcatND(_3Dx32x32x256x256, 32, 32, 256, 256)
BM_ConcatND(_3Dx32x32x2048x2048, 32, 32, 2048, 2048)
BM_ConcatND(_3Dx32x256x32x32, 32, 256, 32, 32)
BM_ConcatND(_3Dx32x256x32x2048, 32, 256, 32, 2048)
BM_ConcatND(_3Dx32x256x2048x32, 32, 256, 2048, 32)
BM_ConcatND(_3Dx32x256x256x256, 32, 256, 256, 256)

// dims == 4
BM_ConcatND(_4Dx2x32x32x32x32, 2, 32, 32, 32, 32)
BM_ConcatND(_4Dx2x256x256x256x256, 2, 256, 256, 256, 256)

BM_ConcatND(_4Dx8x32x32x32x32, 8, 32, 32, 32, 32)
BM_ConcatND(_4Dx8x256x256x256x256, 8, 256, 256, 256, 256)

BM_ConcatND(_4Dx32x32x32x32x32, 32, 32, 32, 32, 32)
BM_ConcatND(_4Dx32x256x256x256x256, 32, 256, 256, 256, 256)

}  // namespace tensorflow

#endif  // INTEL_MKL
