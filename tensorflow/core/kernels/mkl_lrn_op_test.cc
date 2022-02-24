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
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "absl/algorithm/container.h"
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
#include <mutex>

namespace tensorflow {

//----------------------------------------------------------------------------//
// LRN Tests are below.                                                       //
//----------------------------------------------------------------------------//

namespace MKLLRNTestDefs {
    typedef std::tuple<
    DataType,                   // input_type
    std::vector<long long int>, // input_size_0
    std::vector<long long int>
    > LRNTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
    };
    std::vector<std::vector<long long int>> SIZES_4D = {{32, 32, 32, 32}, {32, 64, 32, 128}, {128, 32, 16, 64}};
    std::vector<std::vector<long long int>> DR = {{4}, {2}};
} // namespace LRNTestDefs

using namespace MKLLRNTestDefs;
class LRNTestBase :
    public ::testing::WithParamInterface<MKLLRNTestDefs::LRNTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    DataType input_type;
    std::vector<long long int> input_size;
    std::vector<long long int> depth_radius;
    // Test input Tensors (filled in SetUp)
    Tensor input;
    Tensor zeros;
    // Test attributes (specified in SetUp)
    long long int DR_in;
    // Test output Tensors (filled in Run method)
    Tensor mkl_values;
    Tensor default_values;
    // Srand
    int seed = 2022;

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
      auto attr =
          ops::LRN::DepthRadius(DR_in).Bias(1.0f).Alpha(0.1f).Beta(2.0f);
      Output next_op = ops::LRN(root.WithOpName("LRN"), input_op, attr);
      string last_op = "LRN";
      RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
	    TF_EXPECT_OK(
        NodeDefBuilder("mkl_lrn_op", "_MklLRN") //build node
            .Input(FakeInput(input_type))
            .Input(FakeInput(DT_UINT8))
            .Attr("T", input_type)
            .Attr("depth_radius", DR_in)
            .Attr("bias", 1.0f)
            .Attr("alpha", 0.1f)
            .Attr("beta", 2.0f)
            .Attr("_kernel", "MklLayoutDependentOp")
            .Finalize(node_def()));
      TF_EXPECT_OK(InitOp()); //initial
      switch(input_type) {
        case DT_FLOAT:
            AddInputFromArray<float>(input.shape(), input.flat<float>()); // input_0
            break;
        default:
        GTEST_FAIL() << "Unexpected DataType";
      }
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>());
      TF_EXPECT_OK(RunOpKernel()); //Run the node computation
      mkl_values = *GetOutput(0); //Get outp
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<LRNTestParams> obj) {
        DataType input_type;
        std::vector<long long int> input_size;
        std::vector<long long int> depth_radius;
        std::tie(input_type, input_size, depth_radius) = obj.param;
        std::ostringstream result;
        result << "LRN_Type_";
        switch(input_type) {
            case DataType::DT_FLOAT:
                result << "FLOAT";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }
        result << "_Sizes";
        for (auto &x : input_size) {
            result << "_" << x;
        }
        result << "_DR_" << depth_radius[0];
        return result.str();
    }

    void SetUp() {
        std::tie(input_type, input_size, depth_radius) = this->GetParam();
        input = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size())));
        switch(input_type) {
            case DT_FLOAT:
                {
                srand(seed);
                auto begin = input.flat<float>().data();
                auto end = begin + std::accumulate(std::begin(input_size), std::end(input_size),1,std::multiplies<long long int>());
                auto LO = -10.0f;
                auto HI = 10.0f;

                for( auto current = begin; current != end; current++) {
                    *current = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                }
                break;
                }
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
        DR_in = depth_radius[0];
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

TEST_P(LRNTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(LRN, LRNTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_4D),
        ::testing::ValuesIn(DR)),
    LRNTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// LRNGrad Tests are below.                                                   //
//----------------------------------------------------------------------------//

namespace MKLLRNGradTestDefs {
    typedef std::tuple<
    DataType,                   // input_type
    std::vector<long long int>, // input_size_0
    std::vector<long long int>
    > LRNGradTestParams;
    std::vector<DataType> dataTypesGrad {
        DataType::DT_FLOAT,
    };
    std::vector<std::vector<long long int>> SIZES_4D_GRAD = {{32, 32, 32, 32}, {32, 64, 32, 128}, {128, 32, 16, 64}};
    std::vector<std::vector<long long int>> DR_GRAD = {{4}, {2}};
} // namespace LRNTestDefs

using namespace MKLLRNGradTestDefs;
class LRNGradTestBase :
    public ::testing::WithParamInterface<MKLLRNGradTestDefs::LRNGradTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    DataType input_type;
    std::vector<long long int> input_size;
    std::vector<long long int> depth_radius;
    // Test input Tensors (filled in SetUp)
    Tensor input_grad;
    Tensor input_in0;
    Tensor out;
    Tensor zeros;
    // Test attributes (specified in SetUp)
    long long int DR_in;
    // Test output Tensors (filled in Run method)
    Tensor mkl_values;
    Tensor default_values;
    // Srand
    int seed = 2022;

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
      auto input_grad_op =
          ops::Const(root.WithOpName("input_grad"), Input::Initializer(input_grad));
      auto input_in0_op =
          ops::Const(root.WithOpName("input_in0"), Input::Initializer(input_in0));
      auto input_out_op =
          ops::Const(root.WithOpName("input_out"), Input::Initializer(out));
      auto attr =
          ops::internal::LRNGrad::DepthRadius(DR_in).Bias(1.0f).Alpha(0.1f).Beta(2.0f);
      Output next_op = ops::internal::LRNGrad(root.WithOpName("LRNGrad"), input_grad_op, input_in0_op, input_out_op, attr);
      string last_op = "LRNGrad";
      RunAndFetch(root, last_op, &default_values);
    };

    void runMkl() {
	    TF_EXPECT_OK(
        NodeDefBuilder("mkl_lrngrad_op", "_MklLRNGrad") //build node
            .Input(FakeInput(input_type))
            .Input(FakeInput(input_type))
            .Input(FakeInput(DT_FLOAT))
            .Input(FakeInput(DT_UINT8))
            .Input(FakeInput(DT_UINT8))
            .Input(FakeInput(DT_UINT8))
            .Input(FakeInput(DT_UINT8))
            .Input(FakeInput(DT_UINT8))
            .Attr("T", input_type)
            .Attr("depth_radius", DR_in)
            .Attr("bias", 1.0f)
            .Attr("alpha", 0.1f)
            .Attr("beta", 2.0f)
            .Attr("_kernel", "MklLayoutDependentOp")
            .Finalize(node_def()));
      TF_EXPECT_OK(InitOp()); //initial
      switch(input_type) {
        case DT_FLOAT:
            AddInputFromArray<float>(input_grad.shape(), input_grad.flat<float>()); // input_grad
            AddInputFromArray<float>(input_in0.shape(), input_in0.flat<float>()); // input_in0
            break;
        default:
        GTEST_FAIL() << "Unexpected DataType";
      }
      AddInputFromArray<float>(out.shape(), out.flat<float>()); // input_in0
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>());
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>());
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>());
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>());
      AddInputFromArray<uint8_t>(zeros.shape(), zeros.flat<uint8_t>());
      TF_EXPECT_OK(RunOpKernel()); //Run the node computation
      mkl_values = *GetOutput(0); //Get outp
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<LRNGradTestParams> obj) {
        DataType input_type;
        std::vector<long long int> input_size;
        std::vector<long long int> depth_radius;
        std::tie(input_type, input_size, depth_radius) = obj.param;
        std::ostringstream result;
        result << "LRNGrad_Type_";
        switch(input_type) {
            case DataType::DT_FLOAT:
                result << "FLOAT";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }
        result << "_Sizes";
        for (auto &x : input_size) {
            result << "_" << x;
        }
        result << "_DR_" << depth_radius[0];
        return result.str();
    }

    void SetUp() {
        std::tie(input_type, input_size, depth_radius) = this->GetParam();
        input_grad = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size())));
        input_in0 = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size())));
        out = Tensor(DT_FLOAT, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size()))); // out
        switch(input_type) {
            case DT_FLOAT:
                {
                srand(seed);
                auto begin_grad = input_grad.flat<float>().data();
                auto end_grad = begin_grad + std::accumulate(std::begin(input_size), std::end(input_size),1,std::multiplies<long long int>());
                auto begin_in0 = input_in0.flat<float>().data();
                auto end_in0 = begin_in0 + std::accumulate(std::begin(input_size), std::end(input_size),1,std::multiplies<long long int>());
                auto begin_out = out.flat<float>().data();
                auto end_out = begin_out + std::accumulate(std::begin(input_size), std::end(input_size),1,std::multiplies<long long int>());
                auto LO = -10.0f;
                auto HI = 10.0f;

                for( auto current = begin_grad; current != end_grad; current++) {
                    *current = LO + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(HI-LO)));
                }
                for( auto current = begin_in0; current != end_in0; current++) {
                    *current = LO + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(HI-LO)));
                }
                for( auto current = begin_out; current != end_out; current++) {
                    *current = LO + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(HI-LO)));
                }
                break;
                }
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
        DR_in = depth_radius[0];
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

TEST_P(LRNGradTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(LRNGrad, LRNGradTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypesGrad),
        ::testing::ValuesIn(SIZES_4D_GRAD),
        ::testing::ValuesIn(DR_GRAD)),
    LRNGradTestBase::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* LRN(const string& kind, int DR, const TensorShape& in_shape, 
		  float BIAS = 1.0f, float ALPHA = 0.1f,float BETA = 2.0f) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "LRN" : "_MklLRN";

  Tensor in0(type, in_shape);
  in0.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_in0)
                    .Attr("depth_radius", DR)
                    .Attr("bias", BIAS)
                    .Attr("alpha", ALPHA)
                    .Attr("beta", BETA);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_LRN_Base(kind, DR, in_shape, name, T, DEVICE, NTH)                   \
  static void BM_LRN##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH(            \
      int iters) {                                                              \
    int64 num_elements = in_shape.num_elements();                               \
    testing::UseRealTime();                                                     \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements * DR * 4); \
    SessionOptions opts;                                                        \
    opts.config.set_intra_op_parallelism_threads(NTH);                          \
    test::Benchmark(#DEVICE, LRN<T>(#kind, DR, in_shape), &opts).Run(iters);    \
  }                                                                             \
  BENCHMARK(BM_LRN##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH);             \

#define BM_LRN_kind(DR, in_shape, name, T, DEVICE, NTH)     \
  BM_LRN_Base(Default, DR, in_shape, name, T, DEVICE, NTH); \
  BM_LRN_Base(Mkl, DR, in_shape, name, T, DEVICE, NTH);     \

#define BM_LRN_NTH(DR, in_shape, name, T, DEVICE) \
  BM_LRN_kind(DR, in_shape, name, T, DEVICE, 1);  \
  BM_LRN_kind(DR, in_shape, name, T, DEVICE, 4);  \
  BM_LRN_kind(DR, in_shape, name, T, DEVICE, 8);  \

#define BM_LRN_DT(DR, in_shape, name)         \
  BM_LRN_NTH(DR, in_shape, name, float, cpu); \

#define BM_LRN(name, DR, ...)                      \
  BM_LRN_DT(DR, TensorShape({__VA_ARGS__}), name); \

BM_LRN(_128x12x12x64, 4, 128, 12, 12, 64);
BM_LRN(_128x56x56x64, 2, 128, 56, 56, 64);
BM_LRN(_128x27x27x192, 2, 128, 27, 27, 192);

template <typename T>
static Graph* LRNGrad(const string& kind, int DR, const TensorShape& in_shape,
		      float BIAS = 1.0f, float ALPHA = 0.1f, float BETA = 2.0f) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "LRNGrad" : "_MklLRNGrad";

  Tensor inGrad(type, in_shape);
  inGrad.flat<T>().setRandom();

  Tensor in0(type, in_shape);
  in0.flat<T>().setRandom();

  Tensor out(DT_FLOAT, in_shape);

  Node* input_inGrad = test::graph::Constant(g, inGrad);
  Node* input_in0 = test::graph::Constant(g, in0);
  Node* output = test::graph::Constant(g, out);

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_inGrad)
                    .Input(input_in0)
                    .Input(output)
                    .Attr("depth_radius", DR)
                    .Attr("bias", BIAS)
                    .Attr("alpha", ALPHA)
                    .Attr("beta", BETA);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_LRNGrad_Base(kind, DR, in_shape, name, T, DEVICE, NTH)                \
  static void BM_LRNGrad##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH(         \
      int iters) {                                                               \
    int64 num_elements = in_shape.num_elements();                                \
    testing::UseRealTime();                                                      \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements * DR * 4);  \
    SessionOptions opts;                                                         \
    opts.config.set_intra_op_parallelism_threads(NTH);                           \
    test::Benchmark(#DEVICE, LRNGrad<T>(#kind, DR, in_shape), &opts).Run(iters); \
  }                                                                              \
  BENCHMARK(BM_LRNGrad##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH);          \

#define BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, NTH)     \
  BM_LRNGrad_Base(Default, DR, in_shape, name, T, DEVICE, NTH); \
  BM_LRNGrad_Base(Mkl, DR, in_shape, name, T, DEVICE, NTH);     \

#define BM_LRNGrad_NTH(DR, in_shape, name, T, DEVICE) \
  BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, 1);  \
  BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, 4);  \
  BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, 8);  \

#define BM_LRNGrad_DT(DR, in_shape, name)         \
  BM_LRNGrad_NTH(DR, in_shape, name, float, cpu); \

#define BM_LRNGrad(name, DR, ...)                      \
  BM_LRNGrad_DT(DR, TensorShape({__VA_ARGS__}), name); \

BM_LRNGrad(_128x12x12x64, 4, 128, 12, 12, 64);
BM_LRNGrad(_128x56x56x64, 2, 128, 56, 56, 64);
BM_LRNGrad(_128x27x27x192, 2, 128, 27, 27, 192);

}  // end namespace tensorflow

#endif  // INTEL_MKL
