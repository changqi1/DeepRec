#ifdef INTEL_MKL

#include "absl/algorithm/container.h"
#include "dnnl.hpp"
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

namespace tensorflow {

// static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
// static const TensorShape dummy_shape({8});

class FusedMatMulWithReshapeOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(bool transpose_a, bool transpose_b,
                          const string& activation = "") {
    std::vector<string> fused_ops{"Reshape", "BiasAdd"};

    if (activation != "" && activation != "null") {
      fused_ops.push_back(activation);
    }
    TF_EXPECT_OK(NodeDefBuilder("fused_matmul", "_MklFusedMatMulWithReshape")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(1, DT_FLOAT))
                     .Attr("T", DT_FLOAT)
                     .Attr("Tshape", DT_INT32)
                     .Attr("num_args", 1)
                     .Attr("fused_ops", fused_ops)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Attr("_kernel", "MklNameChangeOp")
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());
  }
};


                    //  .Input(FakeInput(DT_UINT8))
                    //  .Input(FakeInput(DT_UINT8))
                    //  .Input(FakeInput(DT_UINT8))
                    //  .Input(FakeInput(1, DT_UINT8))

TEST_F(FusedMatMulWithReshapeOpTest, Relu) {
  MakeOpAndSetDevice(false, false, "Relu");

  AddInput<float>(TensorShape({50, 100}), [](int i) -> float { return 2.0; });
  AddInput<float>(TensorShape({100, 50}), [](int i) -> float { return 2.0; });
  AddInputFromArray<int>(TensorShape({3}), {25, 2, 50});
  AddInput<float>(TensorShape({50}), [](int i) -> float { return 2.0; });
  // AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  // AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  // AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  // AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({25,2,50}));
  std::vector<float> expected_val(2500, 402);
  test::FillValues<float>(&expected, expected_val);
  // test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));

}
//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* FusedMatMulWithReshape(int m, int k, int n, int a, int b,
                                     bool transpose_a, bool transpose_b,
                                     const string& activation = "") {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  std::vector<string> fused_ops{"Reshape", "BiasAdd"};

  if (activation != "" && activation != "null") {
    fused_ops.push_back(activation);
  }

  int num_args = 1;
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  Tensor bias(type, TensorShape({transpose_b ? k : n}));
  bias.flat<T>().setRandom();
  Tensor reshape_to(DT_INT32, TensorShape({3}));
  reshape_to.flat<int32>().setValues({a, b, n});

  Node* input_in0        = test::graph::Constant(g, in0);
  Node* input_in1        = test::graph::Constant(g, in1);
  Node* input_bias       = test::graph::Constant(g, bias, absl::StrCat("arg", 1));
  Node* input_reshape_to = test::graph::Constant(g, reshape_to);

  Node* not_mkl_shape = test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  std::vector<NodeBuilder::NodeOut> args;
  std::vector<NodeBuilder::NodeOut> args_not_mkl;
  args.push_back(input_bias);
  args_not_mkl.push_back(not_mkl_shape);

  auto nodeBuilder =
      NodeBuilder(g->NewName("fused_matmul"), "_MklFusedMatMulWithReshape")
          .Input(input_in0)
          .Input(input_in1)
          .Input(input_reshape_to)
          .Input(args)
          .Attr("T", type)
          .Attr("Tshape", DT_INT32)
          .Attr("num_args", num_args)
          .Attr("fused_ops", fused_ops)
          .Attr("transpose_a", transpose_a)
          .Attr("transpose_b", transpose_b);

  nodeBuilder.Attr("_kernel", "MklNameChangeOp");
  // nodeBuilder.Attr("_kernel", "MklLayoutDependentOp")
  //     .Input(not_mkl_shape)
  //     .Input(not_mkl_shape)
  //     .Input(not_mkl_shape)
  //     .Input(args_not_mkl);

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_FusedMatMulWithReshape_Base(ACT, M, K, N, A, B, TA, TB, T, DEVICE,                                 \
                                       NTH)                                                                   \
  static void                                                                                                 \
      BM_FusedMatMulWithReshape##_##ACT##_##M##_##K##_##N##_##A##_##B##_##TA##_##TB##_##T##_##DEVICE##_##NTH( \
          int iters) {                                                                                        \
    testing::UseRealTime();                                                                                   \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);                                       \
    SessionOptions opts;                                                                                      \
    opts.config.set_intra_op_parallelism_threads(NTH);                                                        \
    test::Benchmark(#DEVICE,                                                                                  \
                    FusedMatMulWithReshape<T>(M, K, N, A, B, TA, TB, #ACT),                                   \
                    &opts)                                                                                    \
        .Run(iters);                                                                                          \
  }                                                                                                           \
  BENCHMARK(                                                                                                  \
      BM_FusedMatMulWithReshape##_##ACT##_##M##_##K##_##N##_##A##_##B##_##TA##_##TB##_##T##_##DEVICE##_##NTH);


#define BM_FusedMatMulWithReshape_NTH(ACT, M, K, N, A, B, TA, TB, T, DEVICE) \
  BM_FusedMatMulWithReshape_Base(ACT, M, K, N, A, B, TA, TB, T, DEVICE, 1);  \
  BM_FusedMatMulWithReshape_Base(ACT, M, K, N, A, B, TA, TB, T, DEVICE, 4);  \
  BM_FusedMatMulWithReshape_Base(ACT, M, K, N, A, B, TA, TB, T, DEVICE, 8);

#define BM_FusedMatMulWithReshape_ACT(M, K, N, A, B, TA, TB, T, DEVICE)  \
  BM_FusedMatMulWithReshape_NTH(null, M, K, N, A, B, TA, TB, T, DEVICE); \
  BM_FusedMatMulWithReshape_NTH(Relu, M, K, N, A, B, TA, TB, T, DEVICE);

#define BM_FusedMatMulWithReshape(M, K, N, A, B, TA, TB)            \
  BM_FusedMatMulWithReshape_ACT(M, K, N, A, B, TA, TB, float, cpu); \
  BM_FusedMatMulWithReshape_ACT(M, K, N, A, B, TA, TB, bfloat16, cpu);


BM_FusedMatMulWithReshape_NTH(Relu, 50, 50, 50, 25, 2, false, false, float, cpu);

// // Vector * Vector
// BM_FusedMatMulWithReshape(1, 50, 1, 1, 1, false, false);
// BM_FusedMatMulWithReshape(1, 2000, 1, 1, 1, false, false);

// BM_FusedMatMulWithReshape(50, 1, 50, 25, 2, false, false);
// BM_FusedMatMulWithReshape(2000, 1, 2000, 1000, 2, false, false);

// // Vector * Matrix
// BM_FusedMatMulWithReshape(1, 50, 50, 1, 1, false, false);
// BM_FusedMatMulWithReshape(1, 2000, 2000, 1, 1, false, false);

// BM_FusedMatMulWithReshape(50, 50, 1, 25, 2, false, false);
// BM_FusedMatMulWithReshape(2000, 2000, 1, 1000, 2, false, false);

// // Matrix * Matrix
// BM_FusedMatMulWithReshape(32, 32, 32, 16, 2, false, false);
// BM_FusedMatMulWithReshape(51200, 64, 64, 25600, 2, false, false);
// BM_FusedMatMulWithReshape(8, 512, 512, 4, 2, false, false);
// BM_FusedMatMulWithReshape(128, 512, 512, 64, 2, false, false);
// BM_FusedMatMulWithReshape(16, 1024, 1024, 8, 2, false, false);
// BM_FusedMatMulWithReshape(256, 1024, 1024, 128, 2, false, false);
// BM_FusedMatMulWithReshape(4096, 4096, 4096, 2048, 2, false, false);

// BM_FusedMatMulWithReshape(2560, 64, 1, 1280, 2, false, false);
// BM_FusedMatMulWithReshape(2560, 448, 1, 1280, 2, false, false);
// BM_FusedMatMulWithReshape(2560, 2304, 64, 1280, 2, false, false);
// BM_FusedMatMulWithReshape(2560, 1040, 1536, 1280, 2, false, false);
// BM_FusedMatMulWithReshape(2560, 14435, 2304, 1280, 2, false, false);

}  // end namespace tensorflow

#endif  // INTEL_MKL
