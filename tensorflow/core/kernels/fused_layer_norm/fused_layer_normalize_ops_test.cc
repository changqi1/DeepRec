#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace {

enum class Device { CPU, GPU };

class FusedLayerNormalizeOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype, int axis, float epsilon) {
    TF_EXPECT_OK(NodeDefBuilder("fused_l2_normalize", "FusedLayerNorm")
                     .Attr("T", dtype)
                     .Attr("epsilon", epsilon)
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedLayerNormalizeOpTest, 2Dims_Float) {
  const int rows = 4;
  const int cols = 252; //128+64+32+16+8+4=252 1008

  MakeOpAndSetDevice(Device::CPU, DT_FLOAT, 0, 1e-12);

  // x
    float input_array[1008];
    for (int i = 0; i < sizeof(input_array) / sizeof(float); i++) {
      input_array[i] = 1.0;
    }
  AddInputFromArray<float>(TensorShape({rows, cols}), input_array);
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 0.5; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 0.5; });

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_output(allocator(), DT_FLOAT,
                                TensorShape({rows, cols}));
    float output_array[1008];
    for (int i = 0; i < sizeof(output_array) / sizeof(float); i++) {
      output_array[i] = 0.062994122505188;
    }
    test::FillValues<float>(&expected_output, output_array);
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-6);
  }
}

//----------------------------------------------------------------------------//
// Performance benchmarks                                                     //
//----------------------------------------------------------------------------//
static Graph* FusedLayerNormalize(int rows, int cols) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType dtype = DT_FLOAT;

  Tensor in(dtype, TensorShape({rows, cols}));
  in.flat<float>().setRandom();
  Tensor gamma(dtype, TensorShape({cols}));
  in.flat<float>().setRandom();
  Tensor beta(dtype, TensorShape({cols}));
  in.flat<float>().setRandom();

  Node* input_in = test::graph::Constant(g, in);
  Node* input_gamma = test::graph::Constant(g, gamma);
  Node* input_beta = test::graph::Constant(g, beta);
  auto nodeBuilder = NodeBuilder(g->NewName("n"), "FusedLayerNorm")
                    .Input(input_in)
                    .Input(input_gamma)
                    .Input(input_beta)
                    .Attr("T", dtype)
                    .Attr("epsilon", 1e-12);
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_FusedLayerNorm(ROWS, COLS, NTH)                                   \
  static void BM_FusedLayerNorm##_##ROWS##_##COLS##_##NTH##_CPU(             \
  int iters) {                                                               \
  testing::UseRealTime();                                                    \
  testing::ItemsProcessed(static_cast<int64>(iters) * ROWS * COLS * 3);      \
  SessionOptions opts;                                                       \
  opts.config.set_intra_op_parallelism_threads(NTH);                         \
  test::Benchmark("cpu", FusedLayerNormalize(ROWS, COLS), &opts).Run(iters); \
  }                                                                          \
  BENCHMARK(BM_FusedLayerNorm##_##ROWS##_##COLS##_##NTH##_CPU);              \

#define BM_FusedLayerNorm_NTH(ROWS, COLS) \
  BM_FusedLayerNorm(ROWS, COLS, 1);       \
  BM_FusedLayerNorm(ROWS, COLS, 4);       \
  BM_FusedLayerNorm(ROWS, COLS, 8);       \

BM_FusedLayerNorm_NTH(1024, 63);
BM_FusedLayerNorm_NTH(1024, 255);
BM_FusedLayerNorm_NTH(1024, 511);
BM_FusedLayerNorm_NTH(1024, 1023);

}
}
