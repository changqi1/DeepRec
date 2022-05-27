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

class FusedL2NormalizeOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype, int axis, float epsilon) {
    TF_EXPECT_OK(NodeDefBuilder("fused_l2_normalize", "FusedL2Normalize")
                     .Attr("T", dtype)
                     .Attr("axis", axis)
                     .Attr("epsilon", epsilon)
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedL2NormalizeOpTest, 2Dims_Float) {
  const int rows = 4;
  const int cols = 252; //128+64+32+16+8+4=252 1008

  MakeOpAndSetDevice(Device::CPU, DT_FLOAT, 0, 1e-12);

  // x
    float input_array[1008];
    for (int i = 0; i < sizeof(input_array) / sizeof(float); i++) {
      input_array[i] = 1.0;
    }
  AddInputFromArray<float>(TensorShape({rows, cols}), input_array);

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
}
}
