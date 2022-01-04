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

#include "mkldnn.hpp"
#include "absl/algorithm/container.h"
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

namespace tensorflow {

  //----------------------------------------------------------------------------//
  // Performance benchmarks are below.                                          //
  //----------------------------------------------------------------------------//

  template <typename T>
  static Graph* InputConversion(const string& kind, int m, int n, int j, int k) {
    auto* g = new Graph(OpRegistry::Global());
    DataType type = DataTypeToEnum<T>::v();

    string op_name = "_MklInputConversion";

    Tensor tensor_0(type, TensorShape({ m, n }));
    tensor_0.flat<T>().setRandom();
    Node* input_0 = test::graph::Constant(g, tensor_0, "input_0");
    Node* not_mkl_shape_0 =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

    Tensor tensor_1(type, TensorShape({ j, k }));
    tensor_1.flat<T>().setRandom();
    Node* input_1 = test::graph::Constant(g, tensor_1, "input_1");
    Node* not_mkl_shape_1 =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");
      
    auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
      .Input(input_0)
      .Input(input_1)
      .Input(not_mkl_shape_0)
      .Input(not_mkl_shape_1)
      .Attr("T", type);

    nodeBuilder.Attr("_kernel", "MklLayoutDependentOp");

    TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

    return g;
  }

#define BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, NTH)                           \
  static void BM_Input_Conversion##_##kind##_##M##_##N##_##J##_##K##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                           \
    testing::UseRealTime();                                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters));                                      \
    SessionOptions opts;                                                                     \
    opts.config.set_intra_op_parallelism_threads(NTH);                                       \
    test::Benchmark(#DEVICE, InputConversion<T>(#kind, M, N, J, K), &opts).Run(iters);       \
  }                                                                                          \
  BENCHMARK(BM_Input_Conversion##_##kind##_##M##_##N##_##J##_##K##_##T##_##DEVICE##_##NTH);  \

#define BM_Input_Conversion_NTH(kind, M, N, J, K, T, DEVICE) \
  BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, 1);  \
  BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, 4);  \
  BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, 8);  \

#define BM_Input_Conversion_kind(M, N, J, K, T, DEVICE) \
  BM_Input_Conversion_NTH(Mkl, M, N, J, K, T, DEVICE);  \

#define BM_Input_Conversion_DT(M, N, J, K)             \
  BM_Input_Conversion_kind(M, N, J, K, float, cpu);    \
  BM_Input_Conversion_kind(M, N, J, K, bfloat16, cpu); \

  BM_Input_Conversion_DT(128, 128, 128, 128);
  BM_Input_Conversion_DT(128, 128, 256, 128);
  BM_Input_Conversion_DT(256, 128, 128, 128);


}  // end namespace tensorflow

#endif  // INTEL_MKL
