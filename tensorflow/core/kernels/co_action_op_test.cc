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

namespace tensorflow {

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//
template <typename T>
static Graph* Coaction(const string& kind, int m, int k, int n, int bs, int pl) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "CoAction" : "_OptCoAction";

  Tensor in0(type, TensorShape({bs, pl, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, TensorShape({bs, pl, k, n}));
  in1.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);
  Node* input_in1 = test::graph::Constant(g, in1);

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_in0)
                    .Input(input_in1)
                    .Attr("pow_num", 2);

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_Coaction_Base(kind, M, K, N, BS, PL, T, DEVICE, NTH)                              \
  static void BM_Coaction##_##kind##_##M##_##K##_##N##_##BS##_##PL##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                           \
    testing::UseRealTime();                                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters) * BS * PL * M * K * N * 2 * 2);        \
    SessionOptions opts;                                                                     \
    opts.config.set_intra_op_parallelism_threads(NTH);                                       \
    test::Benchmark(#DEVICE, Coaction<T>(#kind, M, K, N, BS, PL), &opts).Run(iters);         \
  }                                                                                          \
  BENCHMARK(BM_Coaction##_##kind##_##M##_##K##_##N##_##BS##_##PL##_##T##_##DEVICE##_##NTH);  \

#define BM_Coaction_kind(M, K, N, BS, PL, T, DEVICE, NTH)     \
  BM_Coaction_Base(Default, M, K, N, BS, PL, T, DEVICE, NTH); \
  // BM_Coaction_Base(Opt, M, K, N, BS, PL, T, DEVICE, NTH);  \

#define BM_Coaction_NTH(M, K, N, BS, PL, T, DEVICE) \
  BM_Coaction_kind(M, K, N, BS, PL, T, DEVICE, 1);  \
  BM_Coaction_kind(M, K, N, BS, PL, T, DEVICE, 4);  \
  BM_Coaction_kind(M, K, N, BS, PL, T, DEVICE, 8);  \

#define BM_Coaction(M, K, N, BS, PL)               \
  BM_Coaction_NTH(M, K, N, BS, PL, float, cpu);    \
  // BM_Coaction_NTH(M, K, N, BS, PL, bfloat16, cpu); \

// Vector * Vector
BM_Coaction(50, 5, 4, 1, 1);
BM_Coaction(50, 5, 4, 166, 1);
BM_Coaction(150, 5, 4, 1, 1);
BM_Coaction(150, 5, 4, 166, 1);

}  // end namespace tensorflow
