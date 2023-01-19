#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status SetOutputShapeFor_MklFusedMatMulWithReshape(InferenceContext* c) {
  ShapeHandle a;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

  ShapeHandle b;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));
  DimensionHandle output_rows = transpose_a ? c->Dim(a, 1) : c->Dim(a, 0);
  DimensionHandle output_cols = transpose_b ? c->Dim(b, 0) : c->Dim(b, 1);

  // Validate that the inner shapes are compatible.
  DimensionHandle inner_a = transpose_a ? c->Dim(a, 0) : c->Dim(a, 1);
  DimensionHandle inner_b = transpose_b ? c->Dim(b, 1) : c->Dim(b, 0);
  DimensionHandle merged;
  TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));

//   c->set_output(0, c->Matrix(output_rows, output_cols));

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &out));

  if (!c->RankKnown(out)) {
    // We have no information about the shape of the output.
    c->set_output(0, out);
    return Status::OK();
  }

  if (c->RankKnown(out)) {
    // We don't know the number of output elements, but we can try to infer
    // the missing dimension.
    bool too_many_unknown = false;
    int32 out_unknown_idx = -1;

    DimensionHandle known_out_elems = c->NumElements(out);
    if (!c->ValueKnown(known_out_elems)) {
      known_out_elems = c->MakeDim(1);
      for (int32 i = 0; i < c->Rank(out); ++i) {
        DimensionHandle dim = c->Dim(out, i);
        if (!c->ValueKnown(dim)) {
          if (out_unknown_idx >= 0) {
            too_many_unknown = true;
            break;
          }
          out_unknown_idx = i;
        } else {
          TF_RETURN_IF_ERROR(
              c->Multiply(known_out_elems, dim, &known_out_elems));
        }
      }
    }
    DimensionHandle known_in_elems;
    TF_RETURN_IF_ERROR(
              c->Multiply(output_rows, output_cols, &known_in_elems));

    if (!too_many_unknown) {
      if (out_unknown_idx < 0) {
        // Just check that the dimensions match.
        if (c->Value(known_in_elems) != c->Value(known_out_elems)) {
          return errors::InvalidArgument(
              "Cannot reshape a tensor with ", c->DebugString(known_in_elems),
              " elements to shape ", c->DebugString(out), " (",
              c->DebugString(known_out_elems), " elements)");
        }
      } else if (out_unknown_idx >= 0 &&
                 c->Value(known_out_elems) > 0) {
        // Input fully known, infer the one missing output dim
        DimensionHandle inferred_dim;
        TF_RETURN_IF_ERROR(c->Divide(known_in_elems, c->Value(known_out_elems),
                                     true /* evenly_divisible */,
                                     &inferred_dim));
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(out, out_unknown_idx, inferred_dim, &out));

      }
    }
  }
  c->set_output(0, out);
  return Status::OK();
}

REGISTER_OP("_MklFusedMatMulWithReshape")
    .Input("a: T")
    .Input("b: T")
    .Input("shape: Tshape")
    .Input("args: num_args * T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, float}")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .Attr("num_args: int >= 0")
    .Attr("fused_ops: list(string) = []")
    // --------------------------------------------- //
    .SetShapeFn([](InferenceContext* c) { return SetOutputShapeFor_MklFusedMatMulWithReshape(c); })
    .Doc(R"doc(
MKL version of FusedMatMulWithReshape operator. Uses MKL-DNN APIs to implement MatMul
operator.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

}  // namespace tensorflow
