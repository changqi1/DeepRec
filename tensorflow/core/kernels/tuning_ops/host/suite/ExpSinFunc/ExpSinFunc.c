//
// Copyright 2020-2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version Septmeber 2018)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "ExpSinFunc.h"

#define PI 3.14159265358979323846

void ExpSinFunc_get_var(Suite *self) {
  char key[] = "param1";
  Vector_String_PushBack(self->var, key);
  Vector_Float_PushBack(self->var_min, 0.0);
  Vector_Float_PushBack(self->var_max, 1.0);
}

Vector_Float ExpSinFunc_evaluate(Suite *self, float *x) {
  float f_x = pow(x[0], 2.0) * pow(sin(5.0 * PI * x[0]), 6.0);
  if (Vector_Float_Size(self->fitness) > 0) {
    *Vector_Float_Visit(self->fitness, 0)->m_val = f_x;
  } else {
    Vector_Float_PushBack(self->fitness, f_x);
  }

  return self->fitness;
}

ExpSinFunc *ExpSinFunc_Ctor() {
  ExpSinFunc *self = (ExpSinFunc *) MALLOC(sizeof(ExpSinFunc));
  if (self) {
    Suite_Ctor(&(self->base));
    self->base.name = EXP_SIN_FUNC;
    self->base.get_var = ExpSinFunc_get_var;
    self->base.evaluate = ExpSinFunc_evaluate;
  }
  return self;
}

void ExpSinFunc_Dtor(ExpSinFunc *self) {
  if (nullptr == self) {
    return;
  }
  Suite_Dtor(&(self->base));
  FREE(self);
  self = nullptr;
}
