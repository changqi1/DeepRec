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

#include "Ackley.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Vector_Float Ackley_evaluate(Suite *self, float *x) {
  Ackley *derived = (Ackley *) self;
  float sigmaSquare = 0.0;
  float sigmaCos = 0.0;
  float part1, part2, sum;
  int i;
  for (i = 0; i < derived->n; ++i) {
    sigmaSquare += (x[i] * x[i]);
    sigmaCos += cos(derived->c * x[i]);
  }
  part1 = -derived->a * exp(-derived->b * sqrt(1.0 / derived->n * sigmaSquare));
  part2 = -exp(1.0 / derived->n * sigmaCos);
  sum = -1.0 * (part1 + part2 + derived->a + exp(1));
  if (Vector_Float_Size(self->fitness) > 0) {
    *Vector_Float_Visit(self->fitness, 0)->m_val = sum;
  } else {
    Vector_Float_PushBack(self->fitness, sum);
  }

  return self->fitness;
}

void Ackley_get_var(Suite *self) {
  Ackley *derived = (Ackley *) self;
  int i;
  for (i = 0; i < derived->n; ++i) {
    char str1[100] = "key";
    char str2[32];
    sprintf(str2, "%d", i);
    strcat(str1, str2);
    Vector_String_PushBack(self->var, str1);
    Vector_Float_PushBack(self->var_min, -32.768);
    Vector_Float_PushBack(self->var_max, 32.768);
  }
  return;
}

Ackley *Ackley_Ctor(int n) {
  Ackley *self = (Ackley *) malloc(sizeof(Ackley));
  if (self) {
    Suite_Ctor(&(self->base));
    self->a = 20;
    self->b = 0.2;
    self->c = 2 * 3.14159265;
    self->n = n;
    self->base.name = ACKLEY;
    self->base.evaluate = Ackley_evaluate;
    self->base.get_var = Ackley_get_var;
  } else {
    PRINTF("%s faild!\n", __func__);
  }
  return self;
}

void Ackley_Dtor(Ackley *self) {
  if (self == nullptr) {
    return;
  }
  Suite_Dtor(&(self->base));
  FREE(self);
  self = nullptr;
}
