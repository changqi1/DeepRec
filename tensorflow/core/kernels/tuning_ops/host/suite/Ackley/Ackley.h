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

#ifndef ACKLEY_H
#define ACKLEY_H
#include "OptimizerIF.h"
#include "VectorFloat.h"
#include "VectorString.h"
#include "Suite.h"

typedef struct Ackley {
  Suite base;
  int n; // x's dimensions
  float a, b, c;
} Ackley;

Ackley *Ackley_Ctor(int n);
void Ackley_Dtor(Ackley *self);

#endif
