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

#ifndef LATENCY_H
#define LATENCY_H
#include "VectorInt.h"
#include "VectorFloat.h"
#include "VectorString.h"
#include "MapStringToInt.h"
#include "OptimizerIF.h"
#include "Suite.h"

typedef struct Latency {
  Suite base;
  int iter;
  char *exe_path;
  char *xml_path;
} Latency;

Latency *Latency_Ctor(void);
void Latency_Dtor(Latency *self);

#endif