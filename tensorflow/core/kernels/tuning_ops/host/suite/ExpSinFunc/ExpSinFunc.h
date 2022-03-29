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

#ifndef BOTESTCASE_H
#define BOTESTCASE_H
#include "Suite.h"

typedef struct ExpSinFunc {
  Suite base;
} ExpSinFunc;
ExpSinFunc *ExpSinFunc_Ctor(void);
void ExpSinFunc_Dtor(ExpSinFunc *self);

#endif
