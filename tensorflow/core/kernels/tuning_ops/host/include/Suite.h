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

#ifndef SUITE_H
#define SUITE_H
#include "VectorFloat.h"
#include "VectorString.h"

typedef enum SuiteCaseName {
  SUITE_NOT_SPECIFIED = 0, // default
  ACKLEY = 1,
  NETWORK_AI = 2,
  EXP_SIN_FUNC = 3,
} SuiteCaseName;

typedef struct Suite {
  SuiteCaseName name;
  Vector_String var;
  Vector_Float var_min;
  Vector_Float var_max;
  Vector_Float fitness;

  Vector_Float (*target_func)(struct Suite *self);
  Vector_Float (*evaluate)(struct Suite *self, float *x);
  void (*get_var)(struct Suite *self);
  bool_t (*repair)(struct Suite *self, float *x);
  int (*parse)(struct Suite *self, int argc, char *argv[]);
} Suite;
void Suite_Ctor(Suite *self);
void Suite_Dtor(Suite *self);
int getSuite(int argc, char *argv[], Suite **pp_Suite);
void freeSuite(Suite *self);

#endif
