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

#include "Suite.h"
#include "Ackley.h"
#include "NetworkAI.h"
#include "Latency.h"
#include "ExpSinFunc.h"

int getSuite(int argc, char *argv[], Suite **pp_Suite) {
  bool_t pp_SuiteSet = false_t;
  int i = 0;

  if (nullptr == argv || nullptr == pp_Suite) {
    PRINTF("Invalid parameter for getSuite()!");
    return -1;
  }

  // Parse -suite, skip others
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-suite") == 0) {
      if (i == argc - 1) {
        PRINTF("Suite not specified after -suite.\n");
        return -1;
      }
      if (strcmp(argv[i + 1], "Ackley") == 0) {
        *pp_Suite = (Suite *) Ackley_Ctor(2);
      } else if (strcmp(argv[i + 1], "NetworkAI") == 0) {
        *pp_Suite = (Suite *) NetworkAI_Ctor();
      } else if (strcmp(argv[i + 1], "ExpSinFunc") == 0) {
        *pp_Suite = (Suite *) ExpSinFunc_Ctor();
      } else if (strcmp(argv[i + 1], "Latency") == 0) {
        *pp_Suite = (Suite *) Latency_Ctor();
      } else {
        PRINTF("Unsupported suite %s.\n", argv[i + 1]);
        PRINTF("Currently support Ackley, NetworkAI, Latency and ExpSinFunc.\n");
        return -1;
      }
      pp_SuiteSet = true_t;
      break;
    }
  }

  if (!pp_SuiteSet) {
    // set Ackley as default test suite
    *pp_Suite = (Suite *) Ackley_Ctor(2);;
  }
  if (nullptr == *pp_Suite) {
    PRINTF("Fail to construct the suite.\n");
    return -1;
  }
  if (nullptr != (*pp_Suite)->parse) {
    (*pp_Suite)->parse((*pp_Suite), argc, argv);
  }
  return 0;
}

void freeSuite(Suite *self) {
  if (!self) {
    return;
  }
  switch (self->name) {
    case SUITE_NOT_SPECIFIED:Suite_Dtor(self);
      break;
    case ACKLEY:Ackley_Dtor((Ackley *) self);
      break;
    case NETWORK_AI:NetworkAI_Dtor((NetworkAI *) self);
      break;
    case EXP_SIN_FUNC:ExpSinFunc_Dtor((ExpSinFunc *) self);
      break;
    default:PRINTF("No support for suite: %d\n", self->name);
      break;
  }
}