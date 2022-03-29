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
#include "OptimizerIF.h"

int main(int argc, char *argv[]) {
  int res = 0;
  Suite *p_suite = nullptr;

  do {
    res = checkHelp(argc, argv);
    if (res != 0) {
      break;
    }

    res = getSuite(argc, argv, &p_suite);
    if (res < 0) {
      PRINTF("Fail to get suite.\n");
      break;
    }

    res = tune(argc, argv, p_suite);
    if (res < 0) {
      PRINTF("Fail to tune!\n");
      break;
    }
  } while (false_t);

  freeSuite(p_suite);
  return 0;
}
