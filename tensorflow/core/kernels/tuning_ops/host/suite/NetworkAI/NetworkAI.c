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

#include "NetworkAI.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void NetworkAI_get_var(Suite *self) {
  Vector_String_PushBack(self->var, "nstreams");
  Vector_Float_PushBack(self->var_min, 1);
  Vector_Float_PushBack(self->var_max, 112);
  Vector_String_PushBack(self->var, "nireq");
  Vector_Float_PushBack(self->var_min, 1);
  Vector_Float_PushBack(self->var_max, 224);
  Vector_String_PushBack(self->var, "nthreads");
  Vector_Float_PushBack(self->var_min, 1);
  Vector_Float_PushBack(self->var_max, 112);
}

Vector_Float NetworkAI_evaluate(Suite *self, float *x) {
  NetworkAI *derived = (NetworkAI *) self;
  char cmd[1024] = "";
  strcat(cmd, derived->exe_path);
  strcat(cmd, " -m ");
  strcat(cmd, derived->xml_path);
  strcat(cmd, " -d CPU -nstreams ");
  static char X_nstreams[5];
  sprintf(X_nstreams, "%d", (int) x[0]);
  strcat(cmd, X_nstreams);
  strcat(cmd, " -nireq ");
  static char X_nireq[5];
  sprintf(X_nireq, "%d", (int) x[1]);
  strcat(cmd, X_nireq);
  strcat(cmd, " -nthreads ");
  static char X_nthreads[5];
  sprintf(X_nthreads, "%d", (int) x[2]);
  strcat(cmd, X_nthreads);
  //strcat(cmd, " -niter 10000 -api async | tee temp.txt");
  strcat(cmd, " -niter 10000 -api async");
  PRINTF("%s\n", cmd);
  strcat(cmd, " > temp.txt");
  if (system(cmd) != -1) {
    float throughput = 0;
    FILE *fp = fopen("temp.txt", "r");
    char line[512];
    while (!feof(fp)) {
      if (fgets(line, 512, fp)) {
        char *flag = strstr(line, "Throughput: ");
        if (flag) {
          char *begin = flag + 12;
          char *end = strstr(line, " FPS");
          char str[64] = "";
          strncpy(str, begin, end - begin);
          str[end - begin] = '\0';
          throughput = atof(str);
        }
      }
    }
    PRINTF("throughput = %f\n\n", throughput);
    fclose(fp);
    if (system("rm temp.txt") == -1) {
      PRINTF("cmd goes wrong!\n");
    }
    if (Vector_Float_Size(self->fitness) > 0) {
      *Vector_Float_Visit(self->fitness, 0)->m_val = throughput;
    } else {
      Vector_Float_PushBack(self->fitness, throughput);
    }
    ++(derived->iter);
  } else {
    PRINTF("cmd goes wrong!\n");
  }
  return self->fitness;
}

int NetworkAI_parse(struct Suite *self, int argc, char *argv[]) {
  NetworkAI *derived = (NetworkAI *) self;
  bool_t exeSet = false_t;
  bool_t xmlSet = false_t;
  int offset = 0;
  int i = 0;
  // Parse and set -exe, -xml, skip others
  for (i = 1; i < argc; i += offset) {
    offset = 1; // default offset for an option
    if (strcmp(argv[i], "-exe") == 0) {
      if (i == argc - 1) {
        PRINTF("%s:%d Path of NetworkAI executable not specified after -exe.\n", __func__, __LINE__);
        return -1;
      }
      derived->exe_path = argv[i + 1];
      exeSet = true_t;
      offset = 2;
    } else if (strcmp(argv[i], "-xml") == 0) {
      if (i == argc - 1) {
        PRINTF("%s:%d Path of NetworkAI xml not specified after -xml.\n", __func__, __LINE__);
        return -1;
      }
      derived->xml_path = argv[i + 1];
      xmlSet = true_t;
      offset = 2;
    }
    if (exeSet && xmlSet) {
      break;
    }
  }
  if (!exeSet || !xmlSet) {
    PRINTF("%s:%d Not specify exe and/or xml for NetworkAI.\n", __func__, __LINE__);
    return -1;
  }
  return 0;
}

NetworkAI *NetworkAI_Ctor(void) {
  NetworkAI *self = (NetworkAI *) malloc(sizeof(NetworkAI));
  if (self) {
    Suite_Ctor(&(self->base));
    self->base.name = NETWORK_AI;
    self->base.evaluate = NetworkAI_evaluate;
    self->base.get_var = NetworkAI_get_var;
    self->base.parse = NetworkAI_parse;
    self->iter = 0;
    self->exe_path = nullptr;
    self->xml_path = nullptr;
  } else {
    PRINTF("%s faild!\n", __func__);
  }
  return self;
}

void NetworkAI_Dtor(NetworkAI *self) {
  if (!self) {
    return;
  }
  Suite_Dtor(&(self->base));
  FREE(self);
  self = nullptr;
}
