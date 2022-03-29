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

#ifndef MAPSTRINGTOFLOAT_H
#define MAPSTRINGTOFLOAT_H

#include "Common.h"

typedef struct Pair_StringToFloat {
  char *m_string;
  float *m_val;
  struct Pair_StringToFloat *m_next;
} Pair_StringToFloat, *Map_StringToFloat;

Map_StringToFloat Map_StringToFloat_Ctor(void);
void Map_StringToFloat_Dtor(Map_StringToFloat msf);
void Map_StringToFloat_Print(Map_StringToFloat msf);
Pair_StringToFloat *Map_StringToFloat_Find(Map_StringToFloat msf, char *key);
void Map_StringToFloat_PushBack(Map_StringToFloat msf, char *str, float val);
Map_StringToFloat Map_StringToFloat_Erase(Map_StringToFloat msf, char *key);
bool_t Map_StringToFloat_Cmp(Map_StringToFloat m1, Map_StringToFloat m2);
Pair_StringToFloat *Map_StringToFloat_Visit(Map_StringToFloat msf, char *str);

#endif
