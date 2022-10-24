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

#ifndef MAPSTRINGTOINT_H
#define MAPSTRINGTOINT_H

#include "Common.h"

typedef struct Pair_StringToInt {
  char *m_string;
  int *m_val;
  struct Pair_StringToInt *m_next;
} Pair_StringToInt, *Map_StringToInt;

Map_StringToInt Map_StringToInt_Ctor(void);
void Map_StringToInt_Dtor(Map_StringToInt msi);
void Map_StringToInt_Print(Map_StringToInt msi);
Pair_StringToInt *Map_StringToInt_Find(Map_StringToInt msi, char *key);
void Map_StringToInt_PushBack(Map_StringToInt msi, char *str, int val);
Map_StringToInt Map_StringToInt_Erase(Map_StringToInt msi, char *key);
bool_t Map_StringToInt_Cmp(Map_StringToInt m1, Map_StringToInt m2);
Pair_StringToInt *Map_StringToInt_Visit(Map_StringToInt msi, char *str);

#endif
