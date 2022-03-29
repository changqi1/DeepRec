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

#ifndef MAPSTRINGTOPOINTER_H
#define MAPSTRINGTOPOINTER_H

#include "Common.h"

struct OptimizedParamIF;

typedef struct Pair_StringToPtr {
  char *m_string;
  struct OptimizedParamIF *m_ptr;
  struct Pair_StringToPtr *m_next;
} Pair_StringToPtr, *Map_StringToPtr;

Map_StringToPtr Map_StringToPtr_Ctor(void);
void Map_StringToPtr_Dtor(Map_StringToPtr msp);
int Map_StringToPtr_Size(Map_StringToPtr msp);
void Map_StringToPtr_Print(Map_StringToPtr msp);
Pair_StringToPtr *Map_StringToPtr_Find(Map_StringToPtr msp, char *key);
void Map_StringToPtr_PushBack(Map_StringToPtr msp, char *str, struct OptimizedParamIF *ptr);
Map_StringToPtr Map_StringToPtr_Erase(Map_StringToPtr msp, char *key);
Pair_StringToPtr *Map_StringToPtr_Visit(Map_StringToPtr msp, char *key);
bool_t Map_StringToPtr_IsSame(Map_StringToPtr m1, Map_StringToPtr m2);
void Map_StringToPtr_Assign(Map_StringToPtr m1, Map_StringToPtr m2);

#endif
