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

#ifndef SIMPLEMAP_H
#define SIMPLEMAP_H

#include "VectorString.h"
#include "Common.h"

typedef struct Pair_StringToString {
  char *m_key;
  char *m_value;
  struct Pair_StringToString *m_next;
} Pair_StringToString, *Map_StringToString;

Map_StringToString Map_StringToString_Ctor(void);
void Map_StringToString_Dtor(Map_StringToString mss);
int Map_StringToString_Size(Map_StringToString mss);
void Map_StringToString_Print(Map_StringToString mss);
Pair_StringToString *Map_StringToString_Find(Map_StringToString mss, char *key);
Pair_StringToString *Map_StringToString_Visit(Map_StringToString mss, char *key);
void Map_StringToString_PushBack(Map_StringToString mss, char *key, char *value);

#endif
