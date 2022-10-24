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

#ifndef MAPVECTORTOMAP_H
#define MAPVECTORTOMAP_H

#include "MapStringToString.h"
#include "VectorString.h"
#include "Common.h"

typedef struct Pair_VectorToMap {
  Node_String *m_vs;
  Pair_StringToString *m_mss;
  struct Pair_VectorToMap *m_next;
} Pair_VectorToMap, *Map_VectorToMap;

Map_VectorToMap Map_VectorToMap_Ctor(void);
void Map_VectorToMap_Dtor(Map_VectorToMap mvm);
int Map_VectorToMap_Size(Map_VectorToMap mvm);
void Map_VectorToMap_Print(Map_VectorToMap mvm);
Pair_VectorToMap *Map_VectorToMap_Find(Map_VectorToMap mvm, Vector_String vs);
void Map_VectorToMap_PushBack(Map_VectorToMap mvm, Vector_String vs, Map_StringToString mss);

#endif
