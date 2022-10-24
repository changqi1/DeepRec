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

#ifndef MAPMAPTOFLOAT_H
#define MAPMAPTOFLOAT_H

#include "MapStringToFloat.h"
#include "Common.h"

typedef struct Pair_MapToFloat {
  Map_StringToFloat m_msf;
  float *m_val;
  struct Pair_MapToFloat *m_next;
} Pair_MapToFloat, *Map_MapToFloat;

Map_MapToFloat Map_MapToFloat_Ctor(void);
void Map_MapToFloat_Dtor(Map_MapToFloat mmf);
void Map_MapToFloat_Print(Map_MapToFloat mmf);
Pair_MapToFloat *Map_MapToFloat_Find(Map_MapToFloat mmf, Map_StringToFloat msf);
void Map_MapToFloat_PushBack(Map_MapToFloat mmf, Map_StringToFloat msf, float val);

#endif
