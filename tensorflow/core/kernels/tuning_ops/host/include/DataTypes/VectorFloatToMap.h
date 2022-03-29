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

#ifndef VECTORFLOATTOMAP_H
#define VECTORFLOATTOMAP_H

#include "MapStringToFloat.h"
#include "Common.h"

typedef struct Pair_FloatToMap {
  float *m_val;
  Pair_StringToFloat *m_msf;
  struct Pair_FloatToMap *m_next;
} Pair_FloatToMap, *Vector_FloatToMap; // vector<pair<float, map<string, float>>>

Vector_FloatToMap Vector_FloatToMap_Ctor(void);
void Vector_FloatToMap_Dtor(Vector_FloatToMap vfm);
void Vector_FloatToMap_Print(Vector_FloatToMap vfm);
int Vector_FloatToMap_Size(Vector_FloatToMap vfm);
Pair_FloatToMap *Vector_FloatToMap_Visit(Vector_FloatToMap vfm, unsigned int i);
void Vector_FloatToMap_PushBack(Vector_FloatToMap vfm, float val, Map_StringToFloat msf);

#endif
