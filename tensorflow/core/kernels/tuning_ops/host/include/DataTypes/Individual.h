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

#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "VectorFloat.h"
#include "MapStringToPtr.h"
#include "OptimizerIF.h"

typedef struct _Individual {
  Map_StringToPtr m_mapPopParam;
  Vector_Float m_fitness;
} Individual;

Individual *Individual_Ctor(void);
Individual *Individual_Copy_Ctor(Individual *param);
void Individual_Copy(Individual *dst, Individual *src);
void Individual_Dtor(Individual *self);
void Individual_Assign(Individual *self, Individual *src);
bool_t Individual_IsSame(Individual *self, Map_StringToString p);
void Individual_Print(Individual *self);

#endif
