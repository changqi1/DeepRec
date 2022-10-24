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

#ifndef VECTORFLOAT_H
#define VECTORFLOAT_H

#include "Common.h"

typedef struct Node_Float {
  float *m_val;
  struct Node_Float *m_next;
} Node_Float, *Vector_Float;

Vector_Float Vector_Float_Ctor(void);
void Vector_Float_Dtor(Vector_Float vf);
int Vector_Float_Size(Vector_Float vf);
void Vector_Float_Print(Vector_Float vf);
void Vector_Float_PushBack(Vector_Float vf, float val);
Node_Float *Vector_Float_Visit(Vector_Float vf, unsigned int i);
void Vector_Float_Resize(Vector_Float vf, int n);
void Vector_Float_Assign(Vector_Float vf, Vector_Float v);
void Vector_Float_Clear(Vector_Float vf);

#endif
