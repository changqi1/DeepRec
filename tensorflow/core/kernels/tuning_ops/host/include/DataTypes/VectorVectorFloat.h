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

#ifndef VECTORVECTORFLOAT_H
#define VECTORVECTORFLOAT_H
#include "VectorFloat.h"
#include "Common.h"

typedef struct Node_Vector_Float {
  Vector_Float m_vf;
  struct Node_Vector_Float *m_next;
} Node_Vector_Float, *Vector_Vector_Float;

Vector_Vector_Float Vector_Vector_Float_Ctor(void);
void Vector_Vector_Float_Dtor(Vector_Vector_Float vvf);
void Vector_Vector_Float_Print(Vector_Vector_Float vvf);
int Vector_Vector_Float_Size(Vector_Vector_Float vvf);
void Vector_Vector_Float_PushBack(Vector_Vector_Float vvf, Vector_Float vf);
Node_Vector_Float *Vector_Vector_Float_Visit(Vector_Vector_Float vvf, unsigned int i);
void Vector_Vector_Float_Resize(Vector_Vector_Float vvf, int n);

#endif
