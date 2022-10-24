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

#ifndef VECTORVECTORINT_H
#define VECTORVECTORINT_H
#include "VectorInt.h"
#include "Common.h"

typedef struct Node_Vector_Int {
  Vector_Int m_vi;
  struct Node_Vector_Int *m_next;
} Node_Vector_Int, *Vector_Vector_Int;

Vector_Vector_Int Vector_Vector_Int_Ctor(void);
void Vector_Vector_Int_Dtor(Vector_Vector_Int vvi);
void Vector_Vector_Int_Print(Vector_Vector_Int vvi);
int Vector_Vector_Int_Size(Vector_Vector_Int vvi);
void Vector_Vector_Int_PushBack(Vector_Vector_Int vvi, Vector_Int vi);
Node_Vector_Int *Vector_Vector_Int_Visit(Vector_Vector_Int vvi, unsigned int i);
void Vector_Vector_Int_Resize(Vector_Vector_Int vvi, int n);

#endif
