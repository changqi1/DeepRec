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

#ifndef VECTORVECTORSTRING_H
#define VECTORVECTORSTRING_H
#include "VectorString.h"
#include "Common.h"

typedef struct Node_Vector_String {
  Vector_String m_vs;
  struct Node_Vector_String *m_next;
} Node_Vector_String, *Vector_Vector_String;

Vector_Vector_String Vector_Vector_String_Ctor(void);
void Vector_Vector_String_Dtor(Vector_Vector_String vvs);
void Vector_Vector_String_Print(Vector_Vector_String vvs);
void Vector_Vector_String_PushBack(Vector_Vector_String vvs, Vector_String vs);

#endif
