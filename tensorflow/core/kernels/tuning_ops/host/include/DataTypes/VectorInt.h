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

#ifndef VECTORINT_H
#define VECTORINT_H

#include "Common.h"

typedef struct Node_Int {
  int *m_val;
  struct Node_Int *m_next;
} Node_Int, *Vector_Int;

Vector_Int Vector_Int_Ctor(void);
void Vector_Int_Dtor(Vector_Int vi);
int Vector_Int_Size(Vector_Int vi);
void Vector_Int_Print(Vector_Int vi);
void Vector_Int_PushBack(Vector_Int vi, int val);
Node_Int *Vector_Int_Visit(Vector_Int vi, unsigned int i);
void Vector_Int_RandomShuffle(Vector_Int vi);
Node_Int *Vector_Int_Find(Vector_Int vi, int val);
Node_Int *Vector_Int_Erase(Vector_Int vi, Node_Int *p);
void Vector_Int_Clear(Vector_Int vi);
bool_t Vector_Int_Next_Permutation(Vector_Int vi);
void Vector_Int_Resize(Vector_Int vi, int n);
void Vector_Int_Assign(Vector_Int vi, Vector_Int v);

#endif
