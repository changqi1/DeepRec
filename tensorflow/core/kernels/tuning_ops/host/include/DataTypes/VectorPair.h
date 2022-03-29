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

#ifndef VECTORPAIR_H
#define VECTORPAIR_H

#include "Common.h"

typedef struct ps2i {
  char *m_string;
  int m_val;
} ps2i;

typedef struct Node_Pair_StringToInt {
  ps2i *m_pair;
  struct Node_Pair_StringToInt *m_next;
} Node_Pair_StringToInt, *Vector_Pair_StringToInt;

Vector_Pair_StringToInt Vector_Pair_StringToInt_Ctor(void);
void Vector_Pair_StringToInt_Dtor(Vector_Pair_StringToInt vpsi);
int Vector_Pair_StringToInt_Size(Vector_Pair_StringToInt vpsi);
void Vector_Pair_StringToInt_PushBack(Vector_Pair_StringToInt vpsi, ps2i *m_pair);
void Vector_Pair_StringToInt_PushBack_param(Vector_Pair_StringToInt vpsi, char *key, int value);
Vector_Pair_StringToInt Vector_Pair_StringToInt_Erase(Vector_Pair_StringToInt vpsi, Node_Pair_StringToInt *p);
void Vector_Pair_StringToInt_Print(Vector_Pair_StringToInt vpsi);

#endif
