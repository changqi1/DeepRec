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

#ifndef SIMPLEVECTOR_H
#define SIMPLEVECTOR_H

#include "Common.h"

typedef struct Node_String {
  char *m_string;
  struct Node_String *m_next;
} Node_String, *Vector_String;

Vector_String Vector_String_Ctor(void);
void Vector_String_Dtor(Vector_String vs);
int Vector_String_Size(Vector_String vs);
void Vector_String_Print(Vector_String vs);
void Vector_String_PushBack(Vector_String vs, char *str);
bool_t Vector_String_Cmp(Vector_String vs1, Vector_String vs2);
Node_String *Vector_String_Visit(Vector_String vs, unsigned int i);

#endif
