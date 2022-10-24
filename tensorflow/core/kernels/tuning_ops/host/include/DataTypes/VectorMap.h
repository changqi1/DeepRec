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

#ifndef VECTORMAP_H
#define VECTORMAP_H

#include "MapStringToPtr.h"
#include "Common.h"

typedef struct Node_Map_StringToPtr {
  Map_StringToPtr m_msp;
  struct Node_Map_StringToPtr *m_next;
} Node_Map_StringToPtr, *Vector_Map_StringToPtr;

Vector_Map_StringToPtr Vector_Map_StringToPtr_Ctor(void);
void Vector_Map_StringToPtr_Dtor(Vector_Map_StringToPtr vmsp);
void Vector_Map_StringToPtr_PushBack(Vector_Map_StringToPtr vmsp, Map_StringToPtr msp);
int Vector_Map_StringToPtr_Size(Vector_Map_StringToPtr vmsp);
Node_Map_StringToPtr *Vector_Map_StringToPtr_Visit(Vector_Map_StringToPtr vmsp, unsigned int i);
void Vector_Map_StringToPtr_Resize(Vector_Map_StringToPtr vmsp, int n);
void Vector_Map_StringToPtr_Erase(Vector_Map_StringToPtr *p_vmsp, Map_StringToPtr msp);
void Vector_Map_StringToPtr_Print(Vector_Map_StringToPtr vmsp);

#endif
