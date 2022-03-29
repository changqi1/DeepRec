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

#ifndef VECTORINDIVIDUAL_H
#define VECTORINDIVIDUAL_H

#include "Individual.h"

typedef struct Node_Individual {
  Individual *m_indi;
  struct Node_Individual *m_next;
} Node_Individual, *Vector_Individual;

Vector_Individual Vector_Individual_Ctor(void);
void Vector_Individual_Dtor(Vector_Individual vi);
Node_Individual *Vector_Individual_Visit(Vector_Individual vi, unsigned int n);
int Vector_Individual_Size(Vector_Individual vi);
void Vector_Individual_RandomShuffle(Vector_Individual vi);
void Vector_Individual_PushBack(Vector_Individual vi, Individual *i);
void Vector_Individual_Resize(Vector_Individual vi, int n);
void Vector_Individual_Erase(Vector_Individual *p_vi, Individual *i);
void Vector_Individual_Print(Vector_Individual vi);

#endif
