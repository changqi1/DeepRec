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

#ifndef MATRIX_H
#define MATRIX_H

#include "Common.h"

typedef struct Matrix {
  void *m_buf;
  int m_len;
  int m_arrSize[3];
  int m_arrStride[3];
} Matrix;

Matrix *Matrix_Ctor(void);
Matrix *Matrix_Ctor_Param(void *buf, int len);
void Matrix_Dtor(Matrix *self);
Matrix *createMatrixFromBuffer(void *p, int dimX, int dimY);

#endif
