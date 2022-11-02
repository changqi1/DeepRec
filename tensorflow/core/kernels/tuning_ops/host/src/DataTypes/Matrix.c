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

#include "Matrix.h"

Matrix *Matrix_Ctor(void)
{
    Matrix *self = (Matrix *)MALLOC(sizeof(Matrix));
    self->m_buf = nullptr;
    self->m_len = 0;
    memset(self->m_arrSize, 0, sizeof(int) * 3);
    memset(self->m_arrStride, 0, sizeof(int) * 3);
    return self;
}

Matrix *Matrix_Ctor_Param(void *buf, int len)
{
    Matrix *self = (Matrix *)MALLOC(sizeof(Matrix));
    self->m_buf = buf;
    self->m_len = len;
    memset(self->m_arrSize, 0, sizeof(int) * 3);
    memset(self->m_arrStride, 0, sizeof(int) * 3);
    return self;
}

void Matrix_Dtor(Matrix *self)
{
    FREE(self);
}
// TODO
Matrix *createMatrixFromBuffer(void *p, int dimX, int dimY)
{
    Matrix *ptr = Matrix_Ctor_Param(p, dimX * dimY);
    Matrix *temp = (Matrix *)p;
    FREE(temp->m_buf);
    temp = nullptr;
    ptr->m_arrSize[0] = dimX;
    ptr->m_arrSize[1] = dimY;
    ptr->m_arrSize[2] = 1;
    ptr->m_arrStride[0] = 1;
    ptr->m_arrStride[1] = dimX;
    ptr->m_arrStride[2] = 1;
    return ptr;
}
