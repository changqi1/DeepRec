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

#include "Kernels.h"

bool_t cal_RBF_K(Kernel *self,
               Matrix *X, Matrix *Y,
               Matrix *K);

void Kernel_Ctor(Kernel *self) {
    if (self) {
        self->K = nullptr; // K is pure virtual function
    }
}

void Kernel_Dtor(Kernel *self) {
    if (self) {
        self->K = nullptr;
    }
}

RBFKernel *RBFKernel_Ctor(REAL length_scale) {
    RBFKernel *self = (RBFKernel *)MALLOC(sizeof(RBFKernel));
    if (self) {
        Kernel_Ctor(&(self->base));
        self->base.K = cal_RBF_K;
        self->length_scale = length_scale;
    }
    return self;
}

void RBFKernel_Dtor(RBFKernel *self) {
    if (NULL == self) {
        return;
    }
    Kernel_Dtor(&(self->base));
    FREE(self);
    self = NULL;
}

bool_t cal_RBF_K(Kernel *self,
               Matrix *X, Matrix *Y,
               Matrix *K)
{
    int i, j;
    RBFKernel *derived = (RBFKernel *)self;
    if (NULL == derived) {
        PRINTF("cal_RBF_K: invalid self\n");
        return false_t;
    }

    if (!(checkMatrix(X) && checkMatrix(Y) && checkMatrix(K))) {
        PRINTF("cal_RBF_K: invalid X, Y or K\n");
        return false_t;
    }

    if (X->n != Y->n) {
        PRINTF("cal_RBF_K: X and Y should have same number of cols\n");
        return false_t;
    }

    if ((K->m != X->m) || (K->n != Y->m)) {
        PRINTF("cal_RBF_K: K should have shape (X->m, Y->m)\n");
        return false_t;
    }

    matrixProductValue(X, 1 / derived->length_scale, nullptr);
    matrixProductValue(Y, 1 / derived->length_scale, nullptr);

    sqEuclideanDistance(X, Y, K);

    for (i = 0; i < K->m; ++i) {
        for (j = 0; j < K->n; ++j) {
            REAL sqdist = K->data[i * K->n + j];
            K->data[i * K->n + j] = exp(-0.5 * sqdist);
        }
    }
    return true_t;
}
