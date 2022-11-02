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

#include "AcquisitionAlgo.h"

void AcquisitionAlgo_Ctor(AcquisitionAlgo* self) {
    if (self) {
        self->acquisition = nullptr;
    }
}

void AcquisitionAlgo_Dtor(AcquisitionAlgo* self) {
    if (self) {
        self->acquisition = nullptr;
    }
}

bool_t acquisition_gp_mi(AcquisitionAlgo *self, Matrix *y_mean, Matrix *y_cov, Matrix *scores, int *best_idx) {
    GP_MI *mi;
    REAL best_score;
    int best_i, i;
    if (NULL == self) {
        PRINTF("acquisition_gp_mi: invalid self pointer\n");
        return false_t;
    }

    if (!(checkMatrix(y_mean) && checkMatrix(y_cov) && checkMatrix(scores))) {
        PRINTF("acquisition_gp_mi: invalid matrix data\n");
        return false_t;
    }

    if ((y_mean->m != y_cov->m) || (y_mean->n != 1) ||
        (y_cov->m != y_cov->n) ||
        (scores->m != y_mean->m) || (scores->n != 1))
    {
        PRINTF("acquisition_gp_mi: invalid matrix shape.\n");
        return false_t;
    }

    mi = (GP_MI *)self;

    best_score = 0.0;
    best_i = 0;
    for (i = 0; i < y_mean->m; ++i) {
        REAL mean = y_mean->data[i];
        REAL var = y_cov->data[i * y_cov->n + i];
        REAL phi = sqrt(mi->alpha) * (sqrt(var + mi->gama) - sqrt(mi->gama));
        scores->data[i] = mean + phi;

        if (i == 0) {
            best_score = scores->data[i];
            best_i = i;
        }
        else if (scores->data[i] > best_score) {
            best_score = scores->data[i];
            best_i = i;
        }
    }
    *best_idx = best_i;

    return true_t;
}


GP_MI* GP_MI_Ctor(REAL alpha) {
    GP_MI *mi = (GP_MI *)MALLOC(sizeof(GP_MI));
    if (mi) {
        mi->base.acquisition = acquisition_gp_mi;
        mi->alpha = alpha;
        mi->gama = 0.0;
    }
    return mi;
}

void GP_MI_Dtor(GP_MI* self) {
    if (self) {
        self->base.acquisition = nullptr;
    }
    FREE(self);
    self = nullptr;
}
