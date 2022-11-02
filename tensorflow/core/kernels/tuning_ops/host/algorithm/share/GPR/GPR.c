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

#include "GPR.h"

bool_t fit(GPR *gpr, Matrix *X, Matrix *y);
bool_t predict(GPR *gpr, Matrix *X, Matrix *y_mean, Matrix *y_cov);
int select_best(GPR *gpr, Matrix *X_train, Matrix *y_train, Matrix *X_samples);

GPR* GPR_Ctor(Kernel *kernel, AcquisitionAlgo *acquiAlgo, REAL var_noise) {
    GPR* gpr;
    if (NULL == kernel) {
        PRINTF("GPR_Ctor: invalid parameter\n");
        return NULL;
    }
    gpr = (GPR*) MALLOC(sizeof(GPR));
    if (gpr) {
        gpr->kernel = kernel;
        gpr->acquiAlgo = acquiAlgo;
        gpr->var_n = var_noise;
        gpr->fit = fit;
        gpr->predict = predict;
        gpr->select_best = select_best;
        gpr->X_train = nullptr;
        gpr->y_train = nullptr;
        gpr->L = nullptr;
        gpr->alpha = nullptr;
    }
    return gpr;
}

void GPR_Dtor(GPR *gpr) {
    if (NULL == gpr) {
        return;
    }
//    if (gpr->X_train && gpr->X_train->data) {
//        FREE(gpr->X_train->data);
//    }
//    if (gpr->y_train && gpr->y_train->data) {
//        FREE(gpr->y_train->data);
//    }
    if (gpr->L) {
        deleteMatrix(gpr->L);
    }
    if (gpr->alpha) {
        deleteMatrix(gpr->alpha);
    }
    FREE(gpr);
    gpr = NULL;
}

bool_t fit(GPR *gpr, Matrix *X, Matrix *y) {
    int N, i;
    Matrix* K;
    bool_t chol_ok;
    if (NULL == gpr) {
        PRINTF("fit: invalid gpr pointer\n");
        return false_t;
    }

    if (!(checkMatrix(X) && checkMatrix(y))) {
        PRINTF("fit: invalid matrix data\n");
        return false_t;
    }

    if ((y->m != X->m) || (y->n != 1)) {
        PRINTF("fit: target value y should have same rows as X and only one col (multiple targets are not suppoeted).\n");
        return false_t;
    }

    gpr->X_train = X;
    gpr->y_train = y;

    N = X->m;
    K = newMatrix(N, N);
    gpr->kernel->K(gpr->kernel, X, X, K);
    for (i = 0; i < K->m; ++i) {
        K->data[i * K->m + i] += gpr->var_n;
    }
    printMatrixWithName(K, "K + var_n*I");

    //Algorithm 2.1 Line 2: L = cholesky(K + var_n*I)
    if (gpr->L) {
        deleteMatrix(gpr->L);
    }
    gpr->L = newMatrix(N, N);
    chol_ok = cholesky(K, gpr->L);
    if (!chol_ok) {
        PRINTF("fit: fail to calculate the cholesky decomposition of matrix (K + var_n*I).\n");
        deleteMatrix(K);
        return false_t;
    }
    printMatrixWithName(gpr->L, "L = cholesky(K + var_n*I)");

    //Algorithm 2.1 Line 3: alpha = cho_solve(L, y)
    if (gpr->alpha) {
        deleteMatrix(gpr->alpha);
    }
    gpr->alpha = newMatrix(N, 1);
    cho_solve(gpr->L, y, gpr->alpha);
    printMatrixWithName(gpr->alpha, "alpha");

    deleteMatrix(K);
    return true_t;
}


bool_t predict(GPR *gpr, Matrix *X, Matrix *y_mean, Matrix *y_cov) {
    Matrix* K_trans;
    Matrix* K;
    Matrix* v;
    Matrix* K_trans_times_v;
    Matrix* K_XX;
    if (NULL == gpr) {
        PRINTF("predict: invalid gpr pointer\n");
        return false_t;
    }

    if (!(checkMatrix(X) && checkMatrix(y_mean) && checkMatrix(y_cov))) {
        PRINTF("predict: invalid matrix data\n");
        return false_t;
    }

    if ((X->n != gpr->X_train->n) ||
        (y_mean->m != X->m) || (y_mean->n != 1) ||
        (y_cov->m != X->m) || (y_cov->n != y_cov->m))
    {
        PRINTF("predict: invalid matrix shape.\n");
        return false_t;
    }

    //K_trans = K(X, X_train), K = K(X_train, X) = transpose(K_trans)
    K_trans = newMatrix(X->m, gpr->X_train->m);
    gpr->kernel->K(gpr->kernel, X, gpr->X_train, K_trans);
    printMatrixWithName(K_trans, "K_trans");

    //Algorithm 2.1 Line 4: y_mean = f_star = K_trans x alpha
    matrixProduct(K_trans, gpr->alpha, y_mean);

    K = newMatrix(K_trans->n, K_trans->m);
    matrixTranspose(K_trans, K);
    printMatrixWithName(K, "K");

    //Algorithm 2.1 Line 5: rewrite A=LL_T, Av=K
    //So, here, v=L_T\(L\K)
    v = newMatrix(K->m, K->n);
    cho_solve(gpr->L, K, v);
    printMatrixWithName(v, "v");

    //Algorithm 2.1 Line 6: according to equation (2.26),
    // y_cov=K(X,X)-K_trans(A^-1)K
    // since K=Av, (A^-1)K=v,
    // so, y_cov=K(X,X)-K_trans_times_v
    K_trans_times_v = newMatrix(K_trans->m, v->n);
    matrixProduct(K_trans, v, K_trans_times_v);
    printMatrixWithName(K_trans_times_v, "K_trans_times_v");

    K_XX = newMatrix(X->m, X->m);
    gpr->kernel->K(gpr->kernel, X, X, K_XX);
    printMatrixWithName(K_XX, "K_XX");

    matrixSubtract(K_XX, K_trans_times_v, y_cov);

    deleteMatrix(K_trans);
    deleteMatrix(K);
    deleteMatrix(v);
    deleteMatrix(K_trans_times_v);
    deleteMatrix(K_XX);

    return true_t;
}

int select_best(GPR *gpr, Matrix *X_train, Matrix *y_train, Matrix *X_samples) {
    int best_sample_index = -1;
    Matrix* y_mean = newMatrix(X_samples->m, 1);
    Matrix* y_cov = newMatrix(X_samples->m, X_samples->m);
    Matrix* sample_scores = newMatrix(X_samples->m, 1);
    bool_t fit_ok, predict_ok, acqui_ok;

    do{
        if (NULL == gpr) {
            PRINTF("select_best: invalid gpr pointer\n");
            break;
        }

        fit_ok = gpr->fit(gpr, X_train, y_train);
        if (!fit_ok) {
            PRINTF("select_best: fail to fit with training data\n");
            break;
        }

        predict_ok = gpr->predict(gpr, X_samples, y_mean, y_cov);
        if(!predict_ok) {
            PRINTF("select_best: fail to predict\n");
            break;
        }

        acqui_ok = gpr->acquiAlgo->acquisition(gpr->acquiAlgo, y_mean, y_cov, sample_scores, &best_sample_index);
        if (!acqui_ok) {
            PRINTF("select_best: fail to select the best sample by using acquisition function\n");
            break;
        }
    }while(false_t);

//    PRINTF("[select_best] X_train = \n");
//    printMatrix(X_train);
//    PRINTF("[select_best] y_train = \n");
//    printMatrix(y_train);
//    PRINTF("[select_best] X_samples = \n");
//    printMatrix(X_samples);
//    PRINTF("[select_best] y_mean = \n");
//    printMatrix(y_mean);
//    PRINTF("[select_best] y_cov = \n");
//    printMatrix(y_cov);
//    PRINTF("[select_best] sample_scores = \n");
//    printMatrix(sample_scores);

    deleteMatrix(y_mean);
    deleteMatrix(y_cov);
    deleteMatrix(sample_scores);
    return best_sample_index;
}
