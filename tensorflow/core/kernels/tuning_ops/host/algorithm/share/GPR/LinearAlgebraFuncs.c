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

#include "LinearAlgebraFuncs.h"

#define DELTA 0.00002

Matrix * newMatrix(int m, int n) {
    Matrix * X = (Matrix *)MALLOC(sizeof(struct Matrix));
    if (X) {
        int data_size = m*n*sizeof(REAL);
        X->data = (REAL *)MALLOC(data_size);
        memset(X->data, 0, data_size);
        X->m = m;
        X->n = n;
    }
    return X;
}

void createMatrixFromData(Matrix *X, REAL *data, int m, int n) {
    X->data = data;
    X->m = m;
    X->n = n;
}

void deleteMatrix(Matrix *X){
    if (NULL == X) {
        return;
    }
    if (X->data) {
        FREE(X->data);
        X->data = nullptr;
    }
    FREE(X);
    X = NULL;
}

bool_t checkMatrix(Matrix *X) {
    if (NULL == X){
        //PRINTF("checkMatrix: invalid pointer X\n");
        return false_t;
    }
    if (NULL == X->data || X->m <=0 || X->n <= 0) {
        //PRINTF("checkMatrix: invalid X content\n");
        return false_t;
    }
    return true_t;
}

void printMatrix(Matrix *X) {
    int i, j;
    if (!checkMatrix(X)){
        PRINTF("printMatrix: invalid Matrix X\n");
        return;
    }

    PRINTF("(%d, %d) = \n", X->m, X->n);
    for (i = 0; i < X->m; ++i) {
        for (j = 0; j < X->n; ++j) {
            PRINTF("%.15f, ", X->data[i * X->n + j]);
        }
        //PRINTF("\n");
    }
    PRINTF("\n");
}

void printMatrixWithName(Matrix *X, char* name) {
#ifdef DEBUG
    int i = 0;
    int j = 0;
    if (!checkMatrix(X)){
        PRINTF("printMatrix: invalid Matrix X\n");
        return;
    }

    PRINTF("%s (%d, %d) = \n", name, X->m, X->n);
    for (i = 0; i < X->m; ++i) {
        for (j = 0; j < X->n; ++j) {
            PRINTF("%.15f\t", X->data[i * X->n + j]);
        }
        PRINTF("\n");
    }
#endif
}

bool_t matrixTranspose(Matrix *X, Matrix *res) {
    int i, j;
    if (!(checkMatrix(X) && checkMatrix(res))) {
        PRINTF("matrixTranspose: invalid input matrices\n");
        return false_t;
    }

    if ((res->m != X->n) || (res->n != X->m)) {
        PRINTF("matrixTranspose: invalid input matrix shape\n");
        return false_t;
    }

    for (i = 0; i < res->m; ++i) {
        for (j = 0; j < res->n; ++j) {
            res->data[i * res->n + j] = X->data[j * X->n + i];
        }
    }
    return true_t;
}

bool_t matrixProductValue(Matrix *X, REAL value, Matrix* res){
    int i, j;
    REAL* ret;
    if (!checkMatrix(X)){
        PRINTF("matrixProductValue: invalid Matrix X\n");
        return false_t;
    }

    ret = NULL;
    if (checkMatrix(res)) {
        //If res is specified and valid, save product to res
        if (!(res->m == X->m && res->n == X->n)) {
            PRINTF("matrixProductValue: res is not same size with X\n");
            return false_t;
        }
        ret = res->data;
    }
    else {
        // save result to X
        ret = X->data;
    }

    for (i = 0; i < X->m; ++i) {
        for (j = 0; j < X->n; ++j) {
            ret[i * X->n + j] = X->data[i * X->n + j] * value;
        }
    }
    return true_t;
}

bool_t matrixProduct(Matrix* A, Matrix* B, Matrix* res) {
    int i, j, k;
    if (!(checkMatrix(A) && checkMatrix(B) && checkMatrix(res))) {
        PRINTF("matrixProduct: invalid input matrices\n");
        return false_t;
    }

    if ((A->n != B->m) || (res->m != A->m) || (res->n != B->n)) {
        PRINTF("matrixProduct: invalid input matrix shape\n");
        return false_t;
    }

    for (i = 0; i < res->m; ++i) {
        for (j = 0; j < res->n; ++j) {
            REAL sum = 0.0;
            for (k = 0; k < A->n; ++k) {
                sum += A->data[i * A->n + k] * B->data[k * B->n + j];
            }
            res->data[i * res->n + j] = sum;
        }
    }
    return true_t;
}

bool_t matrixSubtract(Matrix* A, Matrix* B, Matrix* res) {
    int M, N, i, j, offset;
    if (!(checkMatrix(A) && checkMatrix(B) && checkMatrix(res))) {
        PRINTF("matrixSubtract: invalid input matrices\n");
        return false_t;
    }

    if ((A->m != B->m) || (res->m != A->m) ||
            (A->n != B->n) || (res->n != A->n)) {
        PRINTF("matrixSubtract: invalid input matrix shape\n");
        return false_t;
    }

    M = res->m;
    N = res->n;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            offset = i * N + j;
            res->data[offset] = A->data[offset] - B->data[offset];
        }
    }
    return true_t;
}

bool_t matrixAlmostEqual(Matrix* A, Matrix* B) {
    int M, N, i, j, offset;
    if (!(checkMatrix(A) && checkMatrix(B))) {
        PRINTF("matrixAlmostEqual: invalid input matrices\n");
        return false_t;
    }

    if ((A->m != B->m) || (A->n != B->n)) {
        PRINTF("matrixAlmostEqual: invalid input matrix shape\n");
        return false_t;
    }

    M = A->m;
    N = A->n;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            REAL diff;
            offset = i * N + j;
            diff = A->data[offset] - B->data[offset];
            if (diff > DELTA || diff < -1.0 * DELTA) {
                PRINTF("matrixAlmostEqual: A[%d][%d](%.15f) != B[%d][%d](%.15f), diff = %.15f\n", i, j, A->data[offset], i, j, B->data[offset], diff);
                return false_t;
            }
        }
    }
    return true_t;
}

bool_t sqEuclideanDistance(Matrix* X, Matrix* Y, Matrix* dist)
{
    int i, j, n;
    if (!(checkMatrix(X) && checkMatrix(Y) && checkMatrix(dist))) {
        PRINTF("sqEuclideanDistance: invalid X, Y or dist\n");
        return false_t;
    }

    if (X->n != Y->n) {
        PRINTF("sqEuclideanDistance: X and Y should have same number of cols\n");
        return false_t;
    }

    if ((dist->m != X->m) || (dist->n != Y->m)) {
        PRINTF("sqEuclideanDistance: dist should have shape (X->m, Y->m)\n");
        return false_t;
    }

    for (i = 0; i < dist->m; ++i) {
        for (j = 0; j < dist->n; ++j) {
            REAL sum_sqdist = 0.0;
            for (n = 0; n < X->n; ++n) {
                sum_sqdist += pow(X->data[i * X->n + n] - Y->data[j * X->n + n], 2);
            }
            //dist[i * m_y + j] = sqrt(sum_sqdist);
            dist->data[i * dist->n + j] = sum_sqdist;
        }
    }
    return true_t;
}

//bool_t cholesky(Matrix* A, Matrix* L) {
//    int n = A->m;
//    int i = 0;
//    int j = 0;
//    int k = 0;
//
//    if (!(checkMatrix(A) && checkMatrix(L))) {
//        PRINTF("cholesky: invalid A or L\n");
//        return false_t;
//    }
//
//    if ((A->m != A->n) || (L->m != L->n) || (A->m != L->m)) {
//        PRINTF("cholesky: A and L should have same shape (m, m)\n");
//        return false_t;
//    }
//
//    n = A->m;
//
//    for (i = 0; i < n; i++)
//        for (j = 0; j < (i+1); j++) {
//            REAL s = 0;
//            for (k = 0; k < j; k++)
//                s += L->data[i * n + k] * L->data[j * n + k];
//            L->data[i * n + j] = (i == j) ?
//                           sqrt(A->data[i * n + i] - s) :
//                           (1.0 / L->data[j * n + j] * (A->data[i * n + j] - s));
//        }
//
//    return true_t;
//}
//
bool_t cholesky(Matrix* A, Matrix* L)
{
    REAL* matrix;
    REAL* lower;
    int N, i, j, k;
    if (!(checkMatrix(A) && checkMatrix(L))) {
        PRINTF("cholesky: invalid A or L\n");
        return false_t;
    }

    if ((A->m != A->n) || (L->m != L->n) || (A->m != L->m)) {
        PRINTF("cholesky: A and L should have same shape (m, m)\n");
        return false_t;
    }

    matrix = A->data;
    lower = L->data;
    N = A->m;

    // Decomposing a matrix into Lower Triangular
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL sum = 0.0;

            if (j == i) // summation for diagnols
            {
                for (k = 0; k < j; k++)
                    sum += pow(lower[j * N + k], 2);

                if (matrix[j * N + j] < sum) {
                    PRINTF("cholesky error: about to sqrt a negative value!\n");
                    return false_t;
                }
                lower[j * N + j] = sqrt(matrix[j * N + j] -
                                   sum);
            }
            else if (j < i) {
                // Evaluating L(i, j) using L(j, j)
                for (k = 0; k < j; k++)
                    sum += (lower[i * N + k] * lower[j * N + k]);
                lower[i * N + j] = (matrix[i * N + j] - sum) /
                              lower[j * N + j];
            }
            else {
                lower[i * N + j] = 0.0;
            }
        }
    }
    return true_t;
}

bool_t cho_solve(Matrix* L, Matrix* b, Matrix* x) {
    int M, N, n, m, i;
    Matrix* y;
    if (!(checkMatrix(L) && checkMatrix(b) && checkMatrix(x))) {
        PRINTF("cho_solve: invalid L, b or x\n");
        return false_t;
    }
    if ((L->m != L->n)
        || (b->m != L->n)
        || (x->m != b->m) || (x->n != b->n))
    {
        PRINTF("cho_solve: invalid input matrix shape\n");
        return false_t;
    }

    // To solve Ax=b, since A=LL*,
    // denote y=L*x,
    // first use forward substitution to solve Ly=b and get y,
    // then use backward sbustitution to solve L*x=y and get x.
    M = L->m;
    N = b->n;
    y = newMatrix(M, 1);
    for (n = 0; n < N; ++n) { //For nth col of x and nth col of b:
        for (m = 0; m < M; ++m) {
            REAL sum = 0.0;
            for (i = 0; i < m; ++i) {
                sum += L->data[m * M + i] * y->data[i];
            }
            y->data[m] = (b->data[m * N + n] - sum) / L->data[m * M + m];
        }

        for (m = M - 1; m >= 0; --m) {
            REAL sum = 0.0;
            for (i = m + 1; i < M; ++i) {
                sum += L->data[i * M + m] * x->data[i * N + n];
            }
            x->data[m * N + n] = (y->data[m] - sum) / L->data[m * M + m];
        }
    }

    deleteMatrix(y);
    return true_t;
}
