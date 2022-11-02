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

#ifndef LINEARALGEBRAFUNCS_H
#define LINEARALGEBRAFUNCS_H

#include "Common.h"

//#define DEBUG
typedef double REAL;

typedef struct Matrix
{
    REAL *data; //data array stored in a row major way,
               // i.e., the first n values forms the first row
    int m; //number of rows
    int n; //number of cols
} Matrix;
Matrix * newMatrix(int m, int n);
void createMatrixFromData(Matrix *X, REAL *data, int m, int n);
void deleteMatrix(Matrix *X);
void printMatrix(Matrix *X);
void printMatrixWithName(Matrix *X, char* name);
bool_t checkMatrix(Matrix *X);
bool_t matrixTranspose(Matrix *X, Matrix *res);
bool_t matrixProductValue(Matrix *X, REAL value, Matrix* res);
bool_t matrixProduct(Matrix* A, Matrix* B, Matrix* res);
bool_t matrixSubtract(Matrix* A, Matrix* B, Matrix* res);
bool_t matrixAlmostEqual(Matrix* A, Matrix* B);

bool_t sqEuclideanDistance(Matrix* X, Matrix* Y, Matrix* dist);

//!
//! \brief Compute the Cholesky decomposition of a matrix A. (A=LL*)
//! \param [in] A
//! \param [out] L
//! \return True or false shows if successful.
//!
bool_t cholesky(Matrix* A, Matrix* L);

//!
//! \brief Solve the linear equations A x = b, given the Cholesky factorization of A.
//! \param [in] L: Cholesky factorization of A. (A=LL*)
//! \param [in] b: Right-hand side
//! \param [out] X
//! \return True or false shows if successful.
//!
bool_t cho_solve(Matrix* L, Matrix* b, Matrix* x);

#endif //LINEARALGEBRAFUNCS_H
