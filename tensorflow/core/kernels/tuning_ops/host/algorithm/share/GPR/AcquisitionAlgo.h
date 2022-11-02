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

#ifndef ACQUISITIONALGO_H
#define ACQUISITIONALGO_H

#include "LinearAlgebraFuncs.h"

//!
//! \struct AcquisitionAlgo
//! \brief Base class of all acquisition algorithms
//!
typedef struct AcquisitionAlgo
{
    //!
    //! \brief Acquisition function which evaluates candidate samples.
    //! \param [in] y_mean: result of GPR predict.
    //! \param [in] y_cov: result of GPR predict.
    //! \param [out] scores: score of each query points (with shape (n_samples,).
    //! \param [out] best_idx: index of the best sample from the candidates.
    //! \return true or false shows whether function succeeds.
    //!
    bool_t (*acquisition)(struct AcquisitionAlgo *self, Matrix *y_mean, Matrix *y_cov, Matrix *scores, int *best_idx);
} AcquisitionAlgo;
void AcquisitionAlgo_Ctor(AcquisitionAlgo* self);
void AcquisitionAlgo_Dtor(AcquisitionAlgo* self);

//!
//! \struct GP_MI (Gaussian Process Mutual Information) algorithm
//! \brief The implementation is based on Algorithm 1 of
//! paper "Gaussian Process Optimization with Mutual Information" by Emile Contal ect.
//!
typedef struct GP_MI
{
    AcquisitionAlgo base;
    REAL alpha;
    REAL gama;
} GP_MI;
GP_MI* GP_MI_Ctor(REAL alpha);
void GP_MI_Dtor(GP_MI* self);




#endif //ACQUISITIONALGO_H
