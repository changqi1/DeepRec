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

#ifndef GPR_H
#define GPR_H

#include "Kernels.h"
#include "AcquisitionAlgo.h"
#include "LinearAlgebraFuncs.h"

//!
//! \struct GPR
//! \brief Gaussian process regression (GPR)
//! \details The implementation is based on Algorithm 2.1 of Gaussian Processes
//! for Machine Learning (GPML) by Rasmussen and Williams.
//!
typedef struct GPR
{
    //!
    //! \brief The kernel specifying the covariance function of the GP.
    //!
    Kernel *kernel;

    //!
    //! \brief Acquisition algo used by GPR.
    //!
    AcquisitionAlgo *acquiAlgo;

    //!
    //! \brief The variance of additional Gaussian
    //! \brief measurement noise on the training observations.
    //!
    REAL var_n;

    //!
    //! \brief The training data with shape (n_samples, n_features)
    //! \brief (also required for prediction)
    //!
    Matrix *X_train;

    //!
    //! \brief Target values in training data with shape (n_samples,)
    //! \brief (also required for prediction)
    //!
    Matrix *y_train;

    //!
    //! \brief Lower-triangular Cholesky decomposition of the kernel
    //! \brief with shape (n_samples, n_samples)
    //!
    Matrix *L;

    //!
    //! \brief Dual coefficients of training data points in kernel space
    //! \brief with shape (n_samples,)
    //!
    Matrix *alpha;

    //!
    //! \brief Fit Gaussian process regression model.
    //! \param [in] X: Feature vectors of training data (with shape (n_samples, n_features)).
    //! \param [in] y: Target values (with shape (n_samples,)).
    //! \return true or false shows whether function succeeds.
    //!
    bool_t (*fit)(struct GPR *gpr, Matrix *X, Matrix *y);

    //!
    //! \brief Predict using the Gaussian process regression model.
    //! \param [in] X: Query points where the GP is evaluated (with shape (n_samples, n_features)).
    //! \param [out] y_mean: Mean of predictive distribution at query points (with shape (n_samples,)).
    //! \param [out] y_cov: Covariance of joint predictive distribution a query points. (with shape (n_samples, n_samples)).
    //! \return true or false shows whether function succeeds.
    //!
    bool_t (*predict)(struct GPR *gpr, Matrix *X, Matrix *y_mean, Matrix *y_cov);

    //!
    //! \brief Select the best sample from the candidate samples.
    //! \param [in] X_train: Training data (with shape (n_training_samples, n_features)).
    //! \param [in] y_train: Target values (with shape (n_training_samples,)).
    //! \param [in] X_samples: Candidate samples (with shape (n_candidate_samples, n_features)).
    //! \return best_sample_index: >=0 if succeeds, -1 if fails.
    //!
    int (*select_best)(struct GPR *gpr, Matrix *X_train, Matrix *y_train, Matrix *X_samples);
} GPR;
GPR* GPR_Ctor(Kernel *kernel, AcquisitionAlgo *acAlgo, REAL var_noise);
void GPR_Dtor(GPR *self);

#endif //GPR_H
