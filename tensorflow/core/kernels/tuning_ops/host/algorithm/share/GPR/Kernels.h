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

#ifndef KERNELS_H
#define KERNELS_H

#include "LinearAlgebraFuncs.h"

//!
//! \struct Kernel
//! \brief Base class of all kernels
//!
typedef struct Kernel
{
    bool_t (*K)(struct Kernel *self,
              Matrix *X, Matrix *Y,
              Matrix *K);
} Kernel;
void Kernel_Ctor(Kernel *self);
void Kernel_Dtor(Kernel *self);

//!
//! \struct RBFKernel
//! \brief Radial-basis function (RBF) kernel, aka squared-exponential (SE) kernel
//!
typedef struct RBFKernel
{
    Kernel base;
    REAL length_scale;
} RBFKernel;

RBFKernel *RBFKernel_Ctor(REAL length_scale);
void RBFKernel_Dtor(RBFKernel *self);

#endif
