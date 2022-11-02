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

#ifndef CROSSOVER_H
#define CROSSOVER_H

#ifdef KERNEL_MODULE
//#include "math.h"
#else
#include <math.h>
#endif
#include "VectorFloat.h"
#include "Individual.h"
#include "Common.h"

void Crossover_adaptive_simulatedBinaryCrossover(Individual *p1, Individual *p2, float eta);
void Crossover_simulatedBinaryCrossover(Individual *p1, Individual *p2, float eta);
void Crossover_pointwiseCrossover(Individual *p1, Individual *p2, float eta);

#endif
