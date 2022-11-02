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

#ifndef GAOPTIMIZER_H
#define GAOPTIMIZER_H

#include "OptimizerIF.h"
#include "Optimizer.h"
#include "Mutate.h"
#include "Crossover.h"
#include "Sampling.h"
#include "Common.h"
#include "VectorIndividual.h"

typedef struct GAOptimizer
{
    ParamOptimizer base;

    void (*generateOffPop)(struct GAOptimizer *self, int update_idx);
    void (*initgroup)(struct GAOptimizer *self);
    void (*updateOffSpring)(struct GAOptimizer *self);
    int m_iGenNum;
    int m_iPopSize;
    int m_iGenIdx;
    float m_dmutp;
    Vector_Individual m_vPop;
    Vector_Individual m_vOffPop;
    Vector_Pair_StringToInt m_updateindex;
    Vector_Pair_StringToInt m_waitingindex;
    int m_offSize;
    Individual *m_best;
    bool_t m_bExit;
    bool_t m_initPop;
    int m_rec_idx;
    //void (*m_funcMutate)(Individual *p, float mutp, int eta);
    void (*m_funcMutate)(Individual *p, float mutp, float eta);
    void (*m_funcCrossover)(Individual *p1, Individual *p2, float eta);
    void (*m_funcSampling)(Vector_Vector_Float, int sample_iter);
} GAOptimizer;

GAOptimizer *GAOptimizer_Ctor(OptParam *param);
void GAOptimizer_Dtor(GAOptimizer *self);

#endif
