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

#ifndef DEOPTIMIZER_H
#define DEOPTIMIZER_H

#include "OptimizerIF.h"
#include "Optimizer.h"
#include "Mutate.h"
#include "Crossover.h"
#include "Sampling.h"
#include "Common.h"
#include "VectorIndividual.h"

typedef struct DEOptimizer
{
    ParamOptimizer base;

    void (*crossover)(struct DEOptimizer *self);
    void (*mutation)(struct DEOptimizer *self);
    void (*initgroup)(struct DEOptimizer *self);

    int retry;
    bool over_flag;
    int m_iGenNum;
    int m_iPopSize;
    int m_iGenIdx;
    float m_dcr_h;
    float m_dcr_l;
    float m_df_h;
    float m_df_l;
    Vector_Individual m_vPop;
    Vector_Individual m_vMutationPop;
    Vector_Individual m_vCrossOverPop;
    Vector_Pair_StringToInt m_updateindex;
    Vector_Pair_StringToInt m_waitingindex;
    bool_t m_initPop;
    bool_t m_bExit;
    float m_bestFitness;
    float m_worstFitness;
    float m_averageFitness;
    Map_StringToPtr m_mbestParam;
    void (*m_funcSampling)(Vector_Vector_Float, int sample_iter);
} DEOptimizer;

DEOptimizer *DEOptimizer_Ctor(OptParam *param);
void DEOptimizer_Dtor(DEOptimizer *self);

#endif
