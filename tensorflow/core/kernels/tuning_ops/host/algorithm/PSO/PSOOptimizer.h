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

#ifndef PSOOPTIMIZER_H
#define PSOOPTIMIZER_H

#include "OptimizerIF.h"
#include "Optimizer.h"
#include "Sampling.h"
#include "VectorMap.h"

typedef struct PSOOptimizer
{
    ParamOptimizer base;
    void (*initSwarms)(struct PSOOptimizer *self);
    void (*generateSwarm)(struct PSOOptimizer *self, int SwarmIdx);

    int m_iIterNum;
    int m_iSwarmNum;
    int m_iIterIdx;
    int m_iSwarmIdx;
    float m_dStepCount;
    float m_dStepMin;
    float m_dUpdateWeight;
    float m_dUpdateCp;
    float m_dUpdateCg;
    bool_t m_initSwarm;
    bool_t m_rec;
    Vector_Map_StringToPtr m_mapSwarmVelocity;
    Vector_Map_StringToPtr m_mapSwarmParam;
    Vector_Float m_vPTarget;
    Vector_Map_StringToPtr m_vmSwarmBestParam;
    float m_GTarget;
    Map_StringToPtr m_vmGlobalBestParam;
    void (*m_funcSampling)(Vector_Vector_Float p, int sample_iter);
    Vector_Int m_updateindex;
    Vector_Int m_waitingindex;
} PSOOptimizer;
PSOOptimizer *PSOOptimizer_Ctor(OptParam *param);
void PSOOptimizer_Dtor(PSOOptimizer *self);

#endif
