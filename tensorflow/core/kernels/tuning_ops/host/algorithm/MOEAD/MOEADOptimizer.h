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

#ifndef MOEADOPTIMIZER_H
#define MOEADOPTIMIZER_H

#include "OptimizerIF.h"
#include "Optimizer.h"
#include "Mutate.h"
#include "Crossover.h"
#include "Sampling.h"
#include "Common.h"
#include "VectorIndividual.h"
#include "VectorVectorInt.h"
#include "VectorVectorFloat.h"
#include "VectorVectorString.h"

typedef struct MOEADOptimizer
{
    ParamOptimizer base;

    void (*generateUpdatePop)(struct MOEADOptimizer *self, int index);
    bool_t (*isDominates)(struct MOEADOptimizer *self, Individual *pop1, Individual *pop2);
    void (*initParetoFront)(struct MOEADOptimizer *self);
    void (*updateParetoFront)(struct MOEADOptimizer *self, Individual *p);
    void (*initgroup)(struct MOEADOptimizer *self);
    void (*generate_neighbor)(struct MOEADOptimizer *self);
    void (*update_neighbor_tchebi)(struct MOEADOptimizer *self, int idx, Individual *p);
    void (*update_neighbor_pbi)(struct MOEADOptimizer *self, int idx, Individual *p);
    void (*uniform_weight_generate)(struct MOEADOptimizer *self);

    int m_iGenNum;
    int m_iPopSize;
    int m_iObjNum;
    int m_iNeighNum;
    int m_iGenIdx;
    int m_iPopIdx;
    int m_iUpdatePopIdx;
    double m_dcr;
    double m_dmut;
    double m_dmutp;
    bool_t m_initPop;
    Vector_Individual m_vPop;
    Vector_Individual m_vUpdatePop;
    Vector_Pair_StringToInt m_updateindex;
    Vector_Pair_StringToInt m_waitingindex;
    Vector_Individual m_vPF;
    Vector_Vector_Int m_vNeighbor;
    Vector_Vector_Float m_vWeight;
    Vector_Float m_bestFitness;
    bool_t m_bExit;
    void (*m_funcMutate)(Individual *p, float mutp, float eta);
    void (*m_funcCrossover)(Individual *p1, Individual *p2, float eta);
    void (*m_funcSampling)(Vector_Vector_Float, int sample_iter);
} MOEADOptimizer;

MOEADOptimizer *MOEADOptimizer_Ctor(OptParam *param);
void MOEADOptimizer_Dtor(MOEADOptimizer *self);

#endif
