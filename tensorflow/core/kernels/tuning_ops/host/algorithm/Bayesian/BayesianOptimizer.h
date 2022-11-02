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

#ifndef BAYESIANOPTIMIZER_H
#define BAYESIANOPTIMIZER_H

#include "OptimizerIF.h"
#include "Optimizer.h"
#include "Sampling.h"
#include "VectorIndividual.h"
#include "VectorMap.h"
#include "GPR/GPR.h"

//#define DEBUG_BO
#define TRAIN_NUM_MAX 1000
#define PARAM_NUM_MAX 10
#define TARGET_NUM_MAX 1
#define X_TRAIN_SIZE_MAX (TRAIN_NUM_MAX*PARAM_NUM_MAX)
#define Y_TRAIN_SIZE_MAX (TRAIN_NUM_MAX*TARGET_NUM_MAX)
#define SAMPLE_NUM 10

typedef struct BayesianOptimizer
{
    ParamOptimizer base;

    //void (*dump_iter_to_csv)(struct BayesianOptimizer *self, Individual *p);
    void (*initSample)(struct BayesianOptimizer *self);
    void (*generatePop)(struct BayesianOptimizer *self);
    int m_iIterNum;
    int m_iRandomState;
    int m_iIterIdx;
    Vector_Map_StringToPtr m_vSampleParam;
    Individual *m_best;
    Vector_Individual m_vPop;
    void (*m_funcSampling)(Vector_Vector_Float, int sample_iter);
    int update_step;

    Matrix* X_train;
    Matrix* y_train;
    Kernel* m_kernel;
    AcquisitionAlgo* m_acquiAlgo;
    GPR* m_gpr;
} BayesianOptimizer;

BayesianOptimizer *BayesianOptimizer_Ctor(OptParam *param);
void BayesianOptimizer_Dtor(BayesianOptimizer *self);

bool_t BayesianOptimizer_update_training_data(BayesianOptimizer *self, Individual* pop);

#endif //BAYESIANOPTIMIZER_H
