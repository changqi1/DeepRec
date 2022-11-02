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

#include "BayesianOptimizer.h"

void BayesianOptimizer_update(ParamOptimizerIF *self)
{
    ParamOptimizer_update(self);
}

#if FLOAT_PARAM
void BayesianOptimizer_regist(ParamOptimizerIF *self, char *key, float min, float max, int (*update)(char *, float))
#else
void BayesianOptimizer_regist(ParamOptimizerIF *self, char *key, int min, int max, int (*update)(char *, int))
#endif
{
    int i;
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    OptimizedParamIF *p = (OptimizedParamIF *)OptimizedParam_Ctor(key, min, max, min, update, derived->base.optimize, derived->base.set_value);
    if (Map_StringToPtr_Find(derived->base.m_mapParam, key) == nullptr)
    {
        for (i = 0; i < derived->m_iRandomState; ++i)
        {
            Map_StringToPtr_PushBack(Vector_Map_StringToPtr_Visit(derived->m_vSampleParam, i)->m_msp, key, p->clone(p));
        }
    }
    else
    {
#if FLOAT_PARAM
        *(float *)(p->cur(p)) = *(float *)(Map_StringToPtr_Visit(derived->base.m_mapParam, key)->m_ptr->cur(Map_StringToPtr_Visit(derived->base.m_mapParam, key)->m_ptr));
#else
        *(int *)(p->cur(p)) = *(int *)(Map_StringToPtr_Visit(derived->base.m_mapParam, key)->m_ptr->cur(Map_StringToPtr_Visit(derived->base.m_mapParam, key)->m_ptr));
#endif
    }
    Map_StringToPtr_PushBack(derived->base.m_mapParam, key, p);
    derived->X_train->n = Map_StringToPtr_Size(derived->base.m_mapParam);
}

void BayesianOptimizer_getOptimizedParam(ParamOptimizerIF *self, Map_StringToString param)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    Map_StringToPtr iter = derived->m_best->m_mapPopParam;
    if (!iter->m_string)
        return;
    while (iter)
    {
        Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        iter = iter->m_next;
    }
}

char *BayesianOptimizer_getOptimizedTarget(ParamOptimizerIF *self)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    if(Vector_Float_Visit(derived->m_best->m_fitness, 0)->m_val){
        float_to_string(derived->base.m_strvalue,*Vector_Float_Visit(derived->m_best->m_fitness, 0)->m_val);
        return derived->base.m_strvalue;
    }
    else{
        float_to_string(derived->base.m_strvalue,-32768.0);
        return derived->base.m_strvalue;
    }
}

Algorithm BayesianOptimizer_getAlgorithm(ParamOptimizerIF *self)
{
    return BO;
}

void BayesianOptimizer_initSample(struct BayesianOptimizer *self)
{
    int var_num, i;
    Vector_Float temp;
    Vector_Vector_Float p;
    if (self->m_iRandomState == 0) {
        return;
    }
#ifdef DEBUG_BO
    PRINTF("%s:\n", __func__ );
#endif
    var_num = Map_StringToPtr_Size(Vector_Map_StringToPtr_Visit(self->m_vSampleParam, 0)->m_msp);
    temp = Vector_Float_Ctor();
    for (i = 0; i < var_num; ++i)
    {
        Vector_Float_PushBack(temp, 0.0);
    }
    p = Vector_Vector_Float_Ctor();
    for (i = 0; i < self->m_iRandomState; ++i)
    {
        Vector_Vector_Float_PushBack(p, temp);
    }
    Vector_Float_Dtor(temp);
    temp = nullptr;
    self->m_funcSampling(p, 100);
    for (i = 0; i < self->m_iRandomState; ++i)
    {
        int var = 0;
        Map_StringToPtr iter = Vector_Map_StringToPtr_Visit(self->m_vSampleParam, i)->m_msp;
        while (iter)
        {
            float p_i_var = *(Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val);
            float var_max = iter->m_ptr->dmax(iter->m_ptr);
            float var_min = iter->m_ptr->dmin(iter->m_ptr);
            float d_dest = p_i_var * (var_max - var_min) + var_min;
            //float d_dest = *(Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val) * (iter->m_ptr->dmax(iter->m_ptr) - iter->m_ptr->dmin(iter->m_ptr)) + iter->m_ptr->dmin(iter->m_ptr);
            float_to_string(self->base.m_strvalue,d_dest);
            iter->m_ptr->set(iter->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
            ++var;
            iter = iter->m_next;
        }
    }
#ifdef DEBUG_BO
    Vector_Map_StringToPtr_Print(self->m_vSampleParam);
#endif
}

void BayesianOptimizer_generatePop(struct BayesianOptimizer *self)
{
    Individual* pop = Individual_Ctor();
    int sample_size = Vector_Map_StringToPtr_Size(self->m_vSampleParam);
    if (sample_size != 0) {
        Map_StringToPtr front, iter;
#ifdef DEBUG_BO
        PRINTF("%s: m_iIterIdx=%d, %d samples left -> generatePop based on m_vSampleParam.front().\n", __func__, self->m_iIterIdx, sample_size);
#endif
        front = Vector_Map_StringToPtr_Visit(self->m_vSampleParam, 0)->m_msp;
        iter = front;
        while (iter)
        {
            Map_StringToPtr_PushBack(pop->m_mapPopParam, iter->m_string, iter->m_ptr->clone(iter->m_ptr));
            iter = iter->m_next;
        }
        Vector_Map_StringToPtr_Erase(&(self->m_vSampleParam), front);
        pop->m_fitness = Vector_Float_Ctor();
        Vector_Individual_PushBack(self->m_vPop, pop);

        if (!Vector_Map_StringToPtr_Size(self->m_vSampleParam)) {
            self->update_step = 0;
        }
    }
    else {
        int xlen = Map_StringToPtr_Size(self->base.m_mapParam);
        int sample_num = SAMPLE_NUM;
        REAL *buf = (REAL *)MALLOC(sizeof(REAL) * xlen * sample_num);
        int i = 0;
        Matrix X_samples, best_sample;
        int best_sample_index;
        Map_StringToPtr iter;
        int paramIdx;
#ifndef NORMAL_DISTRIBUTION
        int id;
        for (id = 0; id < sample_num; id++)
        {
            Map_StringToPtr iter = self->m_best->m_mapPopParam;
            while (iter)
            {
                OptimizedParamIF * param = iter->m_ptr;
                REAL y = param->dcur(param);
                REAL max = param->dmax(param);
                REAL min = param->dmin(param);
                REAL r = randomFloat(0.0, 1.0);
                int eta = 10 - self->update_step;
                REAL beta, alpha, d;
#if 0
                buf[i++] = randomFloat(0.0, 1.0) * (param->dmax(param) - param->dmin(param)) + param->dmin(param);
#else
                if (eta < 1) eta = 1;
                if (r <= 0.5)
                {
                    beta = 1.0 - (y - min) / (max - min);
                    alpha = 2 * r + (1.0 - 2.0 * r) * pow(beta, eta + 1);
                    d = pow(alpha, 1.0 / (eta + 1)) - 1;
                }
                else
                {
                    beta = 1.0 - (max - y) / (max - min);
                    alpha = 2 * (1 - r) + (2.0 * r - 1.0) * pow(beta, eta + 1);
                    d = 1.0 - pow(alpha, 1.0 / (eta + 1));
                }
                if (d > 0)
                    y = y + d * ((max - y) > 0 ? max - y : 1);
                else
                    y = y + d * ((y - min) > 0 ? y - min : 1);
                if (y > max)
                    y = max;
                if (y < min)
                    y = min;
                buf[i++] = y;
#endif
                iter = iter->m_next;
            }
        }
#else
        //TODO
//        int j = 0;
//            for (auto iter = m_best.m_mapPopParam.begin(); iter != m_best.m_mapPopParam.end(); ++iter)
//            {
//                float y = iter->second->dcur();
//                float max = iter->second->dmax();
//                float min = iter->second->dmin();
//                float eta = 1.0;
//                if (y > (max + min)/2)
//                    eta =  max - y;
//                else
//                    eta =  y - min;
//                normal_distribution<> distribution{y, eta/3};
//                for (auto id = 0; id < sample_num; id++)
//                {
//                    float r = distribution(generator);
//                    if (r > max)
//                        r = max;
//                    if (r < min)
//                        r = min;
//                    buf[id * m_best.m_mapPopParam.size() + j] = r;
//                }
//                j++;
//            }
#endif
        //Call gpr to select the best sample
        createMatrixFromData(&X_samples, buf, sample_num, xlen);
        best_sample_index = self->m_gpr->select_best(self->m_gpr, self->X_train, self->y_train, &X_samples);
        if (best_sample_index < 0 || best_sample_index >= sample_num) {
            PRINTF("%s: fail to select the best sample (best_sample_index=%d).\n", __func__, best_sample_index);
            best_sample_index = 0;
        }
        createMatrixFromData(&best_sample, &(buf[best_sample_index * xlen]), 1, xlen);

#ifdef DEBUG_BO
        PRINTF("%s: m_iIterIdx=%d, generatePop by selecting the best (with idx %d) from %d samples.\n", __func__, self->m_iIterIdx, best_sample_index, sample_num);
#endif
        //Prepare a pop based on the best sample
        iter = self->base.m_mapParam;
        paramIdx = 0;
        while(iter) {
            Map_StringToPtr_PushBack(pop->m_mapPopParam, iter->m_string, iter->m_ptr->clone(iter->m_ptr));
            float_to_string(self->base.m_strvalue,best_sample.data[paramIdx]);
            pop->m_mapPopParam->m_ptr->set(pop->m_mapPopParam->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
            iter = iter->m_next;
            ++paramIdx;
        }
        pop->m_fitness = Vector_Float_Ctor();
        Vector_Individual_PushBack(self->m_vPop, pop);

        FREE(buf);
        buf = nullptr;
    }
#ifdef DEBUG_BO
    Vector_Individual_Print(self->m_vPop);
    //Vector_Map_StringToPtr_Print(self->m_vSampleParam);
#endif
}

#ifndef KERNEL_MODULE
void BayesianOptimizer_dump_csv(ParamOptimizer *self)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    FILE *fp = self->m_csvOut;
    Map_StringToPtr iter = nullptr;
    float* fitness = nullptr;

    if (!fp || !Vector_Individual_Size(derived->m_vPop)) {
        return;
    }

    if (derived->m_iIterIdx == 0) {
        iter = Vector_Individual_Visit(derived->m_vPop, 0)->m_indi->m_mapPopParam;
        fprintf(fp, "iter,");
        while (iter)
        {
            fprintf(fp, "%s,", iter->m_string);
            iter = iter->m_next;
        }
        fprintf(fp, "fitness\n");
    }
    fprintf(fp, "%d,", derived->m_iIterIdx);
    iter = Vector_Individual_Visit(derived->m_vPop, 0)->m_indi->m_mapPopParam;
    while (iter)
    {
        fprintf(fp, "%s,", iter->m_ptr->to_string(iter->m_ptr));
        iter = iter->m_next;
    }
    fitness = Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, 0)->m_indi->m_fitness, 0)->m_val;

    fprintf(fp, "%f\n", *fitness);
}
#endif

void BayesianOptimizer_update_intern(ParamOptimizer *self)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
#ifdef DEBUG_BO
    PRINTF("%s: m_iIterIdx=%d\n", __func__, derived->m_iIterIdx);
#endif
    if (derived->m_iRandomState && Vector_Map_StringToPtr_Size(derived->m_vSampleParam) == derived->m_iRandomState) {
        // initial state
        BayesianOptimizer_initSample(derived);
    }
    else {
        Individual* vPop_front = Vector_Individual_Visit(derived->m_vPop, 0)->m_indi;
        Vector_Float vPop_front_fitness = Vector_Individual_Visit(derived->m_vPop, 0)->m_indi->m_fitness;
        //Update derived->m_vPop[0].fitness with derived->base.m_prevTarget
        float front_fitness = *(Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val);
        if (Vector_Float_Size(vPop_front_fitness) == 0) {
            Vector_Float_PushBack(vPop_front_fitness, front_fitness);
        }
        else {
            *(Vector_Float_Visit(vPop_front_fitness, 0)->m_val) = *(Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val);
        }
        //Update derived->m_best
        if (derived->m_iIterIdx == 0) {
            Individual_Assign(derived->m_best, vPop_front);
        }
        else {
            float best_fitness = *(Vector_Float_Visit(derived->m_best->m_fitness, 0)->m_val);
            if (best_fitness < front_fitness) {
                Individual_Assign(derived->m_best, vPop_front);
                derived->update_step = 1;
            }
            else {
                derived->update_step++;
            }
        }
#ifdef DEBUG_BO
        PRINTF("vPop_front = :\n");
        Individual_Print(vPop_front);
        PRINTF("Update m_best as:\n");
        Individual_Print(derived->m_best);
#endif
#ifndef KERNEL_MODULE
        BayesianOptimizer_dump_csv((ParamOptimizer *)self);
#endif
        derived->m_iIterIdx++;

        //Update derived->X_train and derived->y_train for GPR
        if (!BayesianOptimizer_update_training_data(derived, vPop_front)) {
            PRINTF("Fail to update GPR training data!\n");
        }

        Vector_Individual_Erase(&(derived->m_vPop), vPop_front);
    }
    BayesianOptimizer_generatePop(derived);
}

bool_t BayesianOptimizer_update_training_data(BayesianOptimizer *self, Individual* pop) {
    int param_num = Map_StringToPtr_Size(pop->m_mapPopParam), offset_y;
    if (self->X_train->n == param_num) {
        //Append the params of pop to X_train
        Pair_StringToPtr *iter = pop->m_mapPopParam;
        int offset_x = self->X_train->m * self->X_train->n;
        while (iter && iter->m_string && iter->m_ptr) {
            if (offset_x >= X_TRAIN_SIZE_MAX) {
                PRINTF("Error occurs when update X_train: overflow! (size=%d, offset_x=%d)\n", X_TRAIN_SIZE_MAX, offset_x);
                return false_t;
            }
            self->X_train->data[offset_x] = iter->m_ptr->dcur(iter->m_ptr);;
            iter = iter->m_next;
            ++offset_x;
        }
        self->X_train->m++;
    }
    else {
        PRINTF("Error occurs when update X_train: param_num (%d) != X_train->n (%d)\n", param_num, self->X_train->n);
        return false_t;
    }

    offset_y = self->y_train->m * self->y_train->n;
    if (offset_y < Y_TRAIN_SIZE_MAX) {
        self->y_train->data[offset_y] = *(Vector_Float_Visit(pop->m_fitness, 0)->m_val);
        self->y_train->m++;
    }
    else {
        PRINTF("Error occurs when update y_train: overflow! (size=%d, offset_y=%d)\n", Y_TRAIN_SIZE_MAX, offset_y);
        return false_t;
    }

    if (self->X_train->m != self->y_train->m) {
        PRINTF("Error occurs when update GPR training data: X_train->m(%d) != y_train->m(%d)\n", self->X_train->m, self->y_train->m);
        return false_t;
    }

    return true_t;
}

void BayesianOptimizer_update_intern_param(ParamOptimizer *self, Map_StringToString param, Vector_Float result) {
    //BayesianOptimizer *derived = (BayesianOptimizer *) self;
    //TODO: check whether this function is ever called
    PRINTF("BayesianOptimizer_update_intern_param is called!\n");
}

bool_t BayesianOptimizer_getTrial(ParamOptimizerIF *self, Map_StringToString param)
{
    int vPopSize;
    Node_Individual* vPopBack;
    Map_StringToPtr iter;
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    PRINTF("BayesianOptimizer_getTrial is called!\n");
    if (derived->m_iRandomState && Vector_Map_StringToPtr_Size(derived->m_vSampleParam) == derived->m_iRandomState) {
        // initial state
        BayesianOptimizer_initSample(derived);
    }
    BayesianOptimizer_generatePop(derived);
    vPopSize = Vector_Individual_Size(derived->m_vPop);
    vPopBack = Vector_Individual_Visit(derived->m_vPop, vPopSize-1);
    iter = vPopBack->m_indi->m_mapPopParam;
    while (iter)
    {
        Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        iter = iter->m_next;
    }
    return true_t;
}

#if FLOAT_PARAM
float BayesianOptimizer_optimize(ParamOptimizer *self, char *key, float min, float max)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    if (!Vector_Individual_Size(derived->m_vPop)) {
        PRINTF("ERROR: optimize is called when queue is empty, should not happen\n");
    }
    else {
        Node_Individual* pop_front = Vector_Individual_Visit(derived->m_vPop, 0);
        Map_StringToPtr found = Map_StringToPtr_Find(pop_front->m_indi->m_mapPopParam, key);
        if (nullptr != found) {
            float cur = *(float *)found->m_ptr->cur(found->m_ptr);
            if (cur > max) {
                cur = max;
            }
            if (cur < min) {
                cur = min;
            }
            *(float *)found->m_ptr->cur(found->m_ptr) = cur;
            return cur;
        }
    }
    return 0.0;
}
#else
int BayesianOptimizer_optimize(ParamOptimizer *self, char *key, int min, int max)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    if (!Vector_Individual_Size(derived->m_vPop)) {
        PRINTF("ERROR: optimize is called when queue is empty, should not happen\n");
    }
    else {
        Node_Individual* pop_front = Vector_Individual_Visit(derived->m_vPop, 0);
        Map_StringToPtr found = Map_StringToPtr_Find(pop_front->m_indi->m_mapPopParam, key);
        if (nullptr != found) {
            int cur = *(int *)found->m_ptr->cur(found->m_ptr);
            if (cur > max) {
                cur = max;
            }
            if (cur < min) {
                cur = min;
            }
            *(int *)found->m_ptr->cur(found->m_ptr) = cur;
            return cur;
        }
    }
    return 0;
}
#endif

bool_t BayesianOptimizer_isTrainingEnd(ParamOptimizerIF *self)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    return (derived->m_iIterIdx == derived->m_iIterNum + derived->m_iRandomState);
}

#if FLOAT_PARAM
void BayesianOptimizer_set_value(ParamOptimizer *self, char *key, float value)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    Node_Individual* pop_front = Vector_Individual_Visit(derived->m_vPop, 0);
    Map_StringToPtr found = Map_StringToPtr_Find(pop_front->m_indi->m_mapPopParam, key);
    if (NULL != found) {
        *(float *) found->m_ptr->cur(found->m_ptr) = value;
    }
}
#else
void BayesianOptimizer_set_value(ParamOptimizer *self, char *key, int value)
{
    BayesianOptimizer *derived = (BayesianOptimizer *)self;
    Node_Individual* pop_front = Vector_Individual_Visit(derived->m_vPop, 0);
    Map_StringToPtr found = Map_StringToPtr_Find(pop_front->m_indi->m_mapPopParam, key);
    if (NULL != found) {
        *(int *) found->m_ptr->cur(found->m_ptr) = value;
    }
}
#endif

BayesianOptimizer *BayesianOptimizer_Ctor(OptParam *param)
{
    BayesianOptimizer *self = (BayesianOptimizer *)MALLOC(sizeof(BayesianOptimizer));
    if (self) {
        ParamOptimizer_Ctor(&(self->base), param);
        self->base.base.update = BayesianOptimizer_update;
        //self->base.base.completeTrial = ;
        self->base.base.getTrial = BayesianOptimizer_getTrial;
        self->base.base.getAlgorithm = BayesianOptimizer_getAlgorithm;
        self->base.base.regist = BayesianOptimizer_regist;
        //self->base.base.unregist = ;
        self->base.base.getOptimizedParam = BayesianOptimizer_getOptimizedParam;
        //self->base.base.getOptimizedParams = ;
        self->base.base.getOptimizedTarget = BayesianOptimizer_getOptimizedTarget;
        //self->base.base.getOptimizedTargets = ;
        //self->base.base.getCurrentParam = ;
        //self->base.base.calibrateParam = ;
        //static struct ParamOptimizerIF *getParamOptimizer(OptParam<TA> &param);
        self->base.base.isTrainingEnd = BayesianOptimizer_isTrainingEnd;
        //self->base.base.initLogging = ;
        //self->base.base.setPCAWindow = ;

        //self->base.pca_analysis = ;
        //self->base.append_sample = ;
        self->base.update_intern = BayesianOptimizer_update_intern;
        self->base.update_intern_param = BayesianOptimizer_update_intern_param;
#ifndef KERNEL_MODULE
        self->base.dump_csv = BayesianOptimizer_dump_csv;
#endif
        //self->base.isInHistory = ;
        //self->base.isSame = ;
        self->base.optimize = BayesianOptimizer_optimize;
        self->base.set_value = BayesianOptimizer_set_value;

        if (param->algorithm == BO) {
            BOOptParam *p = (BOOptParam *)param;
            self->m_iIterNum = p->iter_num;
            self->m_iRandomState = p->random_state;
        }

        self->initSample = BayesianOptimizer_initSample;
        self->generatePop = BayesianOptimizer_generatePop;
        self->m_iIterIdx = 0;
        self->m_vSampleParam = Vector_Map_StringToPtr_Ctor();
        Vector_Map_StringToPtr_Resize(self->m_vSampleParam, self->m_iRandomState);
        self->m_best = Individual_Ctor();
        self->m_vPop = Vector_Individual_Ctor();
        self->m_funcSampling = Sampling_LatinHypercubeSampling;
        self->update_step = 0;

        self->X_train = newMatrix(TRAIN_NUM_MAX, PARAM_NUM_MAX);
        self->X_train->m = 0;
        self->X_train->n = 0; //Will be updated when regist params
        self->y_train = newMatrix(TRAIN_NUM_MAX, TARGET_NUM_MAX);
        self->y_train->m = 0;
        self->y_train->n = 1; //Currently only one target is supported
        self->m_kernel = &(RBFKernel_Ctor(1.0)->base);
        self->m_acquiAlgo = &(GP_MI_Ctor(log(2.0/pow(10.0, -9.0)))->base);
        self->m_gpr = GPR_Ctor(self->m_kernel, self->m_acquiAlgo, 1e-10);
    }
    return self;
}

void BayesianOptimizer_Dtor(BayesianOptimizer *self)
{
    if (self) {
        Vector_Map_StringToPtr_Dtor(self->m_vSampleParam);
        self->m_vSampleParam = nullptr;
        Individual_Dtor(self->m_best);
        self->m_best = nullptr;
        Vector_Individual_Dtor(self->m_vPop);
        self->m_vPop = nullptr;
        GPR_Dtor(self->m_gpr);
        self->m_gpr = nullptr;
        RBFKernel_Dtor((RBFKernel*)(self->m_kernel));
        self->m_kernel = nullptr;
        ParamOptimizer_Dtor(&(self->base));
        FREE(self);
    }
}
