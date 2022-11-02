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

#include "DEOptimizer.h"

void DEOptimizer_update(ParamOptimizerIF *self)
{
    ParamOptimizer_update(self);
}

#if FLOAT_PARAM
void DEOptimizer_regist(ParamOptimizerIF *self, char *key, float min, float max, int (*update)(char *, float))
#else
void DEOptimizer_regist(ParamOptimizerIF *self, char *key, int min, int max, int (*update)(char *, int))
#endif
{
    DEOptimizer *derived = (DEOptimizer *)self;
    OptimizedParamIF *p = (OptimizedParamIF *)OptimizedParam_Ctor(key, min, max, min, update, derived->base.optimize, derived->base.set_value);
    if (Map_StringToPtr_Find(derived->base.m_mapParam, key) == nullptr)
    {
        int i;
        for (i = 0; i < derived->m_iPopSize; ++i)
        {
            Map_StringToPtr_PushBack(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_mapPopParam, key, p->clone(p));
            Map_StringToPtr_PushBack(Vector_Individual_Visit(derived->m_vMutationPop, i)->m_indi->m_mapPopParam, key, p->clone(p));
            Map_StringToPtr_PushBack(Vector_Individual_Visit(derived->m_vCrossOverPop, i)->m_indi->m_mapPopParam, key, p->clone(p));
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
}

void DEOptimizer_getOptimizedParam(ParamOptimizerIF *self, Map_StringToString param)
{
    DEOptimizer *derived = (DEOptimizer *)self;
    Map_StringToPtr iter;
    derived->base.pca_analysis(&(derived->base), derived->m_mbestParam);
    iter = derived->m_mbestParam;
    while (iter && iter->m_string && iter->m_ptr)
    {
        Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        iter = iter->m_next;
    }
}

char *DEOptimizer_getOptimizedTarget(ParamOptimizerIF *self)
{
    DEOptimizer *derived = (DEOptimizer *)self;
    float_to_string(derived->base.m_strvalue,derived->m_bestFitness);
    return derived->base.m_strvalue;
}

Algorithm DEOptimizer_getAlgorithm(ParamOptimizerIF *self)
{
    return DE;
}

void lambda_scanPop(DEOptimizer *self, Vector_Individual PopPtr)
{
    int i;
    self->m_bestFitness = 0;
    self->m_averageFitness = 0;
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        if (i == 0)
        {
            self->m_bestFitness = self->m_worstFitness = *Vector_Float_Visit(Vector_Individual_Visit(PopPtr, i)->m_indi->m_fitness, 0)->m_val;
        }
        if (*Vector_Float_Visit(Vector_Individual_Visit(PopPtr, i)->m_indi->m_fitness, 0)->m_val > self->m_bestFitness)
        {
            Map_StringToPtr_Assign(self->m_mbestParam, Vector_Individual_Visit(PopPtr, i)->m_indi->m_mapPopParam);
            self->m_bestFitness = *Vector_Float_Visit(Vector_Individual_Visit(PopPtr, i)->m_indi->m_fitness, 0)->m_val;
        }
        if (*Vector_Float_Visit(Vector_Individual_Visit(PopPtr, i)->m_indi->m_fitness, 0)->m_val < self->m_worstFitness)
        {
            self->m_worstFitness = *Vector_Float_Visit(Vector_Individual_Visit(PopPtr, i)->m_indi->m_fitness, 0)->m_val;
        }
        self->m_averageFitness += *Vector_Float_Visit(Vector_Individual_Visit(PopPtr, i)->m_indi->m_fitness, 0)->m_val;
    }
    self->m_averageFitness = self->m_averageFitness / self->m_iPopSize;
}

void DEOptimizer_update_intern(ParamOptimizer *self)
{
    DEOptimizer *derived = (DEOptimizer *)self;
    if (derived->m_iGenIdx == 0 && Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
    {
        // initial state
        derived->initgroup(derived);
    }
    else
    {
        void (*scanPop)(DEOptimizer *, Vector_Individual) = lambda_scanPop;
        if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
        {   
            int update_idx = derived->m_updateindex->m_pair->m_val;
            if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
            {
                *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_fitness, 0)->m_val = *Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val;
            }
            else
            {   
                *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_fitness, 0)->m_val = *Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val;
                if (*Vector_Float_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_fitness, 0)->m_val > *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_fitness, 0)->m_val)
                {
                    Individual_Assign(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi, Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi);
                }
            }
            derived->m_updateindex = Vector_Pair_StringToInt_Erase(derived->m_updateindex, derived->m_updateindex);

            while (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
            {   
                if (derived->over_flag)
                {   
                    derived->m_bExit = true_t;   
                    Vector_Pair_StringToInt_PushBack_param(derived->m_updateindex,"CrossOver", 0);                
                    break;
                }                
                if (derived->m_iGenIdx != derived->m_iGenNum)
                {
#ifndef KERNEL_MODULE
                    self->dump_csv(self);
#endif

                    scanPop(derived, derived->m_vPop);
                    derived->mutation(derived);
                    derived->crossover(derived);
                    ++(derived->m_iGenIdx);
                }
                else
                {
                    scanPop(derived, derived->m_vPop);
#ifndef KERNEL_MODULE
                    self->dump_csv(self);
#endif
                    break;
                }
            }
        }
    }
    if (!derived->m_bExit)
        derived->m_bExit = (derived->m_iGenIdx == derived->m_iGenNum - 1) && (Vector_Pair_StringToInt_Size(derived->m_updateindex) < 2);
}

void DEOptimizer_update_intern_param(ParamOptimizer *self, Map_StringToString param, Vector_Float result)
{
    DEOptimizer *derived = (DEOptimizer *)self;
    Node_Pair_StringToInt *idx_iter = derived->m_waitingindex;
    bool_t found = false_t;
    Individual *p;
    while (idx_iter != nullptr)
    {
        if (strcmp(idx_iter->m_pair->m_string, "Initial"))
        {
            p = Vector_Individual_Visit(derived->m_vPop, idx_iter->m_pair->m_val)->m_indi;
        }
        else
        {
            p = Vector_Individual_Visit(derived->m_vCrossOverPop, idx_iter->m_pair->m_val)->m_indi;
        }
        if (Individual_IsSame(p, param))
        {
            Vector_Float_Assign(p->m_fitness, result);
            found = true_t;
            break;
        }
        idx_iter = idx_iter->m_next;
    }
    if (!found)
    {
        PRINTF("The Trail is not is the waiting list\n");
        return;
    }
    if (strcmp(idx_iter->m_pair->m_string, "CrossOver") == 0 && *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, idx_iter->m_pair->m_val)->m_indi->m_fitness, 0)->m_val > *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, idx_iter->m_pair->m_val)->m_indi->m_fitness, 0)->m_val)
    {
        Individual_Assign(Vector_Individual_Visit(derived->m_vPop, idx_iter->m_pair->m_val)->m_indi, Vector_Individual_Visit(derived->m_vCrossOverPop, idx_iter->m_pair->m_val)->m_indi);
    }
    derived->m_waitingindex = Vector_Pair_StringToInt_Erase(derived->m_waitingindex, idx_iter);
    derived->m_bExit = (derived->m_iGenIdx >= derived->m_iGenNum) && (Vector_Pair_StringToInt_Size(derived->m_waitingindex) == 0);
}

bool_t DEOptimizer_getTrial(ParamOptimizerIF *self, Map_StringToString param)
{
    int update_idx;
    Map_StringToPtr iter;
    DEOptimizer *derived = (DEOptimizer *)self;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0 && Vector_Pair_StringToInt_Size(derived->m_waitingindex) != 0)
    {
        return false_t;
    }
    if (derived->m_initPop)
    {
        // initial state
        derived->initgroup(derived);
        derived->m_initPop = false_t;
    }
    else
    {
        void (*scanPop)(DEOptimizer *, Vector_Individual) = lambda_scanPop;
        while (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
        {
            if (derived->m_iGenIdx != derived->m_iGenNum)
            {
#ifndef KERNEL_MODULE
                derived->base.dump_csv((ParamOptimizer *)derived);
#endif
                scanPop(derived, derived->m_vPop);
                derived->mutation(derived);
                derived->crossover(derived);
                ++(derived->m_iGenIdx);
            }
            else
            {
                scanPop(derived, derived->m_vPop);
#ifndef KERNEL_MODULE
                derived->base.dump_csv((ParamOptimizer *)derived);
#endif
                break;
            }
        }
    }
    update_idx = derived->m_updateindex->m_pair->m_val;
    iter = nullptr;
    if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial"))
    {
        for (iter = Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        }
    }
    else
    {
        for (iter = Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        }
    }
    Vector_Pair_StringToInt_PushBack(derived->m_waitingindex, derived->m_updateindex->m_pair);
    derived->m_updateindex = Vector_Pair_StringToInt_Erase(derived->m_updateindex, derived->m_updateindex);
    return true_t;
}

#if FLOAT_PARAM
float DEOptimizer_optimize(ParamOptimizer *self, char *key, float min, float max)
#else

int DEOptimizer_optimize(ParamOptimizer *self, char *key, int min, int max)
#endif
{
    DEOptimizer *derived = (DEOptimizer *)self;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
    {
        int update_idx = derived->m_updateindex->m_pair->m_val;
        if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
        {
#if FLOAT_PARAM
            float cur = *(float *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr));
#else
            int cur = *(int *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr));
#endif
            if (cur > max)
            {
                cur = max;
            }
            if (cur < min)
            {
                cur = min;
            }
#if FLOAT_PARAM
            *(float *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr)) = cur;
#else
            *(int *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr)) = cur;
#endif
            return cur;
        }
        else
        {
#if FLOAT_PARAM
            float cur = *(float *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr));
#else
            int cur = *(int *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr));
#endif
            if (cur > max)
            {
                cur = max;
            }
            if (cur < min)
            {
                cur = min;
            }
#if FLOAT_PARAM
            *(float *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr)) = cur;
#else
            *(int *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr)) = cur;
#endif
            return cur;
        }
    }
    else
    {
        PRINTF("optimize is called when m_updateindex queue is empty, should not happen\n");
        return -32768;
    }
}

#if FLOAT_PARAM
void DEOptimizer_set_value(ParamOptimizer *self, char *key, float value)
#else
void DEOptimizer_set_value(ParamOptimizer *self, char *key, int value)
#endif
{
    DEOptimizer *derived = (DEOptimizer *)self;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
    {
        int update_idx = derived->m_updateindex->m_pair->m_val;
        if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
        {
            Map_StringToPtr iter = nullptr;
            iter = Map_StringToPtr_Find(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key);
            if (iter)
            {
#if FLOAT_PARAM
                *(float *)(iter->m_ptr->cur(iter->m_ptr)) = value;
#else
                *(int *)(iter->m_ptr->cur(iter->m_ptr)) = value;
#endif
            }
        }
        else
        {
            Map_StringToPtr iter = nullptr;
            iter = Map_StringToPtr_Find(Vector_Individual_Visit(derived->m_vCrossOverPop, update_idx)->m_indi->m_mapPopParam, key);
            if (iter)
            {
#if FLOAT_PARAM
                *(float *)(iter->m_ptr->cur(iter->m_ptr)) = value;
#else
                *(int *)(iter->m_ptr->cur(iter->m_ptr)) = value;
#endif
            }
        }
    }
    else
    {
        PRINTF("set value is called when m_updateindex queue is empty, should not happen\n");
    }
}

void DEOptimizer_crossover(DEOptimizer *self)
{
    float rTemp, cr;
    Map_StringToPtr iter = nullptr, iterm = nullptr, iterd = nullptr;
    int i;
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        iter = Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_mapPopParam;
        iterm = Vector_Individual_Visit(self->m_vMutationPop, i)->m_indi->m_mapPopParam;
        iterd = Vector_Individual_Visit(self->m_vCrossOverPop, i)->m_indi->m_mapPopParam;
        cr = self->m_dcr_l;
        if (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_fitness, 0)->m_val < self->m_averageFitness)
        {
            cr = (self->m_bestFitness - *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_fitness, 0)->m_val) / (self->m_bestFitness - self->m_worstFitness);
        }
        do
        {
            rTemp = randomFloat(0.0, 1.0);
            if (rTemp <= cr)
            {
                iterd->m_ptr = iterm->m_ptr->clone(iterm->m_ptr);
            }
            else
            {
                iterd->m_ptr = iter->m_ptr->clone(iter->m_ptr);
            }
            iterd = iterd->m_next;
            iterm = iterm->m_next;
            iter = iter->m_next;
        } while (iter);
        if (!self->base.isInHistory((ParamOptimizer *)self, Vector_Individual_Visit(self->m_vCrossOverPop, i)->m_indi->m_mapPopParam))
            {
                Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "CrossOver", i);
            }
        else
            {
            if (self->retry == 0) {
                self->over_flag = true;
            } else {
                self->retry -= 1;
            }
            }


    }
    if (!self->over_flag){
        self->retry = 15; 
    }
}

void DEOptimizer_mutation(DEOptimizer *self)
{   
    int r1, r2, r3;
    //OptimizedParamIF *temp;
    Map_StringToPtr iterb = nullptr;
    Map_StringToPtr iterm = nullptr;
    Map_StringToPtr iterw = nullptr;
    Map_StringToPtr iterd = nullptr;
    Vector_Int sequence = Vector_Int_Ctor();
    Vector_Int result = Vector_Int_Ctor();
    int i;
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        Vector_Int_PushBack(sequence, i);
    }
    for (i = 0; i < 3; ++i)
    {
        Vector_Int vi;
        Vector_Int_RandomShuffle(sequence);
        vi = sequence;
        while (vi)
        {
            Vector_Int_PushBack(result, *vi->m_val);
            vi = vi->m_next;
        }
    }
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        int best, worst, middle;
        float f;

        r1 = *Vector_Int_Visit(result, 3 * i)->m_val;
        r2 = *Vector_Int_Visit(result, 3 * i + 1)->m_val;
        r3 = *Vector_Int_Visit(result, 3 * i + 2)->m_val;
        if (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r1)->m_indi->m_fitness, 0)->m_val > *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r2)->m_indi->m_fitness, 0)->m_val)
        {
            if (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r1)->m_indi->m_fitness, 0)->m_val > *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r3)->m_indi->m_fitness, 0)->m_val)
            {
                best = r1;
                if (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r2)->m_indi->m_fitness, 0)->m_val > *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r3)->m_indi->m_fitness, 0)->m_val)
                {
                    middle = r2;
                    worst = r3;
                }
                else
                {
                    worst = r2;
                    middle = r3;
                }
            }
            else
            {
                middle = r1;
                worst = r2;
                best = r3;
            }
        }
        else
        {
            if (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r1)->m_indi->m_fitness, 0)->m_val < *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r3)->m_indi->m_fitness, 0)->m_val)
            {
                worst = r1;
                if (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r2)->m_indi->m_fitness, 0)->m_val > *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, r3)->m_indi->m_fitness, 0)->m_val)
                {
                    best = r2;
                    middle = r3;
                }
                else
                {
                    middle = r2;
                    best = r3;
                }
            }
            else
            {
                middle = r1;
                worst = r3;
                best = r2;
            }
        }
        iterd = Vector_Individual_Visit(self->m_vMutationPop, i)->m_indi->m_mapPopParam;
        iterm = Vector_Individual_Visit(self->m_vPop, middle)->m_indi->m_mapPopParam;
        iterw = Vector_Individual_Visit(self->m_vPop, worst)->m_indi->m_mapPopParam;
        iterb = Vector_Individual_Visit(self->m_vPop, best)->m_indi->m_mapPopParam;
        f = self->m_df_l;
        if (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, best)->m_indi->m_fitness, 0)->m_val != *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, worst)->m_indi->m_fitness, 0)->m_val)
        {
            f = self->m_df_l + (self->m_df_h - self->m_df_l) *
                                   (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, best)->m_indi->m_fitness, 0)->m_val - *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, middle)->m_indi->m_fitness, 0)->m_val) /
                                   (*Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, best)->m_indi->m_fitness, 0)->m_val - *Vector_Float_Visit(Vector_Individual_Visit(self->m_vPop, worst)->m_indi->m_fitness, 0)->m_val);
        }
        do
        {
            float d_b = iterb->m_ptr->dcur(iterb->m_ptr);
            float d_m = iterm->m_ptr->dcur(iterm->m_ptr);
            float d_w = iterw->m_ptr->dcur(iterw->m_ptr);
            float d_dest = d_b + f * (d_m - d_w);
            if (d_dest > iterd->m_ptr->dmax(iterd->m_ptr))
            {
                d_dest = randomFloat(iterd->m_ptr->dmin(iterd->m_ptr), iterd->m_ptr->dmax(iterd->m_ptr));
            }
            if (d_dest < iterd->m_ptr->dmin(iterd->m_ptr))
            {
                d_dest = randomFloat(iterd->m_ptr->dmin(iterd->m_ptr), iterd->m_ptr->dmax(iterd->m_ptr));
            }
            float_to_string(self->base.m_strvalue,d_dest);
            iterd->m_ptr->set(iterd->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
            iterd = iterd->m_next;
            iterm = iterm->m_next;
            iterb = iterb->m_next;
            iterw = iterw->m_next;
        } while (iterd);
    }
}

void DEOptimizer_initgroup(DEOptimizer *self)
{
    int var_num, i;
    Vector_Float temp;
    Vector_Vector_Float p;
    if (self->m_iPopSize == 0)
    {
        return;
    }
    var_num = Map_StringToPtr_Size(Vector_Individual_Visit(self->m_vPop, 0)->m_indi->m_mapPopParam);
    temp = Vector_Float_Ctor();
    for (i = 0; i < var_num; ++i)
    {
        Vector_Float_PushBack(temp, 0.0);
    }
    p = Vector_Vector_Float_Ctor();
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        Vector_Vector_Float_PushBack(p, temp);
    }
    Vector_Float_Dtor(temp);
    temp = nullptr;
    self->m_funcSampling(p, 100);

    for (i = 0; i < self->m_iPopSize; ++i)
    {
        int var = 0;
        Map_StringToPtr iter;
        for (iter = Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            float d_dest = *Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val * (iter->m_ptr->dmax(iter->m_ptr) - iter->m_ptr->dmin(iter->m_ptr)) + iter->m_ptr->dmin(iter->m_ptr);
            float_to_string(self->base.m_strvalue,d_dest);
            iter->m_ptr->set(iter->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
            ++var;
        }
        Vector_Float_Resize(Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_fitness, 1);
        Vector_Float_Resize(Vector_Individual_Visit(self->m_vMutationPop, i)->m_indi->m_fitness, 1);
        Vector_Float_Resize(Vector_Individual_Visit(self->m_vCrossOverPop, i)->m_indi->m_fitness, 1);
        Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Initial", i);
    }
}

bool_t DEOptimizer_isTrainingEnd(ParamOptimizerIF *self)
{
    DEOptimizer *derived = (DEOptimizer *)self;
    if (derived->m_iGenNum == 1)
    {
        return derived->m_iGenIdx == derived->m_iGenNum;
    }
    else
    {
        return derived->m_bExit;
    }
}

#ifndef KERNEL_MODULE
void DEOptimizer_dump_csv(ParamOptimizer *self)
{
    DEOptimizer *derived = (DEOptimizer *)self;
    //float sum = 0, average = 0;
    FILE *fp = self->m_csvOut;
    if (!fp || !derived->m_iPopSize)
    {
        return;
    }

    if (!derived->m_iGenIdx)
    {
        fprintf(fp, "iter");
        Map_StringToPtr iter = nullptr;
        for (iter = Vector_Individual_Visit(derived->m_vPop, 0)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            fprintf(fp, "%s,", iter->m_string);
        }
        fprintf(fp, "fitness\n");
    }
    int i = 0;
    for (i = 0; i < derived->m_iPopSize; ++i)
    {
        int size = Map_StringToPtr_Size(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_mapPopParam);
        int j = 0;
        Map_StringToPtr iter = nullptr;
        for (iter = Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            if (j == 0)
            {
                fprintf(fp, "%d,", derived->m_iGenIdx);
            }
            fprintf(fp, "%s,", iter->m_ptr->to_string(iter->m_ptr));
            if (j == size - 1)
            {
                fprintf(fp, "%f\n", *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_fitness, 0)->m_val);
            }
            ++j;
        }
    }
}
#endif

DEOptimizer *DEOptimizer_Ctor(OptParam *param)
{
    DEOptimizer *self = (DEOptimizer *)MALLOC(sizeof(DEOptimizer));
    if (self)
    {
        ParamOptimizer_Ctor(&(self->base), param);
        self->base.base.update = DEOptimizer_update;
        //self->base.base.completeTrial = ;
        self->base.base.getTrial = DEOptimizer_getTrial;
        self->base.base.getAlgorithm = DEOptimizer_getAlgorithm;
        self->base.base.regist = DEOptimizer_regist;
        //self->base.base.unregist = ;
        self->base.base.getOptimizedParam = DEOptimizer_getOptimizedParam;
        //self->base.base.getOptimizedParams = ;
        self->base.base.getOptimizedTarget = DEOptimizer_getOptimizedTarget;
        //self->base.base.getOptimizedTargets = ;
        //self->base.base.getCurrentParam = ;
        //self->base.base.calibrateParam = ;
        self->base.base.isTrainingEnd = DEOptimizer_isTrainingEnd;
        //self->base.base.initLogging = ;
        //self->base.base.setPCAWindow = ;

        //self->base.pca_analysis = ;
        //self->base.append_sample = ;
        self->base.update_intern = DEOptimizer_update_intern;
        self->base.update_intern_param = DEOptimizer_update_intern_param;
#ifndef KERNEL_MODULE
        self->base.dump_csv = DEOptimizer_dump_csv;
#endif
        //self->base.isInHistory = ;
        //self->base.isSame = ;
        self->base.optimize = DEOptimizer_optimize;
        self->base.set_value = DEOptimizer_set_value;

        self->crossover = DEOptimizer_crossover;
        self->mutation = DEOptimizer_mutation;
        self->initgroup = DEOptimizer_initgroup;
        self->m_updateindex = Vector_Pair_StringToInt_Ctor();
        self->m_waitingindex = Vector_Pair_StringToInt_Ctor();
        self->m_mbestParam = Map_StringToPtr_Ctor();
        self->m_initPop = true_t;
        self->m_bExit = false_t;
        self->m_bestFitness = 0;
        self->m_worstFitness = 0;
        self->m_averageFitness = 0;
        self->m_funcSampling = Sampling_LatinHypercubeSampling;
        self->m_iGenNum = 50;
        self->m_iPopSize = 100;
        self->m_iGenIdx = 0;
        self->m_dcr_h = 0.6;
        self->m_dcr_l = 0.1;
        self->m_df_h = 0.9;
        self->m_df_l = 0.1;
        self->retry = 15;
        self->over_flag = false;
        if (param->algorithm == DE)
        {
            DEOptParam *p = (DEOptParam *)param;
            self->m_iGenNum = p->gen_num;
            self->m_iPopSize = p->pop_size;
        }
        self->m_vPop = Vector_Individual_Ctor();
        Vector_Individual_Resize(self->m_vPop, self->m_iPopSize);
        self->m_vMutationPop = Vector_Individual_Ctor();
        Vector_Individual_Resize(self->m_vMutationPop, self->m_iPopSize);
        self->m_vCrossOverPop = Vector_Individual_Ctor();
        Vector_Individual_Resize(self->m_vCrossOverPop, self->m_iPopSize);
    }
    return self;
}
void DEOptimizer_Dtor(DEOptimizer *self)
{
    Vector_Individual_Dtor(self->m_vPop);
    self->m_vPop = nullptr;
    Vector_Individual_Dtor(self->m_vMutationPop);
    self->m_vMutationPop = nullptr;
    Vector_Individual_Dtor(self->m_vCrossOverPop);
    self->m_vCrossOverPop = nullptr;
    ParamOptimizer_Dtor(&(self->base));
    FREE(self);
}
