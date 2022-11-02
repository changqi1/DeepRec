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

#include "MOEADOptimizer.h"

#define RANGE_CLIP(val, max, min) \
    if (val > max)                \
        val = max;                \
    if (val < min)                \
        val = min;

#define UNIFORM_WEIGHT

#define SWAP(A, B)  \
    float temp = A; \
    A = B;          \
    B = temp;

void MOEADOptimizer_update(ParamOptimizerIF *self)
{
    ParamOptimizer_update(self);
}
#if FLOAT_PARAM
void MOEADOptimizer_regist(ParamOptimizerIF *self, char *key, float min, float max, int (*update)(char *, float))
#else
void MOEADOptimizer_regist(ParamOptimizerIF *self, char *key, int min, int max, int (*update)(char *, int))
#endif
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    OptimizedParamIF *p = (OptimizedParamIF *)OptimizedParam_Ctor(key, min, max, min, update, derived->base.optimize, derived->base.set_value);
    if (Map_StringToPtr_Find(derived->base.m_mapParam, key) == nullptr)
    {
        int i;
        for (i = 0; i < derived->m_iPopSize; ++i)
        {
            Map_StringToPtr_PushBack(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_mapPopParam, key, p->clone(p));
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

void MOEADOptimizer_getOptimizedParams(ParamOptimizerIF *self, Map_VectorToMap param)
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    int i;
    for (i = 0; i < Vector_Individual_Size(derived->m_vPF); ++i)
    {
        Vector_String target = Vector_String_Ctor();
        Map_StringToString p = Map_StringToString_Ctor();
        Vector_Float t = nullptr;
        Map_StringToPtr iter;
        for (t = Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_fitness; t != nullptr; t = t->m_next)
        {
            float_to_string(derived->base.m_strvalue,*t->m_val);
            Vector_String_PushBack(target,derived->base.m_strvalue);
        }
        iter = nullptr;
        for (iter = Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            Map_StringToString_PushBack(p, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        }
    }
}

Vector_Vector_String MOEADOptimizer_getOptimizedTargets(ParamOptimizerIF *self)
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    Vector_Vector_String p = Vector_Vector_String_Ctor();
    int i;
    for (i = 0; i < Vector_Individual_Size(derived->m_vPF); ++i)
    {
        Vector_String target = Vector_String_Ctor();
        Vector_Float t = nullptr;
        for (t = Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_fitness; t != nullptr; t = t->m_next)
        {
            float_to_string(derived->base.m_strvalue,*t->m_val);
            Vector_String_PushBack(target, derived->base.m_strvalue);
        }
        Vector_Vector_String_PushBack(p, target);
    }
    return p;
}

Algorithm MOEADOptimizer_getAlgorithm(ParamOptimizerIF *self)
{
    return MOEAD;
}

void MOEADOptimizer_generateUpdatePop(MOEADOptimizer *self, int index)
{
    int pop1 = randomInt(0, self->m_iNeighNum - 1);
    int pop2 = randomInt(0, self->m_iNeighNum);
    if (pop1 == pop2)
    {
        pop2 = pop1 + 1;
    }
    RANGE_CLIP(pop1, self->m_iNeighNum - 1, 0);
    RANGE_CLIP(pop2, self->m_iNeighNum - 1, 0);
    pop1 = *Vector_Int_Visit(Vector_Vector_Int_Visit(self->m_vNeighbor, index)->m_vi, pop1)->m_val;
    pop2 = *Vector_Int_Visit(Vector_Vector_Int_Visit(self->m_vNeighbor, index)->m_vi, pop2)->m_val;
    Individual_Assign(Vector_Individual_Visit(self->m_vUpdatePop, 0)->m_indi, Vector_Individual_Visit(self->m_vPop, pop1)->m_indi);
    Individual_Assign(Vector_Individual_Visit(self->m_vUpdatePop, 1)->m_indi, Vector_Individual_Visit(self->m_vPop, pop2)->m_indi);
    self->m_funcCrossover(Vector_Individual_Visit(self->m_vUpdatePop, 0)->m_indi, Vector_Individual_Visit(self->m_vUpdatePop, 1)->m_indi, self->m_dcr);
    self->m_funcMutate(Vector_Individual_Visit(self->m_vUpdatePop, 0)->m_indi, self->m_dmutp, self->m_dmut);
    self->m_funcMutate(Vector_Individual_Visit(self->m_vUpdatePop, 1)->m_indi, self->m_dmutp, self->m_dmut);
    Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Update", 0);
    Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Update", 1);
}

void lambda_scanPop_m(MOEADOptimizer *self, Vector_Individual p)
{
    int idx;
    for (idx = 0; idx < self->m_iObjNum; ++idx)
    {
        int i;
        for (i = 0; i < self->m_iPopSize; ++i)
        {
            if (i == 0)
            {
                *Vector_Float_Visit(self->m_bestFitness, idx)->m_val = *Vector_Float_Visit(Vector_Individual_Visit(p, i)->m_indi->m_fitness, idx)->m_val;
            }
            if (*Vector_Float_Visit(Vector_Individual_Visit(p, i)->m_indi->m_fitness, idx)->m_val < *Vector_Float_Visit(self->m_bestFitness, idx)->m_val)
            {
                *Vector_Float_Visit(self->m_bestFitness, idx)->m_val = *Vector_Float_Visit(Vector_Individual_Visit(p, i)->m_indi->m_fitness, idx)->m_val;
            }
        }
    }
}

void lambda_updateBestFitness_m(MOEADOptimizer *self, Individual *p)
{
    int idx;
    for (idx = 0; idx < self->m_iObjNum; ++idx)
    {
        if (*Vector_Float_Visit(p->m_fitness, idx)->m_val < *Vector_Float_Visit(self->m_bestFitness, idx)->m_val)
        {
            *Vector_Float_Visit(self->m_bestFitness, idx)->m_val = *Vector_Float_Visit(p->m_fitness, idx)->m_val;
        }
    }
}

void MOEADOptimizer_update_intern(ParamOptimizer *self)
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    if (derived->m_iGenIdx == 0 && Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
    {
        derived->initgroup(derived);
    }
    else
    {
        void (*scanPop)(MOEADOptimizer * self, Vector_Individual p) = lambda_scanPop_m;
        void (*updateBestFitness)(MOEADOptimizer * self, Individual * p) = lambda_updateBestFitness_m;
        if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
        {
            int update_idx = derived->m_updateindex->m_pair->m_val;
            if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
            {
                Vector_Float_Assign(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_fitness, self->m_prevTarget);
            }
            else
            {
                Vector_Float_Assign(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_fitness, self->m_prevTarget);
            }
            derived->m_updateindex = Vector_Pair_StringToInt_Erase(derived->m_updateindex, derived->m_updateindex);
            if (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
            {
                if (!(derived->m_iGenIdx))
                {
                    derived->initParetoFront(derived);
#ifndef KERNEL_MODULE
                    self->dump_csv(self);
#endif
                    scanPop(derived, derived->m_vPop);
                    ++(derived->m_iGenIdx);
                    derived->generateUpdatePop(derived, derived->m_iPopIdx);
                    ++(derived->m_iPopIdx);
                }
                else
                {
                    Individual *p = Individual_Ctor();
                    if (derived->isDominates(derived, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi))
                    {
                        Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi);
                    }
                    else if (derived->isDominates(derived, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi))
                    {
                        Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi);
                    }
                    else if (randomFloat(0.0, 1.0) < 0.5)
                    {
                        Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi);
                    }
                    else
                    {
                        Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi);
                    }
                    updateBestFitness(derived, p);
                    if (derived->m_iObjNum == 2)
                    {
                        derived->update_neighbor_tchebi(derived, derived->m_iUpdatePopIdx, p);
                    }
                    else
                    {
                        derived->update_neighbor_pbi(derived, derived->m_iUpdatePopIdx, p);
                    }
                    derived->updateParetoFront(derived, p);
#ifndef KERNEL_MODULE
                    self->dump_csv(self);
#endif
                    derived->generateUpdatePop(derived, derived->m_iPopSize);
                    ++(derived->m_iPopIdx);
                    ++(derived->m_iUpdatePopIdx);
                }
                if (derived->m_iPopIdx == derived->m_iPopSize)
                {
                    derived->m_iPopIdx = 0;
                    ++(derived->m_iGenIdx);
                }
                if (derived->m_iUpdatePopIdx == derived->m_iPopSize)
                {
                    derived->m_iUpdatePopIdx = 0;
                }
            }
        }
    }
    derived->m_bExit = derived->m_iGenIdx >= derived->m_iGenNum;
}

void MOEADOptimizer_update_intern_param(ParamOptimizer *self, Map_StringToString param, Vector_Float result)
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    Vector_Pair_StringToInt idx_iter = derived->m_waitingindex;
    bool_t found = false_t;
    Individual *p = nullptr;
    while (idx_iter != nullptr)
    {
        if (strcmp(idx_iter->m_pair->m_string, "Initial") == 0)
        {
            p = Vector_Individual_Visit(derived->m_vPop, idx_iter->m_pair->m_val)->m_indi;
        }
        else
        {
            p = Vector_Individual_Visit(derived->m_vUpdatePop, idx_iter->m_pair->m_val)->m_indi;
        }
        if (Individual_IsSame(p, param))
        {
            Vector_Float_Assign(p->m_fitness, result);
            found = true_t;
            derived->m_waitingindex = Vector_Pair_StringToInt_Erase(derived->m_waitingindex, idx_iter);
            break;
        }
        idx_iter = idx_iter->m_next;
    }
    if (!found)
    {
        PRINTF("The Trail is not is the waiting list\n");
        return;
    }
    derived->m_bExit = (derived->m_iGenIdx >= derived->m_iGenNum) && (Vector_Pair_StringToInt_Size(derived->m_waitingindex) == 0);
}

bool_t MOEADOptimizer_getTrial(ParamOptimizerIF *self, Map_StringToString param)
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    int update_idx;
    Map_StringToPtr iter;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0 && Vector_Pair_StringToInt_Size(derived->m_waitingindex))
    {
        return false_t;
    }
    if (derived->m_initPop)
    {
        derived->initgroup(derived);
        derived->m_initPop = false_t;
    }
    else
    {
        void (*scanPop)(MOEADOptimizer * self, Vector_Individual p) = lambda_scanPop_m;
        void (*updateBestFitness)(MOEADOptimizer * self, Individual * p) = lambda_updateBestFitness_m;
        if (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
        {
            if (!derived->m_iGenIdx)
            {
                derived->initParetoFront(derived);
#ifndef KERNEL_MODULE
                derived->base.dump_csv((ParamOptimizer *)derived);
#endif
                scanPop(derived, derived->m_vPop);
                ++(derived->m_iGenIdx);
                derived->generateUpdatePop(derived, derived->m_iPopIdx);
                ++(derived->m_iPopIdx);
            }
            else
            {
                Individual *p = Individual_Ctor();
                if (derived->isDominates(derived, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi))
                {
                    Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi);
                }
                else if (derived->isDominates(derived, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi))
                {
                    Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi);
                }
                else if (randomFloat(0.0, 1.0) < 0.5)
                {
                    Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 0)->m_indi);
                }
                else
                {
                    Individual_Assign(p, Vector_Individual_Visit(derived->m_vUpdatePop, 1)->m_indi);
                }
                updateBestFitness(derived, p);
                if (derived->m_iObjNum == 2)
                {
                    derived->update_neighbor_tchebi(derived, derived->m_iUpdatePopIdx, p);
                }
                else
                {
                    derived->update_neighbor_pbi(derived, derived->m_iUpdatePopIdx, p);
                }
                derived->updateParetoFront(derived, p);
#ifndef KERNEL_MODULE
                derived->base.dump_csv((ParamOptimizer *)derived);
#endif
                derived->generateUpdatePop(derived, derived->m_iPopIdx);
                ++(derived->m_iPopIdx);
                ++(derived->m_iUpdatePopIdx);
            }
            if (derived->m_iPopIdx == derived->m_iPopSize)
            {
                derived->m_iPopIdx = 0;
                ++(derived->m_iGenIdx);
            }
            if (derived->m_iUpdatePopIdx == derived->m_iPopSize)
            {
                derived->m_iUpdatePopIdx = 0;
            }
        }
    }
    update_idx = derived->m_updateindex->m_pair->m_val;
    iter = nullptr;
    if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
    {
        for (iter = Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        }
    }
    else
    {
        for (iter = Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        }
    }
    Vector_Pair_StringToInt_PushBack(derived->m_waitingindex, derived->m_updateindex->m_pair);
    derived->m_updateindex = Vector_Pair_StringToInt_Erase(derived->m_updateindex, derived->m_updateindex);
    return true_t;
}

#if FLOAT_PARAM
float MOEADOptimizer_optimize(ParamOptimizer *self, char *key, float min, float max)
#else
int MOEADOptimizer_optimize(ParamOptimizer *self, char *key, int min, int max)
#endif
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
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
            float cur = *(float *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr));
#else
            int cur = *(int *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr));
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
            *(float *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr)) = cur;
#else
            *(int *)(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr)) = cur;
#endif
            return cur;
        }
    }
    else
    {
        PRINTF("optimize is called when m_updateindex queue is empty, should not happen\n");
        return -32768.0;
    }
}
#if FLOAT_PARAM
void MOEADOptimizer_set_value(ParamOptimizer *self, char *key, float value)
#else
void MOEADOptimizer_set_value(ParamOptimizer *self, char *key, int value)
#endif
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
    {
        int update_idx = derived->m_updateindex->m_pair->m_val;
        if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
        {
            Map_StringToPtr iter = Map_StringToPtr_Find(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key);
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
            Map_StringToPtr iter = Map_StringToPtr_Find(Vector_Individual_Visit(derived->m_vUpdatePop, update_idx)->m_indi->m_mapPopParam, key);
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

static float norm(Vector_Float w)
{
    float sum = 0;
    int i;
    for (i = 0; i < Vector_Float_Size(w); ++i)
    {
        sum = sum + (*Vector_Float_Visit(w, i)->m_val) * (*Vector_Float_Visit(w, i)->m_val);
    }
    return sqrt(sum);
}
#ifndef UNIFORM_WEIGHT
static void adjust_normal(Vector_Float v)
{
    float normal = norm(v);
    int index;
    for (index = 0; index < Vector_Float_Size(v); ++index)
    {
        *Vector_Float_Visit(v, index)->m_val = *Vector_Float_Visit(v, index)->m_val / normal;
    }
}
#endif
static float calc_normal(Vector_Float v1, Vector_Float v2)
{
    float sum = 0.0;
    int index;
    if (Vector_Float_Size(v1) != Vector_Float_Size(v2))
    {
        PRINTF("v1 and v2 should be same size\n");
    }
    for (index = 0; index < Vector_Float_Size(v1); ++index)
    {
        float temp = (*Vector_Float_Visit(v1, index)->m_val - *Vector_Float_Visit(v2, index)->m_val);
        sum += (temp * temp);
    }
    return sqrt(sum);
}

static Vector_Int argsort(Vector_Float v)
{
    Vector_Int index = Vector_Int_Ctor();
    int i, bubble_i, bubble_j;
    for (i = 0; i < Vector_Float_Size(v); ++i)
    {
        Vector_Int_PushBack(index, i);
    }
    for (bubble_i = Vector_Int_Size(index) - 1; bubble_i > 0; --bubble_i)
    {
        for (bubble_j = 0; bubble_j < bubble_i; ++bubble_j)
        {
            if (*Vector_Float_Visit(v, *Vector_Int_Visit(index, bubble_j)->m_val)->m_val > *Vector_Float_Visit(v, *Vector_Int_Visit(index, bubble_j + 1)->m_val)->m_val)
            {
                int t = *Vector_Int_Visit(index, bubble_j)->m_val;
                *Vector_Int_Visit(index, bubble_j)->m_val = *Vector_Int_Visit(index, bubble_j + 1)->m_val;
                *Vector_Int_Visit(index, bubble_j + 1)->m_val = t;
            }
        }
    }
    return index;
}

void MOEADOptimizer_initgroup(MOEADOptimizer *self)
{
    int var, var_num, i;
#ifndef UNIFORM_WEIGHT
    int index;
#endif
    Vector_Float vf;
    Vector_Vector_Float p;
    Map_StringToPtr iter;
    if (self->m_iPopSize == 0)
    {
        return;
    }
    var_num = Map_StringToPtr_Size(Vector_Individual_Visit(self->m_vPop, 0)->m_indi->m_mapPopParam);
    vf = Vector_Float_Ctor();
    p = Vector_Vector_Float_Ctor();
    for (i = 0; i < var_num; ++i)
    {
        Vector_Float_PushBack(vf, 0.0);
    }
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        Vector_Vector_Float_PushBack(p, vf);
    }
    Vector_Float_Dtor(vf);
    vf = nullptr;
    self->m_funcSampling(p, 20);
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        var = 0;
        iter = nullptr;
        for (iter = Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            float d_dest = *Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val * (iter->m_ptr->dmax(iter->m_ptr) - iter->m_ptr->dmin(iter->m_ptr)) + iter->m_ptr->dmin(iter->m_ptr);
            float_to_string(self->base.m_strvalue,d_dest);
            iter->m_ptr->set(iter->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
            ++var;
        }
#ifndef UNIFORM_WEIGHT
        for (index = 0; index < self->m_iObjNum; ++index)
        {
            *Vector_Float_Visit(Vector_Vector_Float_Visit(self->m_vWeight, i)->m_vf, index)->m_val = randomFloat(0.0, 1.0);
        }
        adjust_normal(Vector_Vector_Float_Visit(self->m_vWeight, i)->m_vf);
#endif
        Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Initial", i);
    }
#ifdef UNIFORM_WEIGHT
    self->uniform_weight_generate(self);
#endif
    self->generate_neighbor(self);
}

void MOEADOptimizer_uniform_weight_generate(MOEADOptimizer *self)
{
    int i, stepSize, idx;
    long int temp, a;
    Vector_Int sequence;
    if (self->m_iObjNum == 2)
    {
        for (i = 0; i < self->m_iPopSize; ++i)
        {
            *Vector_Float_Visit(Vector_Vector_Float_Visit(self->m_vWeight, i)->m_vf, 0)->m_val = i * 1.0 / self->m_iPopSize;
            *Vector_Float_Visit(Vector_Vector_Float_Visit(self->m_vWeight, i)->m_vf, 1)->m_val = 1.0 - i * 1.0 / self->m_iPopSize;
        }
        return;
    }
    temp = 1;
    for (i = self->m_iObjNum - 1; i > 0; --i)
    {
        temp *= i;
    }
    stepSize = 1;
    a = 1;
    while (a < self->m_iPopSize * temp)
    {
        a = 1;
        for (i = self->m_iObjNum + stepSize - 1; i > stepSize; --i)
        {
            a *= i;
        }
        ++stepSize;
    }
    sequence = Vector_Int_Ctor();
    for (i = 0; i < stepSize; ++i)
    {
        Vector_Int_PushBack(sequence, 0);
    }
    for (i = 0; i < self->m_iObjNum - 1; ++i)
    {
        Vector_Int_PushBack(sequence, 1);
    }
    idx = 0;
    do
    {
        int s = -1;
        float w;
        Vector_Float weight = Vector_Float_Ctor();
        for (i = 0; i < (stepSize + self->m_iObjNum - 1); ++i)
        {
            if (*Vector_Int_Visit(sequence, i)->m_val == 1)
            {
                float w = i - s;
                w = (w - 1) / stepSize;
                s = i;
                Vector_Float_PushBack(weight, w);
            }
        }
        w = stepSize + self->m_iObjNum - 1 - s;
        w = (w - 1) / stepSize;
        Vector_Float_PushBack(weight, w);
        Vector_Float_Assign(Vector_Vector_Float_Visit(self->m_vWeight, idx)->m_vf, weight);
        ++idx;
    } while (Vector_Int_Next_Permutation(sequence) && idx < self->m_iPopSize);
}

void MOEADOptimizer_generate_neighbor(MOEADOptimizer *self)
{
    Vector_Vector_Float dist = Vector_Vector_Float_Ctor();
    Vector_Float temp = Vector_Float_Ctor();
    int i;
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        Vector_Float_PushBack(temp, 0.0);
    }
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        Vector_Vector_Float_PushBack(dist, temp);
    }
    Vector_Float_Dtor(temp);
    temp = nullptr;
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        int j;
        for (j = i + 1; j < self->m_iPopSize; ++j)
        {
            *Vector_Float_Visit(Vector_Vector_Float_Visit(dist, i)->m_vf, j)->m_val = calc_normal(Vector_Vector_Float_Visit(self->m_vWeight, i)->m_vf, Vector_Vector_Float_Visit(self->m_vWeight, j)->m_vf);
            *Vector_Float_Visit(Vector_Vector_Float_Visit(dist, j)->m_vf, i)->m_val = calc_normal(Vector_Vector_Float_Visit(self->m_vWeight, i)->m_vf, Vector_Vector_Float_Visit(self->m_vWeight, j)->m_vf);
        }
    }
    for (i = 0; i < self->m_iPopSize; ++i)
    {
        Vector_Int idx = argsort(Vector_Vector_Float_Visit(dist, i)->m_vf);
        int j;
        for (j = 0; j < self->m_iNeighNum; ++j)
        {
            *Vector_Int_Visit(Vector_Vector_Int_Visit(self->m_vNeighbor, i)->m_vi, j)->m_val = *Vector_Int_Visit(idx, j)->m_val;
        }
    }
}

float lambda_getMax_m(Vector_Float w, Vector_Float f1, Vector_Float f2)
{
    float max = 0;
    int i;
    for (i = 0; i < Vector_Float_Size(w); ++i)
    {
        if (i == 0)
        {
            max = *Vector_Float_Visit(w, i)->m_val * fabs(*Vector_Float_Visit(f1, i)->m_val - *Vector_Float_Visit(f2, i)->m_val);
        }
        else
        {
            if (max < *Vector_Float_Visit(w, i)->m_val * fabs(*Vector_Float_Visit(f1, i)->m_val - *Vector_Float_Visit(f2, i)->m_val))
            {
                max = *Vector_Float_Visit(w, i)->m_val * fabs(*Vector_Float_Visit(f1, i)->m_val - *Vector_Float_Visit(f2, i)->m_val);
            }
        }
    }
    return max;
}

void MOEADOptimizer_update_neighbor_tchebi(MOEADOptimizer *self, int idx, Individual *p)
{
    float (*getMax)(Vector_Float w, Vector_Float f1, Vector_Float f2) = lambda_getMax_m;
    int i;
    for (i = 0; i < Vector_Int_Size(Vector_Vector_Int_Visit(self->m_vNeighbor, idx)->m_vi); ++i)
    {
        int neighbor = *Vector_Int_Visit(Vector_Vector_Int_Visit(self->m_vNeighbor, idx)->m_vi, i)->m_val;
        float y = getMax(Vector_Vector_Float_Visit(self->m_vWeight, neighbor)->m_vf, p->m_fitness, self->m_bestFitness);
        float x = getMax(Vector_Vector_Float_Visit(self->m_vWeight, neighbor)->m_vf, Vector_Individual_Visit(self->m_vPop, neighbor)->m_indi->m_fitness, self->m_bestFitness);
        if (x >= y)
        {
            Individual_Assign(Vector_Individual_Visit(self->m_vPop, neighbor)->m_indi, p);
        }
    }
}

float lambda_getPbi_m(Vector_Float w, Vector_Float f1, Vector_Float f2)
{
    float temp = 0;
    float d1;
    float theta = 5.0;
    float dnorm_w = norm(w);
    Vector_Float vtemp;
    int i;
    for (i = 0; i < Vector_Float_Size(w); ++i)
    {
        temp += (*Vector_Float_Visit(w, i)->m_val * (*Vector_Float_Visit(f1, i)->m_val - *Vector_Float_Visit(f2, i)->m_val));
    }
    d1 = temp / dnorm_w;
    vtemp = Vector_Float_Ctor();
    for (i = 0; i < Vector_Float_Size(w); ++i)
    {
        Vector_Float_PushBack(vtemp, *Vector_Float_Visit(f1, i)->m_val - *Vector_Float_Visit(f2, i)->m_val - (d1 * *Vector_Float_Visit(w, i)->m_val / dnorm_w));
    }
    return d1 + theta * norm(vtemp);
}

void MOEADOptimizer_update_neighbor_pbi(MOEADOptimizer *self, int idx, Individual *p)
{
    float (*getPbi)(Vector_Float w, Vector_Float f1, Vector_Float f2) = lambda_getPbi_m;
    int i;
    for (i = 0; i < Vector_Int_Size(Vector_Vector_Int_Visit(self->m_vNeighbor, idx)->m_vi); ++i)
    {
        int neighbor = *Vector_Int_Visit(Vector_Vector_Int_Visit(self->m_vNeighbor, idx)->m_vi, i)->m_val;
        float y = getPbi(Vector_Vector_Float_Visit(self->m_vWeight, neighbor)->m_vf, p->m_fitness, self->m_bestFitness);
        float x = getPbi(Vector_Vector_Float_Visit(self->m_vWeight, neighbor)->m_vf, Vector_Individual_Visit(self->m_vPop, neighbor)->m_indi->m_fitness, self->m_bestFitness);
        if (x >= y)
        {
            Individual_Assign(Vector_Individual_Visit(self->m_vPop, neighbor)->m_indi, p);
        }
    }
}

bool_t MOEADOptimizer_isDominates(MOEADOptimizer *self, Individual *pop1, Individual *pop2)
{
    bool_t dominate = true_t;
    bool_t equal = true_t;
    int i;
    if (Vector_Float_Size(pop1->m_fitness) != Vector_Float_Size(pop2->m_fitness))
    {
        PRINTF("pop1 fitness and pop2 fitness should be same size\n");
    }
    for (i = 0; i < Vector_Float_Size(pop1->m_fitness); ++i)
    {
        equal = equal ? *Vector_Float_Visit(pop1->m_fitness, i)->m_val == *Vector_Float_Visit(pop2->m_fitness, i)->m_val : equal;
        if (*Vector_Float_Visit(pop1->m_fitness, i)->m_val > *Vector_Float_Visit(pop2->m_fitness, i)->m_val)
        {
            dominate = false_t;
            break;
        }
    }
    return equal ? false_t : dominate;
}

bool_t lambda_checkEqual_m(Individual *pop1, Individual *pop2)
{
    int i;
    bool_t equal = true_t;
    if (Vector_Float_Size(pop1->m_fitness) != Vector_Float_Size(pop2->m_fitness))
    {
        PRINTF("pop1 fitness and pop2 fitness should be same size\n");
    }
    for (i = 0; i < Vector_Float_Size(pop1->m_fitness); ++i)
    {
        if (*Vector_Float_Visit(pop1->m_fitness, i)->m_val != *Vector_Float_Visit(pop2->m_fitness, i)->m_val)
        {
            equal = false_t;
            break;
        }
    }
    return equal;
}

void MOEADOptimizer_updateParetoFront(MOEADOptimizer *self, Individual *p)
{
    Vector_Individual iter = self->m_vPF;
    bool_t drop = false_t;
    bool_t (*checkEqual)(Individual * pop1, Individual * pop2) = lambda_checkEqual_m;
    while (iter != nullptr)
    {
        if (self->isDominates(self, p, iter->m_indi))
        {
            Vector_Individual_Erase(&(self->m_vPF), iter->m_indi);
            iter = self->m_vPF; // dc
        }
        else
        {
            if (self->isDominates(self, iter->m_indi, p) || checkEqual(iter->m_indi, p))
            {
                drop = true_t;
            }
            iter = iter->m_next;
        }
    }
    if (!drop)
    {
        Vector_Individual_PushBack(self->m_vPF, p);
    }
}

void MOEADOptimizer_initParetoFront(MOEADOptimizer *self)
{
    int size = Vector_Individual_Size(self->m_vPop);
    Vector_Int result = Vector_Int_Ctor();
    int i;
    for (i = 0; i < size; ++i)
    {
        Vector_Int_PushBack(result, 0);
    }
    for (i = 0; i < size; ++i)
    {
        int j;
        for (j = i + 1; j < size; j++)
        {
            if (self->isDominates(self, Vector_Individual_Visit(self->m_vPop, i)->m_indi, Vector_Individual_Visit(self->m_vPop, j)->m_indi))
            {
                *Vector_Int_Visit(result, j)->m_val = 1;
            }
            if (self->isDominates(self, Vector_Individual_Visit(self->m_vPop, j)->m_indi, Vector_Individual_Visit(self->m_vPop, i)->m_indi))
            {
                *Vector_Int_Visit(result, i)->m_val = 1;
            }
        }
    }
    for (i = 0; i < size; ++i)
    {
        if (!Vector_Int_Visit(result, i)->m_val)
        {
            Vector_Individual_PushBack(self->m_vPF, Vector_Individual_Visit(self->m_vPop, i)->m_indi);
        }
    }
}

bool_t MOEADOptimizer_isTrainingEnd(ParamOptimizerIF *self)
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    bool_t terminate = false_t;
    if (derived->m_iGenNum == 1)
    {
        terminate = (derived->m_iGenIdx == derived->m_iGenNum);
    }
    if (!terminate && !derived->m_bExit)
    {
        return false_t;
    }
    return true_t;
}

#ifndef KERNEL_MODULE
void MOEADOptimizer_dump_csv(ParamOptimizer *self)
{
    MOEADOptimizer *derived = (MOEADOptimizer *)self;
    FILE *fp = self->m_csvOut;
    if (!fp || !Vector_Individual_Size(derived->m_vPF))
    {
        return;
    }

    if (!derived->m_iGenIdx)
    {
        fprintf(fp, "iter,");
        Map_StringToPtr iter = nullptr;
        for (iter = Vector_Individual_Visit(derived->m_vPF, 0)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            fprintf(fp, "%s,", iter->m_string);
        }
        int i;
        for (i = 0; i < Vector_Float_Size(Vector_Individual_Visit(derived->m_vPF, 0)->m_indi->m_fitness); ++i)
        {
            if (i == Vector_Float_Size(Vector_Individual_Visit(derived->m_vPF, 0)->m_indi->m_fitness) - 1)
            {
                fprintf(fp, "fitness%d\n", i);
            }
            else
            {
                fprintf(fp, "fitness%d,", i);
            }
        }
    }
    int i;
    for (i = 0; i < Vector_Individual_Size(derived->m_vPF); ++i)
    {
        int size = Map_StringToPtr_Size(Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_mapPopParam);
        int j = 0;
        Map_StringToPtr iter = nullptr;
        for (iter = Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_mapPopParam; iter != nullptr; iter = iter->m_next)
        {
            if (j == 0)
            {
                fprintf(fp, "%d,", derived->m_iGenIdx);
            }
            fprintf(fp, "%s,", iter->m_ptr->to_string(iter->m_ptr));
            if (j == size - 1)
            {
                int idx;
                for (idx = 0; idx < Vector_Float_Size(Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_fitness); ++idx)
                {
                    if (idx == Vector_Float_Size(Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_fitness) - 1)
                    {
                        fprintf(fp, "%f\n", *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_fitness, idx)->m_val);
                    }
                    else
                    {
                        fprintf(fp, "%f,", *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPF, i)->m_indi->m_fitness, idx)->m_val);
                    }
                }
            }
            ++j;
        }
    }
}
#endif

MOEADOptimizer *MOEADOptimizer_Ctor(OptParam *param)
{
    int i;
    MOEADOptimizer *self = (MOEADOptimizer *)MALLOC(sizeof(MOEADOptimizer));
    if (self)
    {
        ParamOptimizer_Ctor(&(self->base), param);
        self->base.base.update = MOEADOptimizer_update;
        //self->base.base.completeTrial = ;
        self->base.base.getTrial = MOEADOptimizer_getTrial;
        self->base.base.getAlgorithm = MOEADOptimizer_getAlgorithm;
        self->base.base.regist = MOEADOptimizer_regist;
        //self->base.base.unregist = ;
        //self->base.base.getOptimizedParam = ;
        self->base.base.getOptimizedParams = MOEADOptimizer_getOptimizedParams;
        //self->base.base.getOptimizedTarget = ;
        self->base.base.getOptimizedTargets = MOEADOptimizer_getOptimizedTargets;
        //self->base.base.getCurrentParam = ;
        //self->base.base.calibrateParam = ;
        self->base.base.isTrainingEnd = MOEADOptimizer_isTrainingEnd;
        //self->base.base.initLogging = ;
        //self->base.base.setPCAWindow = ;

        //self->base.pca_analysis = ;
        //self->base.append_sample = ;
        self->base.update_intern = MOEADOptimizer_update_intern;
        self->base.update_intern_param = MOEADOptimizer_update_intern_param;
#ifndef KERNEL_MODULE
        self->base.dump_csv = MOEADOptimizer_dump_csv;
#else
        self->base.dump_csv = NULL;
#endif
        //self->base.isInHistory = ;
        //self->base.isSame = ;
        self->base.optimize = MOEADOptimizer_optimize;
        self->base.set_value = MOEADOptimizer_set_value;

        self->generateUpdatePop = MOEADOptimizer_generateUpdatePop;
        self->isDominates = MOEADOptimizer_isDominates;
        self->initParetoFront = MOEADOptimizer_initParetoFront;
        self->updateParetoFront = MOEADOptimizer_updateParetoFront;
        self->initgroup = MOEADOptimizer_initgroup;
        self->generate_neighbor = MOEADOptimizer_generate_neighbor;
        self->update_neighbor_tchebi = MOEADOptimizer_update_neighbor_tchebi;
        self->update_neighbor_pbi = MOEADOptimizer_update_neighbor_pbi;
        self->uniform_weight_generate = MOEADOptimizer_uniform_weight_generate;

        self->m_iGenNum = 50;
        self->m_iPopSize = 100;
        self->m_iObjNum = 2;
        self->m_iNeighNum = 2;
        self->m_iGenIdx = 0;
        self->m_iPopIdx = 0;
        self->m_iUpdatePopIdx = 0;
        self->m_dcr = 20.0;
        self->m_dmut = 20.0;
        self->m_dmutp = 0.1;
        self->m_initPop = true_t;
        self->m_vPop = Vector_Individual_Ctor();
        self->m_vUpdatePop = Vector_Individual_Ctor();
        self->m_updateindex = Vector_Pair_StringToInt_Ctor();
        self->m_waitingindex = Vector_Pair_StringToInt_Ctor();
        self->m_vPF = Vector_Individual_Ctor();
        self->m_vNeighbor = Vector_Vector_Int_Ctor();
        self->m_vWeight = Vector_Vector_Float_Ctor();
        self->m_bestFitness = Vector_Float_Ctor();
        self->m_bExit = false_t;

        if (param->algorithm == MOEAD)
        {
            MOEADOptParam *p = (MOEADOptParam *)param;
            self->m_iGenNum = p->gen_num;
            self->m_iPopSize = p->pop_size;
            self->m_iObjNum = p->obj_num;
            self->m_iNeighNum = self->m_iPopSize * 0.15;
            RANGE_CLIP(self->m_iNeighNum, 15, 2);
        }
        Vector_Individual_Resize(self->m_vPop, self->m_iPopSize);
        Vector_Individual_Resize(self->m_vUpdatePop, 2);
        Vector_Float_Resize(Vector_Individual_Visit(self->m_vUpdatePop, 0)->m_indi->m_fitness, self->m_iObjNum);
        Vector_Float_Resize(Vector_Individual_Visit(self->m_vUpdatePop, 1)->m_indi->m_fitness, self->m_iObjNum);
        Vector_Vector_Float_Resize(self->m_vWeight, self->m_iPopSize); // dc
        Vector_Vector_Int_Resize(self->m_vNeighbor, self->m_iPopSize); // dc
        for (i = 0; i < self->m_iPopSize; ++i)
        {
            Vector_Float_Resize(Vector_Vector_Float_Visit(self->m_vWeight, i)->m_vf, self->m_iObjNum);
            Vector_Int_Resize(Vector_Vector_Int_Visit(self->m_vNeighbor, i)->m_vi, self->m_iNeighNum);
            Vector_Float_Resize(Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_fitness, self->m_iObjNum);
        }
        Vector_Float_Resize(self->m_bestFitness, self->m_iObjNum);
        self->m_funcCrossover = Crossover_adaptive_simulatedBinaryCrossover;
        self->m_funcMutate = Mutate_adaptive_polynomialMutate;
        self->m_funcSampling = Sampling_MonteCarloSampling;
    }
    return self;
}

void MOEADOptimizer_Dtor(MOEADOptimizer *self)
{
    ParamOptimizer_Dtor(&(self->base));
    self->generateUpdatePop = nullptr;
    self->isDominates = nullptr;
    self->initParetoFront = nullptr;
    self->updateParetoFront = nullptr;
    self->initgroup = nullptr;
    self->generate_neighbor = nullptr;
    self->update_neighbor_tchebi = nullptr;
    self->update_neighbor_pbi = nullptr;
    self->uniform_weight_generate = nullptr;
    Vector_Individual_Dtor(self->m_vPop);
    self->m_vPop = nullptr;
    Vector_Individual_Dtor(self->m_vUpdatePop);
    self->m_vUpdatePop = nullptr;
    Vector_Pair_StringToInt_Dtor(self->m_updateindex);
    self->m_updateindex = nullptr;
    Vector_Pair_StringToInt_Dtor(self->m_waitingindex);
    self->m_waitingindex = nullptr;
    Vector_Individual_Dtor(self->m_vPF);
    self->m_vPF = nullptr;
    Vector_Vector_Int_Dtor(self->m_vNeighbor);
    self->m_vNeighbor = nullptr;
    Vector_Vector_Float_Dtor(self->m_vWeight);
    self->m_vWeight = nullptr;
    Vector_Float_Dtor(self->m_bestFitness);
    self->m_bestFitness = nullptr;
    FREE(self);
    return;
}
