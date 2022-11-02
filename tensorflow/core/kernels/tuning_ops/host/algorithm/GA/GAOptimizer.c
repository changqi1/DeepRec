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

#include "GAOptimizer.h"

void GAOptimizer_update(ParamOptimizerIF *self)
{
    ParamOptimizer_update(self);
}

#if FLOAT_PARAM
void GAOptimizer_regist(ParamOptimizerIF *self, char *key, float min, float max, int (*update)(char *, float))
#else
void GAOptimizer_regist(ParamOptimizerIF *self, char *key, int min, int max, int (*update)(char *, int))
#endif
{
    GAOptimizer *derived = (GAOptimizer *)self;
    OptimizedParamIF *p = (OptimizedParamIF *)OptimizedParam_Ctor(key, min, max, min, update, derived->base.optimize, derived->base.set_value);
    if (Map_StringToPtr_Find(derived->base.m_mapParam, key) == nullptr)
    {
        int i = 0;
        for (; i < derived->m_iPopSize; ++i)
        {
            Map_StringToPtr_PushBack(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_mapPopParam, key, p->clone(p));
            Map_StringToPtr_PushBack(Vector_Individual_Visit(derived->m_vOffPop, i)->m_indi->m_mapPopParam, key, p->clone(p));
            Vector_Float_Resize(Vector_Individual_Visit(derived->m_vOffPop, i)->m_indi->m_fitness, 1);
            *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vOffPop, i)->m_indi->m_fitness, 0)->m_val= -32768.0;
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

void GAOptimizer_getOptimizedParam(ParamOptimizerIF *self, Map_StringToString param)
{
    Map_StringToPtr iter;
    GAOptimizer *derived = (GAOptimizer *)self;
    derived->base.pca_analysis(&(derived->base), derived->m_best->m_mapPopParam);
    iter = derived->m_best->m_mapPopParam;
    if (!iter->m_string)
        return;
    while (iter)
    {
        Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        iter = iter->m_next;
    }
}

char *GAOptimizer_getOptimizedTarget(ParamOptimizerIF *self)
{
    GAOptimizer *derived = (GAOptimizer *)self;
    if (derived){
        if(Vector_Float_Visit(derived->m_best->m_fitness, 0)->m_val){
            float_to_string(derived->base.m_strvalue,*Vector_Float_Visit(derived->m_best->m_fitness, 0)->m_val);
            return derived->base.m_strvalue;
        }
        else{
            float_to_string(derived->base.m_strvalue,-32768.0);
            return derived->base.m_strvalue;
        }

    }
    else{
        return nullptr;
    }
}

Algorithm GAOptimizer_getAlgorithm(ParamOptimizerIF *self)
{
    return GA;
}

void lambda_select_best(GAOptimizer *derived)
{
    float fitness = *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, 0)->m_indi->m_fitness, 0)->m_val;
    int index = 0;
    int i = 1;
    for (; i < derived->m_iPopSize; ++i)
    {
        if (*Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_fitness, 0)->m_val > fitness)
        {
            index = i;
            fitness = *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_fitness, 0)->m_val;
        }
    }
    Individual_Copy(derived->m_best, Vector_Individual_Visit(derived->m_vPop, index)->m_indi);
}

void GAOptimizer_update_intern(ParamOptimizer *self)
{   

    GAOptimizer *derived = (GAOptimizer *)self;
    if (derived->m_iGenIdx == 0 && Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
    {
        derived->initgroup(derived);
    }
    else
    {   
        void (*select_best)(GAOptimizer *) = lambda_select_best;
        if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
        {
            int update_idx = derived->m_updateindex->m_pair->m_val;
            int off_idx;
            int retry;
            if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
            {   
                *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_fitness, 0)->m_val = *Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val;
            }
            else
            {   
                *Vector_Float_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_fitness, 0)->m_val = *Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val;
            }
            derived->m_updateindex = Vector_Pair_StringToInt_Erase(derived->m_updateindex, derived->m_updateindex);
            off_idx = update_idx + 1;
            retry = 10;
            while (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
            {   
                if (retry == 0)
                {   
                    if (!off_idx || derived->m_iGenIdx == derived->m_iGenNum - 1)
                    {   
                        derived->m_bExit = true_t;
                        Vector_Pair_StringToInt_PushBack_param(derived->m_updateindex, "Off", off_idx);                      
                        break;
                    }
                    else
                    {   
                        derived->m_offSize = off_idx;
                        retry = 10;
                    }
                }
                if (derived->m_iGenIdx != derived->m_iGenNum)
                {
                    if (off_idx == derived->m_offSize)
                    {   
                        if (derived->m_iGenIdx)
                        {   
                            derived->updateOffSpring(derived);
                        }
                        select_best(derived);
                        self->dump_csv(self);
                        off_idx = 0;
                    }
                    derived->generateOffPop(derived, off_idx);
                    if (off_idx == 0 && Vector_Pair_StringToInt_Size(derived->m_updateindex))
                    {
                        ++(derived->m_iGenIdx);
                        derived->m_offSize = derived->m_iPopSize;
                        break;
                    }
                }
                else
                {
                    derived->updateOffSpring(derived);
                    select_best(derived);
                    self->dump_csv(self);
                    break;
                }
                --retry;
            }
        }

    }
    if (!derived->m_bExit)
    {   
        derived->m_bExit = (derived->m_iGenIdx >= derived->m_iGenNum - 1) && Vector_Pair_StringToInt_Size(derived->m_updateindex) && derived->m_updateindex->m_pair->m_val == derived->m_iPopSize - 1;
    }
}

void GAOptimizer_update_intern_param(ParamOptimizer *self, Map_StringToString param, Vector_Float result)
{
    GAOptimizer *derived = (GAOptimizer *)self;
    Vector_Pair_StringToInt idx_iter = derived->m_waitingindex;
    bool_t found = false_t;
    Individual *p = nullptr;
    while (idx_iter)
    {
        if (strcmp(idx_iter->m_pair->m_string, "Initial") == 0)
        {
            p = Vector_Individual_Visit(derived->m_vPop, idx_iter->m_pair->m_val)->m_indi;
        }
        else
        {
            p = Vector_Individual_Visit(derived->m_vOffPop, idx_iter->m_pair->m_val)->m_indi;
        }
        idx_iter = idx_iter->m_next;
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
        return;
    }
    derived->m_bExit = (derived->m_iGenIdx >= derived->m_iGenNum) && (Vector_Pair_StringToInt_Size(derived->m_waitingindex) == 0);
}

bool_t GAOptimizer_getTrial(ParamOptimizerIF *self, Map_StringToString param)
{
    GAOptimizer *derived = (GAOptimizer *)self;
    int update_idx;
    Map_StringToPtr iter;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0 && Vector_Pair_StringToInt_Size(derived->m_waitingindex) != 0)
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
        void (*select_best)(GAOptimizer *) = lambda_select_best;
        int off_idx = derived->m_rec_idx + 1;
        int retry = 10;
        while (Vector_Pair_StringToInt_Size(derived->m_updateindex) == 0)
        {
            if (retry == 0)
            {
                if (!off_idx || derived->m_iGenIdx == derived->m_iGenNum - 1)
                {
                    derived->m_bExit = true_t;
                    Vector_Pair_StringToInt_PushBack_param(derived->m_updateindex, "Off", off_idx);
                    break;
                }
                else
                {
                    derived->m_offSize = off_idx;
                    retry = 10;
                }
            }
            if (derived->m_iGenIdx != derived->m_iGenNum)
            {
                if (off_idx == derived->m_offSize)
                {
                    if (derived->m_iGenIdx)
                    {
                        derived->updateOffSpring(derived);
                    }
                    select_best(derived);
                    derived->base.dump_csv((ParamOptimizer *)derived);
                    off_idx = 0;
                }
                derived->generateOffPop(derived, off_idx);
                if (off_idx == 0 && Vector_Pair_StringToInt_Size(derived->m_updateindex))
                {
                    ++(derived->m_iGenIdx);
                    derived->m_offSize = derived->m_iPopSize;
                    break;
                }
            }
            else
            {
                derived->updateOffSpring(derived);
                select_best(derived);
                derived->base.dump_csv((ParamOptimizer *)derived);
                break;
            }
            --retry;
        }
    }
    update_idx = derived->m_updateindex->m_pair->m_val;
    iter = nullptr;
    if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
    {
        iter = Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam;
        while (iter)
        {
            Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
            iter = iter->m_next;
        }
    }
    else
    {
        iter = Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam;
        while (iter)
        {
            Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
            iter = iter->m_next;
        }
    }
    derived->m_rec_idx = update_idx;
    Vector_Pair_StringToInt_PushBack(derived->m_waitingindex, derived->m_updateindex->m_pair);
    derived->m_updateindex = Vector_Pair_StringToInt_Erase(derived->m_updateindex, derived->m_updateindex);
    return true_t;
}

#if FLOAT_PARAM
float GAOptimizer_optimize(ParamOptimizer *self, char *key, float min, float max)
{
    GAOptimizer *derived = (GAOptimizer *)self;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
    {
        int update_idx = derived->m_updateindex->m_pair->m_val;
        if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
        {
            float cur = *(float *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr);
            if (cur > max)
            {
                cur = max;
            }
            if (cur < min)
            {
                cur = min;
            }
            *(float *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr) = cur;
            return cur;
        }
        else
        {
            float cur = *(float *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr);
            if (cur > max)
            {
                cur = max;
            }
            if (cur < min)
            {
                cur = min;
            }
            *(float *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr) = cur;
            return cur;
        }
    }
    else
    {
        PRINTF("optimize is called when m_updateindex queue is empty, should not happen\n");
        return -32768.0;
    }
}
#else
int GAOptimizer_optimize(ParamOptimizer *self, char *key, int min, int max)
{
    GAOptimizer *derived = (GAOptimizer *)self;
    if (Vector_Pair_StringToInt_Size(derived->m_updateindex))
    {
        int update_idx = derived->m_updateindex->m_pair->m_val;
        if (strcmp(derived->m_updateindex->m_pair->m_string, "Initial") == 0)
        {   
            int cur = *(int *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr);
            if (cur > max)
            {
                cur = max;
            }
            if (cur < min)
            {
                cur = min;
            }
            *(int *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr) = cur;
            return cur;
        }
        else
        {   
            int cur = *(int *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr);
            if (cur > max)
            {
                cur = max;
            }
            if (cur < min)
            {
                cur = min;
            }
            *(int *)Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key)->m_ptr) = cur;
            return cur;
        }
    }
    else
    {
        return -32768;
    }
}
#endif

#if FLOAT_PARAM
void GAOptimizer_set_value(ParamOptimizer *self, char *key, float value)
#else
void GAOptimizer_set_value(ParamOptimizer *self, char *key, int value)
#endif
{
    GAOptimizer *derived = (GAOptimizer *)self;
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
            Map_StringToPtr iter = Map_StringToPtr_Find(Vector_Individual_Visit(derived->m_vOffPop, update_idx)->m_indi->m_mapPopParam, key);
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

void GAOptimizer_generateOffPop(GAOptimizer *self, int update_idx)
{   
    int idx;
    if (Vector_Individual_Size(self->m_vPop) < 1)
    {
        return;
    }
    Vector_Individual_RandomShuffle(self->m_vPop);
    if (Vector_Individual_Size(self->m_vPop) == 1)
    {   
        Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi, Vector_Individual_Visit(self->m_vPop, 0)->m_indi);
    }
    else
    {   
        if (*Vector_Individual_Visit(self->m_vPop, 0)->m_indi->m_fitness->m_val > *Vector_Individual_Visit(self->m_vPop, 1)->m_indi->m_fitness->m_val)
        {
            Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi, Vector_Individual_Visit(self->m_vPop, 0)->m_indi);
        }
        else
        {
            Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi, Vector_Individual_Visit(self->m_vPop, 1)->m_indi);
        }
    }
    if (update_idx == self->m_iPopSize - 1)
    {
        self->m_funcMutate(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi, self->m_dmutp, 5);
        if (!self->base.isInHistory((ParamOptimizer *)self, Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi->m_mapPopParam))
        {
            Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Off", update_idx);
        }
        return;
    }
    if (Vector_Individual_Size(self->m_vPop) == 2)
    {
        if (*Vector_Individual_Visit(self->m_vPop, 0)->m_indi->m_fitness->m_val < *Vector_Individual_Visit(self->m_vPop, 1)->m_indi->m_fitness->m_val)
        {
            Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi, Vector_Individual_Visit(self->m_vPop, 0)->m_indi);
        }
        else
        {
            Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi, Vector_Individual_Visit(self->m_vPop, 1)->m_indi);
        }
    }
    else if (Vector_Individual_Size(self->m_vPop) == 3)
    {
        Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi, Vector_Individual_Visit(self->m_vPop, 2)->m_indi);
    }
    else
    {   
        if (*Vector_Individual_Visit(self->m_vPop, 2)->m_indi->m_fitness->m_val > *Vector_Individual_Visit(self->m_vPop, 3)->m_indi->m_fitness->m_val)
        {
            Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi, Vector_Individual_Visit(self->m_vPop, 2)->m_indi);
        }
        else
        {
            Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi, Vector_Individual_Visit(self->m_vPop, 3)->m_indi);
        }
    }

    self->m_funcCrossover(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi, Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi, 3.0);
    self->m_funcMutate(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi, self->m_dmutp, 5.0);
    self->m_funcMutate(Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi, self->m_dmutp, 5.0);
    float min_val = -32768.0;
    *Vector_Float_Visit(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi->m_fitness, 0)->m_val = min_val;
    *Vector_Float_Visit(Vector_Individual_Visit(self->m_vOffPop, update_idx+1)->m_indi->m_fitness, 0)->m_val = min_val;
    idx = update_idx;

    if (!self->base.isInHistory((ParamOptimizer *)self, Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi->m_mapPopParam))
    {
        Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Off", idx++);
    }
    if (!self->base.isInHistory((ParamOptimizer *)self, Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi->m_mapPopParam) && !self->base.isSame((ParamOptimizer *)self, Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi->m_mapPopParam, Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi->m_mapPopParam))
    {
        if (idx == update_idx)
        {   
            Individual_Copy(Vector_Individual_Visit(self->m_vOffPop, update_idx)->m_indi, Vector_Individual_Visit(self->m_vOffPop, update_idx + 1)->m_indi);
        }
        Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Off", idx);
    }
}

void GAOptimizer_initgroup(GAOptimizer *self)
{
    int var_num;
    Vector_Float temp;
    int i;
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
        Map_StringToPtr iter = Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_mapPopParam;
        while (iter)
        {
            float d_dest = *Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val * (iter->m_ptr->dmax(iter->m_ptr) - iter->m_ptr->dmin(iter->m_ptr)) + iter->m_ptr->dmin(iter->m_ptr);
            float_to_string(self->base.m_strvalue,d_dest);
            iter->m_ptr->set(iter->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
            ++var;
            iter = iter->m_next;
        }
        Vector_Pair_StringToInt_PushBack_param(self->m_updateindex, "Initial", i);
        Vector_Float_Resize(Vector_Individual_Visit(self->m_vPop, i)->m_indi->m_fitness, 1);
    }
    Vector_Vector_Float_Dtor(p);
    p = nullptr;
}

bool_t GAOptimizer_isTrainingEnd(ParamOptimizerIF *self)
{
    GAOptimizer *derived = (GAOptimizer *)self;
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

void GAOptimizer_updateOffSpring(GAOptimizer *self)
{
    Vector_Individual temp = Vector_Individual_Ctor();
    Vector_Int index = Vector_Int_Ctor();
    int i;
    int pop_id;
    int bubble_i;
    int bubble_j;
    for (i = 0; i < self->m_iPopSize; ++i)
    {   
        Vector_Individual_PushBack(temp, Vector_Individual_Visit(self->m_vPop, i)->m_indi);
        Vector_Int_PushBack(index, i * 2);
        Vector_Individual_PushBack(temp, Vector_Individual_Visit(self->m_vOffPop, i)->m_indi);
        Vector_Int_PushBack(index, i * 2 + 1);
    }

    for (bubble_i = Vector_Int_Size(index) - 1; bubble_i > 0; --bubble_i)
    {
        for (bubble_j = 0; bubble_j < bubble_i; ++bubble_j)
        {
            if (*Vector_Individual_Visit(temp, *Vector_Int_Visit(index, bubble_j)->m_val)->m_indi->m_fitness->m_val < *Vector_Individual_Visit(temp, *Vector_Int_Visit(index, bubble_j + 1)->m_val)->m_indi->m_fitness->m_val)
            {   //bubble_j fitness<bubble_j+1 fitness:
                int t = *Vector_Int_Visit(index, bubble_j)->m_val;
                *Vector_Int_Visit(index, bubble_j)->m_val = *Vector_Int_Visit(index, bubble_j + 1)->m_val;
                *Vector_Int_Visit(index, bubble_j + 1)->m_val = t;
            }
        }
    }

    i = 0;
    pop_id = 0;
    while (pop_id < self->m_iPopSize && i < self->m_iPopSize * 2)
    {
        if (!pop_id || (pop_id && !self->base.isSame((ParamOptimizer *)self, Vector_Individual_Visit(temp, *Vector_Int_Visit(index, i)->m_val)->m_indi->m_mapPopParam, Vector_Individual_Visit(self->m_vPop, pop_id - 1)->m_indi->m_mapPopParam)))
        {
            Individual_Copy(Vector_Individual_Visit(self->m_vPop, pop_id)->m_indi, Vector_Individual_Visit(temp, *Vector_Int_Visit(index, i)->m_val)->m_indi);
            ++pop_id;
            ++i;
        }
        else
        {
            ++i;
        }
    }

    for (i = pop_id; i < self->m_iPopSize; ++i)
    {
        Individual_Copy(Vector_Individual_Visit(self->m_vPop, i)->m_indi, Vector_Individual_Visit(self->m_vPop, i - pop_id)->m_indi);
    }
    Vector_Individual_Dtor(temp);
    temp = nullptr;
    Vector_Int_Dtor(index);
    index = nullptr;

}

#ifndef KERNEL_MODULE
void GAOptimizer_dump_csv(ParamOptimizer *self)
{
    GAOptimizer *derived = (GAOptimizer *)self;
    //float sum = 0, average = 0;
    int i;
    FILE *fp = self->m_csvOut;
    if (!fp || !derived->m_iPopSize)
    {
        return;
    }
    if (!derived->m_iGenIdx)
    {
        fprintf(fp, "iter,");
        Map_StringToPtr iter = Vector_Individual_Visit(derived->m_vPop, 0)->m_indi->m_mapPopParam;
        while (iter)
        {
            fprintf(fp, "%s,", iter->m_string);
            iter = iter->m_next;
        }
        fprintf(fp, "fitness\n");
    }
    for (i = 0; i < derived->m_iPopSize; ++i)
    {
        int size = Map_StringToPtr_Size(Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_mapPopParam);
        int j = 0;
        Map_StringToPtr iter = Vector_Individual_Visit(derived->m_vPop, i)->m_indi->m_mapPopParam;
        while (iter)
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
            iter = iter->m_next;
        }
    }
}
#endif

GAOptimizer *GAOptimizer_Ctor(OptParam *param)
{
    GAOptimizer *self = (GAOptimizer *)MALLOC(sizeof(GAOptimizer));
    if (self)
    {
        ParamOptimizer_Ctor(&(self->base), param);
        self->base.base.update = GAOptimizer_update;
        //self->base.base.completeTrial = ;
        self->base.base.getTrial = GAOptimizer_getTrial;
        self->base.base.getAlgorithm = GAOptimizer_getAlgorithm;
        self->base.base.regist = GAOptimizer_regist;
        //self->base.base.unregist = ;
        self->base.base.getOptimizedParam = GAOptimizer_getOptimizedParam;
        //self->base.base.getOptimizedParams = ;
        self->base.base.getOptimizedTarget = GAOptimizer_getOptimizedTarget;
        //self->base.base.getOptimizedTargets = ;
        //self->base.base.getCurrentParam = ;
        //self->base.base.calibrateParam = ;
        //static struct ParamOptimizerIF *getParamOptimizer(OptParam<TA> &param);
        self->base.base.isTrainingEnd = GAOptimizer_isTrainingEnd;
        //self->base.base.initLogging = ;
        //self->base.base.setPCAWindow = ;

        //self->base.pca_analysis = ;
        //self->base.append_sample = ;
        self->base.update_intern = GAOptimizer_update_intern;
        self->base.update_intern_param = GAOptimizer_update_intern_param;
#ifndef KERNEL_MODULE
        self->base.dump_csv = GAOptimizer_dump_csv;
#endif
        //self->base.isInHistory = ;
        //self->base.isSame = ;
        self->base.optimize = GAOptimizer_optimize;
        self->base.set_value = GAOptimizer_set_value;

        self->generateOffPop = GAOptimizer_generateOffPop;
        self->initgroup = GAOptimizer_initgroup;
        self->updateOffSpring = GAOptimizer_updateOffSpring;
        self->m_iGenNum = 50;
        self->m_iPopSize = 100;
        self->m_iGenIdx = 0;
        self->m_dmutp = 0.1;
        self->m_updateindex = Vector_Pair_StringToInt_Ctor();
        self->m_waitingindex = Vector_Pair_StringToInt_Ctor();
        self->m_best = Individual_Ctor();
        self->m_bExit = false_t;
        self->m_initPop = true_t;
        self->m_rec_idx = 0;
        if (param->algorithm == GA)
        {
            GAOptParam *p = (GAOptParam *)param;
            self->m_iGenNum = p->gen_num;
            self->m_iPopSize = p->pop_size;
            self->m_dmutp = p->mutp;
        }
        self->m_vPop = Vector_Individual_Ctor();
        Vector_Individual_Resize(self->m_vPop, self->m_iPopSize);
        self->m_vOffPop = Vector_Individual_Ctor();
        Vector_Individual_Resize(self->m_vOffPop, self->m_iPopSize);
        self->m_offSize = self->m_iPopSize;
        self->m_funcCrossover = Crossover_adaptive_simulatedBinaryCrossover;
        self->m_funcMutate = Mutate_adaptive_polynomialMutate;
        self->m_funcSampling = Sampling_LatinHypercubeSampling;
    }
    return self;
}

void GAOptimizer_Dtor(GAOptimizer *self)
{
    Vector_Individual_Dtor(self->m_vPop);
    self->m_vPop = nullptr;
    Vector_Individual_Dtor(self->m_vOffPop);
    self->m_vOffPop = nullptr;
    Vector_Pair_StringToInt_Dtor(self->m_updateindex);
    self->m_updateindex = nullptr;
    Vector_Pair_StringToInt_Dtor(self->m_waitingindex);
    self->m_waitingindex = nullptr;
    Individual_Dtor(self->m_best);
    self->m_best = nullptr;
    ParamOptimizer_Dtor(&(self->base));
    FREE(self);
}
