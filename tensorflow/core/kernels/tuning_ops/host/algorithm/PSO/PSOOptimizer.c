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

#include "PSOOptimizer.h"

static void PSOOptimizer_update(ParamOptimizerIF *self)
{   
    PSOOptimizer *derived = (PSOOptimizer *)self;
    //ParamOptimizer<TA>::update();
    ParamOptimizer_update(self);
    if (derived->m_iSwarmIdx == derived->m_iSwarmNum - 1)
    {
        derived->m_iSwarmIdx = 0;
        ++(derived->m_iIterIdx);
    }
    else
    {
        ++(derived->m_iSwarmIdx);
    }
}

#if FLOAT_PARAM
static void PSOOptimizer_regist(ParamOptimizerIF *self, char *key, float min, float max, int (*update)(char *, float))
#else
static void PSOOptimizer_regist(ParamOptimizerIF *self, char *key, int min, int max, int (*update)(char *, int))
#endif
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    OptimizedParamIF *p = (OptimizedParamIF *)OptimizedParam_Ctor(key, min, max, min, update, derived->base.optimize, derived->base.set_value);

    if (Map_StringToPtr_Find(derived->base.m_mapParam, key) == nullptr)
    {
        int i;
        float step;
        OptimizedParamIF *v;
        for (i = 0; i < derived->m_iSwarmNum; ++i)
        {
            Map_StringToPtr_PushBack(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, i)->m_msp, key, p->clone(p));
            Map_StringToPtr_PushBack(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, i)->m_msp, key, p->clone(p));
        }
        step = ceil(((float)(max - min) / derived->m_iIterNum / derived->m_dStepCount));
        derived->m_dStepMin = ((float)(max - min)) * 0.1;
        if (step < derived->m_dStepMin)
        {
            step = derived->m_dStepMin;
        }

        v = (OptimizedParamIF *)OptimizedParam_Ctor(key, min, step, -step, update, derived->base.optimize, derived->base.set_value);

        for (i = 0; i < derived->m_iSwarmNum; ++i)
        {
            Map_StringToPtr_PushBack(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmVelocity, i)->m_msp, key, v->clone(v));
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

static void PSOOptimizer_getOptimizedParam(ParamOptimizerIF *self, Map_StringToString param)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    Map_StringToPtr iter = derived->m_vmGlobalBestParam;
    if (!iter->m_string)
        return;
    derived->base.pca_analysis(&(derived->base), derived->m_vmGlobalBestParam);
    
    while (iter)
    {
        Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        iter = iter->m_next;
    }
}

static char *PSOOptimizer_getOptimizedTarget(ParamOptimizerIF *self)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    if (derived)
    {
        sprintf(derived->base.m_strvalue, "%f", derived->m_GTarget);
        return derived->base.m_strvalue;
    }
    else
    {
        return nullptr;
    }
}

static Algorithm PSOOptimizer_getAlgorithm(ParamOptimizerIF *self)
{
    return PSO;
}

static void PSOOptimizer_update_intern(ParamOptimizer *self)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    if (derived->m_iSwarmIdx != 0 || derived->m_iIterIdx != 0)
    {   
        int iPrevSwarmIdx = (derived->m_iIterIdx * derived->m_iSwarmNum + derived->m_iSwarmIdx - 1) % derived->m_iSwarmNum;
        int iPrevIterIdx = (derived->m_iIterIdx * derived->m_iSwarmNum + derived->m_iSwarmIdx - 1) / derived->m_iSwarmNum;
        if (iPrevIterIdx == 0)
        {
            Map_StringToPtr iter;
            *(Vector_Float_Visit(derived->m_vPTarget, iPrevSwarmIdx)->m_val) = *(Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val);
            iter = derived->base.m_mapParam;
            while (iter)
            {
                Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, iPrevSwarmIdx)->m_msp, iter->m_string)->m_ptr = iter->m_ptr->clone(iter->m_ptr);
                iter = iter->m_next;
            }
            if (iPrevSwarmIdx == 0 || *(Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val) > derived->m_GTarget)
            {
                Map_StringToPtr iter;
                derived->m_GTarget = *(Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val);
                iter = derived->base.m_mapParam;
                while (iter)
                {
                    if (Map_StringToPtr_Find(derived->m_vmGlobalBestParam, iter->m_string))
                    {
                        Map_StringToPtr_Visit(derived->m_vmGlobalBestParam, iter->m_string)->m_ptr = iter->m_ptr->clone(iter->m_ptr);
                    }
                    else
                    {
                        Map_StringToPtr_PushBack(derived->m_vmGlobalBestParam, iter->m_string, iter->m_ptr->clone(iter->m_ptr));
                    }
                    iter = iter->m_next;
                }
            }
        }
        else
        {
            if (*Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val > derived->m_GTarget)
            {
                Map_StringToPtr iter;
                derived->m_GTarget = *(Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val);
                iter = derived->base.m_mapParam;
                while (iter)
                {
                    Map_StringToPtr_Visit(derived->m_vmGlobalBestParam, iter->m_string)->m_ptr = iter->m_ptr->clone(iter->m_ptr);
                    iter = iter->m_next;
                }
            }
            if (*Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val > *(Vector_Float_Visit(derived->m_vPTarget, iPrevSwarmIdx)->m_val))
            {

                Map_StringToPtr iter;
                *(Vector_Float_Visit(derived->m_vPTarget, iPrevSwarmIdx)->m_val) = *(Vector_Float_Visit(derived->base.m_prevTarget, 0)->m_val);
                iter = derived->base.m_mapParam;
                while (iter)
                {
                    Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, iPrevSwarmIdx)->m_msp, iter->m_string)->m_ptr = iter->m_ptr->clone(iter->m_ptr);
                    iter = iter->m_next;
                }
            }
        }
        if (derived->m_iSwarmIdx == 0)
        {
#ifndef KERNEL_MODULE
            self->dump_csv(self);
#endif
        }
    }
    else
    {  
        int var_num = Map_StringToPtr_Size(derived->base.m_mapParam), i;
        Vector_Float temp = Vector_Float_Ctor();
        Vector_Vector_Float p;
        for (i = 0; i < var_num; ++i)
        {
            Vector_Float_PushBack(temp, 0.0);
        }
        p = Vector_Vector_Float_Ctor();
        for (i = 0; i < derived->m_iSwarmNum; ++i)
        {
            Vector_Vector_Float_PushBack(p, temp);
        }
        derived->m_funcSampling(p, 100);
        for (i = 0; i < derived->m_iSwarmNum; ++i)
        {
            int var = 0;
            Map_StringToPtr iter = Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, i)->m_msp;
            while (iter)
            {
                float d_dest = *(Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val) * (iter->m_ptr->dmax(iter->m_ptr) - iter->m_ptr->dmin(iter->m_ptr)) + iter->m_ptr->dmin(iter->m_ptr);
                float_to_string(derived->base.m_strvalue,d_dest);
                iter->m_ptr->set(iter->m_ptr, derived->base.m_strvalue, false_t, self);
                ++var;
                iter = iter->m_next;
            }
        }
    }
}

static bool_t PSOOptimizer_isSameParam(Map_StringToPtr param, Map_StringToString p)
{
    Map_StringToPtr iter;
    Map_StringToString piter;
    int i;
    if (Map_StringToPtr_Size(param) != Map_StringToString_Size(p))
    {
        return false_t;
    }
    iter = param;
    piter = p;
    for (i = 0; i < Map_StringToString_Size(p); ++i)
    {
        if (strcmp(piter->m_key, piter->m_key) || strcmp(piter->m_value, iter->m_ptr->to_string(iter->m_ptr)))
        {
            return false_t;
        }
        iter = iter->m_next;
        piter = piter->m_next;
    }
    return true_t;
}

static void PSOOptimizer_update_intern_param(ParamOptimizer *self, Map_StringToString param, Vector_Float result)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    Vector_Int idx_iter = derived->m_waitingindex;
    bool_t found = false_t;
    while (idx_iter)
    {
        Map_StringToPtr p = Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, *(idx_iter->m_val))->m_msp;
        if (PSOOptimizer_isSameParam(p, param))
        {
            if (derived->m_iIterIdx == 0 || *(Vector_Float_Visit(derived->m_vPTarget, *idx_iter->m_val)->m_val) < *(Vector_Float_Visit(result, 0)->m_val))
            {
                Map_StringToPtr iter;
                *(Vector_Float_Visit(derived->m_vPTarget, *idx_iter->m_val)->m_val) = *(Vector_Float_Visit(result, 0)->m_val);
                iter = p;
                while (iter)
                {
                    Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, *(idx_iter->m_val))->m_msp, iter->m_string)->m_ptr = iter->m_ptr->clone(iter->m_ptr);
                    iter = iter->m_next;
                }
            }
            if (!derived->m_rec || derived->m_GTarget < *(Vector_Float_Visit(result, 0)->m_val))
            {
                Map_StringToPtr iter;
                derived->m_GTarget = *(Vector_Float_Visit(result, 0)->m_val);
                iter = p;
                while (p)
                {
                    Map_StringToPtr_Visit(derived->m_vmGlobalBestParam, iter->m_string)->m_ptr = iter->m_ptr->clone(iter->m_ptr);
                    p = p->m_next;
                }
                derived->m_rec = true_t;
            }
            found = true_t;
            break;
        }
        idx_iter = idx_iter->m_next;
    }
    if (!found)
    {
        PRINTF("The trail is not is the waiting list\n");
        return;
    }
    Vector_Int_Erase(derived->m_waitingindex, idx_iter);
}

static bool_t PSOOptimizer_getTrial(ParamOptimizerIF *self, Map_StringToString param)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    int update_idx;
    Map_StringToPtr iter;
    if (Vector_Int_Size(derived->m_updateindex) == 0)
    {
        Vector_Int result = Vector_Int_Find(derived->m_waitingindex, derived->m_iSwarmIdx);
        if (result)
        {
            return false_t;
        }
    }
    if (derived->m_initSwarm)
    {
        derived->initSwarms(derived);
        derived->m_initSwarm = false_t;
    }
    else
    {
        if (Vector_Int_Size(derived->m_updateindex) == 0)
        {
            if (derived->m_iSwarmIdx == 0)
            {
                ++(derived->m_iIterIdx);
#ifndef KERNEL_MODULE
                ParamOptimizer *base = (ParamOptimizer *)derived;
                base->dump_csv(base);
#endif
            }
            derived->generateSwarm(derived, derived->m_iSwarmIdx);
            Vector_Int_PushBack(derived->m_updateindex, derived->m_iSwarmIdx);
            ++(derived->m_iSwarmIdx);
            if (derived->m_iSwarmIdx == derived->m_iSwarmNum)
            {
                derived->m_iSwarmIdx = 0;
            }
        }
    }
    update_idx = *(derived->m_updateindex->m_val);
    iter = Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, update_idx)->m_msp;
    while (iter)
    {
        Map_StringToString_Visit(param, iter->m_string)->m_key = iter->m_ptr->to_string(iter->m_ptr);
        iter = iter->m_next;
    }
    Vector_Int_PushBack(derived->m_waitingindex, *(derived->m_updateindex->m_val));
    derived->m_updateindex = Vector_Int_Erase(derived->m_updateindex, derived->m_updateindex);
    return true_t;
}

#if FLOAT_PARAM
static float PSOOptimizer_optimize(ParamOptimizer *self, char *key, float min, float max)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    if (derived->m_iIterIdx != 0)
    {
        OptimizedParamIF *pv = Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmVelocity, derived->m_iSwarmIdx)->m_msp, key)->m_ptr;
        float cur = *(float *)pv->cur(pv);
        float bp = *(float *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
        float bg = *(float *)Map_StringToPtr_Visit(derived->m_vmGlobalBestParam, key)->m_ptr->cur(Map_StringToPtr_Visit(derived->m_vmGlobalBestParam, key)->m_ptr);
        float sc = *(float *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
        float velocity = cur * derived->m_dUpdateWeight + (bp - sc) * derived->m_dUpdateCp * (float)RAND_FUNC / RAND_MAXV + (bg - sc) * derived->m_dUpdateCg * (float)RAND_FUNC / RAND_MAXV;
        float scale;
        if (velocity > max/**(float *)pv->max(pv)*/)
        {
            velocity = max/**(float *)pv->max(pv)*/;
        }
        if (velocity < min/**(float *)pv->min(pv)*/)
        {
            velocity = min/**(float *)pv->min(pv)*/;
        }
        *(float *)pv->cur(pv) = velocity;
        scale = *(float *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr) + velocity;
        if (scale > max)
        {
            scale = max;
        }
        if (scale < min)
        {
            scale = min;
        }
        *(float *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr) = scale;
        return *(float *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
    }
    else
    {
        return *(float *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
    }
}
#else
//更新每一个参数用的函数
static int PSOOptimizer_optimize(ParamOptimizer *self, char *key, int min, int max)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    if (derived->m_iIterIdx != 0)
    {
        OptimizedParamIF *pv = Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmVelocity, derived->m_iSwarmIdx)->m_msp, key)->m_ptr;//visit当前速度
        //PRINTF("!!!GET CUR PV\n");
        int cur = *(int *)pv->cur(pv);//得到当前速度的值//void *OptimizedParam_cur(OptimizedParamIF *self)
        int bp = *(int *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
        int bg = *(int *)Map_StringToPtr_Visit(derived->m_vmGlobalBestParam, key)->m_ptr->cur(Map_StringToPtr_Visit(derived->m_vmGlobalBestParam, key)->m_ptr);//会参考当前key的最优值
        //当前位置
        int sc = *(int *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
        int velocity = cur * derived->m_dUpdateWeight + (bp - sc) * derived->m_dUpdateCp * (float)RAND_FUNC / RAND_MAXV + (bg - sc) * derived->m_dUpdateCg * (float)RAND_FUNC / RAND_MAXV;
        int scale;
        if (velocity > max/**(int *)pv->max(pv)*/)
        {
            velocity = max/**(int *)pv->max(pv)*/;
        }
        if (velocity < min/**(int *)pv->min(pv)*/)
        {
            velocity = min/**(int *)pv->min(pv)*/;
        }
        *(int *)pv->cur(pv) = velocity;//更新速度
        //当前值+速度
        scale = *(int *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr) + velocity;
        if (scale > max)
        {
            scale = max;
        }
        if (scale < min)
        {
            scale = min;
        }
        //赋值更新后的参数
        *(int *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr) = scale;
        return *(int *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
    }
    else
    {
        return *(int *)Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr->cur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, derived->m_iSwarmIdx)->m_msp, key)->m_ptr);
    }
}
#endif

static bool_t PSOOptimizer_isTrainingEnd(ParamOptimizerIF *self)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    return derived->m_iIterIdx == derived->m_iIterNum;
}

static void PSOOptimizer_generateSwarm(PSOOptimizer *self, int SwarmIdx)
{
    Map_StringToPtr iter = Vector_Map_StringToPtr_Visit(self->m_mapSwarmParam, self->m_iSwarmIdx)->m_msp;
    while (iter)
    {
        OptimizedParamIF *pv = Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(self->m_mapSwarmVelocity, self->m_iSwarmIdx)->m_msp, iter->m_string)->m_ptr;
        float cur = pv->dcur(pv);
        float bp = Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(self->m_vmSwarmBestParam, self->m_iSwarmIdx)->m_msp, iter->m_string)->m_ptr->dcur(Map_StringToPtr_Visit(Vector_Map_StringToPtr_Visit(self->m_vmSwarmBestParam, self->m_iSwarmIdx)->m_msp, iter->m_string)->m_ptr);
        float bg = Map_StringToPtr_Visit(self->m_vmGlobalBestParam, iter->m_string)->m_ptr->dcur(Map_StringToPtr_Visit(self->m_vmGlobalBestParam, iter->m_string)->m_ptr);
        float sc = iter->m_ptr->dcur(iter->m_ptr);
        float velocity = cur * self->m_dUpdateWeight + (bp - sc) * self->m_dUpdateCp * (float)RAND_FUNC / RAND_MAXV + (bg - sc) * self->m_dUpdateCg * (float)RAND_FUNC / RAND_MAXV;
        float scale;
        if (velocity > pv->dmax(pv))
        {
            velocity = pv->dmax(pv);
        }
        if (velocity < pv->dmin(pv))
        {
            velocity = pv->dmin(pv);
        }
        float_to_string(self->base.m_strvalue,velocity);
        pv->set(pv, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
        scale = sc + velocity;
        if (scale > iter->m_ptr->dmax(iter->m_ptr))
        {
            scale = iter->m_ptr->dmax(iter->m_ptr);
        }
        if (scale < iter->m_ptr->dmin(iter->m_ptr))
        {
            scale = iter->m_ptr->dmin(iter->m_ptr);
        }
        float_to_string(self->base.m_strvalue,scale);
        iter->m_ptr->set(iter->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
        iter = iter->m_next;
    }
}

static void PSOOptimizer_initSwarms(PSOOptimizer *self)
{
    int var_num, i;
    Vector_Float temp;
    Vector_Vector_Float p;
    if (self->m_iSwarmNum == 0)
        return;
    var_num = Vector_Map_StringToPtr_Size(self->m_mapSwarmParam);
    temp = Vector_Float_Ctor();
    p = Vector_Vector_Float_Ctor();
    for (i = 0; i < var_num; ++i)
    {
        Vector_Float_PushBack(temp, 0.0);
    }
    for (i = 0; i < self->m_iSwarmNum; ++i)
    {
        Vector_Vector_Float_PushBack(p, temp);
    }
    self->m_funcSampling(p, 100);
    for (i = 0; i < self->m_iSwarmNum; ++i)
    {
        int var = 0;
        Map_StringToPtr iter = Vector_Map_StringToPtr_Visit(self->m_mapSwarmParam, i)->m_msp;
        while (iter)
        {
            float d_dest = *(Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val) * (iter->m_ptr->dmax(iter->m_ptr) - iter->m_ptr->dmin(iter->m_ptr)) + iter->m_ptr->dmin(iter->m_ptr);
            float_to_string(self->base.m_strvalue,d_dest);
            iter->m_ptr->set(iter->m_ptr, self->base.m_strvalue, false_t, (ParamOptimizer *)self);
            ++var;
            iter = iter->m_next;
        }
        Vector_Int_PushBack(self->m_updateindex, i);
    }
}
#if FLOAT_PARAM
static void PSOOptimizer_set_value(ParamOptimizer *self, char *key, float value)
#else
static void PSOOptimizer_set_value(ParamOptimizer *self, char *key, int value)
#endif
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    Map_StringToPtr iter;
    int iPrevSwarmIdx = (derived->m_iIterIdx * derived->m_iSwarmNum + derived->m_iSwarmIdx - 1) % derived->m_iSwarmNum;
    iter = Map_StringToPtr_Find(Vector_Map_StringToPtr_Visit(derived->m_mapSwarmParam, iPrevSwarmIdx)->m_msp, key);
    if (iter)
    {
#if FLOAT_PARAM
        *(float *)iter->m_ptr->cur(iter->m_ptr) = value;
#else
        *(int *)iter->m_ptr->cur(iter->m_ptr) = value;
#endif
    }
}

#ifndef KERNEL_MODULE
static void PSOOptimizer_dump_csv(ParamOptimizer *self)
{
    PSOOptimizer *derived = (PSOOptimizer *)self;
    //float sum = 0, average = 0;
    int i;
    FILE *fp = derived->base.m_csvOut;
    if (!fp || !derived->m_iSwarmNum)
        return;

    if (derived->m_iIterIdx == 1)
    {
        fprintf(fp, "iter,");
        Map_StringToPtr iter = Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, 0)->m_msp;
        while (iter)
        {
            fprintf(fp, "%s,", iter->m_string);
            iter = iter->m_next;
        }
        fprintf(fp, "fitness\n");
    }
    for (i = 0; i < derived->m_iSwarmNum; ++i)
    {
        int size = Map_StringToPtr_Size(Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, i)->m_msp);
        int j = 0;
        Map_StringToPtr iter = Vector_Map_StringToPtr_Visit(derived->m_vmSwarmBestParam, i)->m_msp;
        while (iter)
        {
            if (j == 0)
            {
                fprintf(fp, "%d,", derived->m_iIterIdx - 1);
            }
            fprintf(fp, "%s,", iter->m_ptr->to_string(iter->m_ptr));
            if (j == size - 1)
            {
                fprintf(fp, "%f\n", *Vector_Float_Visit(derived->m_vPTarget, i)->m_val);
            }
            ++j;
            iter = iter->m_next;
        }
    }
}
#endif

PSOOptimizer *PSOOptimizer_Ctor(OptParam *param)
{
    PSOOptimizer *self = (PSOOptimizer *)MALLOC(sizeof(PSOOptimizer));
    if (self)
    {
        ParamOptimizer_Ctor(&(self->base), param);
        self->base.base.update = PSOOptimizer_update;
        //self->base.base.completeTrial = ;
        self->base.base.getTrial = PSOOptimizer_getTrial;
        self->base.base.getAlgorithm = PSOOptimizer_getAlgorithm;
        self->base.base.regist = PSOOptimizer_regist;
        //self->base.base.unregist = ;
        self->base.base.getOptimizedParam = PSOOptimizer_getOptimizedParam;
        //self->base.base.getOptimizedParams = ;
        self->base.base.getOptimizedTarget = PSOOptimizer_getOptimizedTarget;
        //self->base.base.getOptimizedTargets = ;
        //self->base.base.getCurrentParam = ;
        //self->base.base.calibrateParam = ;
        //static struct ParamOptimizerIF *getParamOptimizer(OptParam<TA> &param);
        self->base.base.isTrainingEnd = PSOOptimizer_isTrainingEnd;
        //self->base.base.initLogging = ;
        //self->base.base.setPCAWindow = ;

        //self->base.pca_analysis = ;
        //self->base.append_sample = ;
        self->base.update_intern = PSOOptimizer_update_intern;
        self->base.update_intern_param = PSOOptimizer_update_intern_param;
#ifndef KERNEL_MODULE
        self->base.dump_csv = PSOOptimizer_dump_csv;
#endif
        //self->base.isInHistory = ;
        //self->base.isSame = ;
        self->base.optimize = PSOOptimizer_optimize;
        self->base.set_value = PSOOptimizer_set_value;

        self->initSwarms = PSOOptimizer_initSwarms;
        self->generateSwarm = PSOOptimizer_generateSwarm;
        self->m_iIterNum = 10;
        self->m_iSwarmNum = 5;
        self->m_iIterIdx = 0;
        self->m_iSwarmIdx = 0;
        self->m_dStepCount = 2.0;
        self->m_dStepMin = 5.0;
        self->m_dUpdateWeight = 0.8;
        self->m_dUpdateCp = 0.5;
        self->m_initSwarm = true_t;
        self->m_rec = false_t;
        if (param->algorithm == PSO)
        {
            PSOOptParam *p = (PSOOptParam *)param;
            self->m_iIterNum = p->iter_num;
            self->m_iSwarmNum = p->swarm_num;
            self->m_dUpdateWeight = p->update_weight;
            self->m_dUpdateCp = p->update_cp;
            self->m_dUpdateCg = p->update_cg;
        }
        self->m_mapSwarmVelocity = Vector_Map_StringToPtr_Ctor();
        self->m_mapSwarmParam = Vector_Map_StringToPtr_Ctor();
        self->m_vPTarget = Vector_Float_Ctor();
        self->m_vmSwarmBestParam = Vector_Map_StringToPtr_Ctor();
        self->m_GTarget = -32768.0;
        self->m_vmGlobalBestParam = Map_StringToPtr_Ctor();
        Vector_Float_Resize(self->m_vPTarget, self->m_iSwarmNum);
        Vector_Map_StringToPtr_Resize(self->m_vmSwarmBestParam, self->m_iSwarmNum);
        Vector_Map_StringToPtr_Resize(self->m_mapSwarmParam, self->m_iSwarmNum);
        Vector_Map_StringToPtr_Resize(self->m_mapSwarmVelocity, self->m_iSwarmNum);
        self->m_funcSampling = Sampling_LatinHypercubeSampling;
        self->m_updateindex = Vector_Int_Ctor();
        self->m_waitingindex = Vector_Int_Ctor();
    }
    return self;
}

void PSOOptimizer_Dtor(PSOOptimizer *self)
{
    if (self)
    {
        Vector_Float_Dtor(self->m_vPTarget);
        self->m_vPTarget = nullptr;
        Vector_Map_StringToPtr_Dtor(self->m_vmSwarmBestParam);
        self->m_vmSwarmBestParam = nullptr;
        Vector_Map_StringToPtr_Dtor(self->m_mapSwarmParam);
        self->m_mapSwarmParam = nullptr;
        Vector_Map_StringToPtr_Dtor(self->m_mapSwarmVelocity);
        self->m_mapSwarmVelocity = nullptr;
        ParamOptimizer_Dtor(&(self->base));
        FREE(self);
    }
    return;
}
