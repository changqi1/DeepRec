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

#include "Individual.h"
#include "Optimizer.h"

Individual *Individual_Ctor()
{
    Individual *self = (Individual *)MALLOC(sizeof(Individual));
    if (self)
    {
        self->m_mapPopParam = Map_StringToPtr_Ctor();
        self->m_fitness = Vector_Float_Ctor();
        self->m_valueStr = strValue_Ctor();
    }
    else
    {
        PRINTF("%s:%d failed\n", __func__, __LINE__);
    }
    return self;
}

Individual *Individual_Copy_Ctor(Individual *param)
{
    Individual *self = Individual_Ctor();
    Map_StringToPtr iter;
    Vector_Float_Assign(self->m_fitness, param->m_fitness);
    iter = param->m_mapPopParam;
    while (iter)
    {
        if (iter->m_ptr && iter->m_string)
        {
            Map_StringToPtr_PushBack(self->m_mapPopParam, iter->m_string, iter->m_ptr->clone(iter->m_ptr));
            iter = iter->m_next;
        }
        else
        {
            break;
        }
    }
    return self;
}

void Individual_Copy(Individual *dst, Individual *src)
{
    Vector_Float iter1;
    Map_StringToPtr iter2;
    if (dst->m_fitness)
    {
        Vector_Float_Dtor(dst->m_fitness);
        dst->m_fitness = nullptr;
    }
    if (dst->m_mapPopParam)
    {
        Map_StringToPtr_Dtor(dst->m_mapPopParam);
        dst->m_mapPopParam = nullptr;
    }
    if (dst->m_valueStr)
    {
        strValue_Dtor(dst->m_valueStr);
        dst->m_valueStr = nullptr;
    }
    dst->m_fitness = Vector_Float_Ctor();
    iter1 = src->m_fitness;
    while (iter1)
    {
        Vector_Float_PushBack(dst->m_fitness, *iter1->m_val);
        iter1 = iter1->m_next;
    }
    dst->m_mapPopParam = Map_StringToPtr_Ctor();
    iter2 = src->m_mapPopParam;
    while (iter2)
    {
        Map_StringToPtr_PushBack(dst->m_mapPopParam, iter2->m_string, iter2->m_ptr->clone(iter2->m_ptr));
        iter2 = iter2->m_next;
    }
    dst->m_valueStr = strValue_Ctor();
    if (src &&src->m_valueStr&& dst && dst->m_valueStr){
        strcpy(dst->m_valueStr, src->m_valueStr);
    }
}

void Individual_Dtor(Individual *self)
{
    Map_StringToPtr_Dtor(self->m_mapPopParam);
    self->m_mapPopParam = nullptr;
    Vector_Float_Dtor(self->m_fitness);
    strValue_Dtor(self->m_valueStr);
    self->m_valueStr=nullptr;
    self->m_fitness = nullptr;
    FREE(self);
    self = nullptr;
    return;
}

void Individual_Assign(Individual *self, Individual *src)
{
    Map_StringToPtr iter;
    Vector_Float_Assign(self->m_fitness, src->m_fitness);
    iter = src->m_mapPopParam;
    while (iter)
    {
        Map_StringToPtr_PushBack(self->m_mapPopParam, iter->m_string, iter->m_ptr->clone(iter->m_ptr));
        iter = iter->m_next;
    }
    return;
}

bool_t Individual_IsSame(Individual *self, Map_StringToString p)
{
    Map_StringToPtr iter;
    Map_StringToString piter;
    int i;
    if (Map_StringToPtr_Size(self->m_mapPopParam) != Map_StringToString_Size(p))
    {
        return false_t;
    }
    iter = self->m_mapPopParam;
    piter = p;
    for (i = 0; i < Map_StringToString_Size(p); ++i)
    {
        if (strcmp(piter->m_key, iter->m_string) || strcmp(piter->m_value, iter->m_ptr->to_string(iter->m_ptr)))
        {
            return false_t;
        }
        iter = iter->m_next;
        piter = piter->m_next;
    }
    return true_t;
}

void Individual_Print(Individual *self)
{
    PRINTF("{");
    Map_StringToPtr_Print(self->m_mapPopParam);
    PRINTF(", ");
    Vector_Float_Print(self->m_fitness);
    PRINTF("}");
}
