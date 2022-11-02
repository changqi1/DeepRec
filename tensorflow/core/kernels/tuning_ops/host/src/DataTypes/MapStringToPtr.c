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

#include "MapStringToPtr.h"
#include "Optimizer.h"

Map_StringToPtr Map_StringToPtr_Ctor()
{
    Pair_StringToPtr *p = (Pair_StringToPtr *)MALLOC(sizeof(Pair_StringToPtr));
    if (p)
    {
        p->m_string = nullptr;
        p->m_ptr = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Map_StringToPtr_Dtor(Map_StringToPtr msp)
{
    while (msp)
    {
        Pair_StringToPtr *cur = msp;
        msp = msp->m_next;
        FREE(cur->m_string);
        cur->m_string = nullptr;
        OptimizedParam *derived = (OptimizedParam *)cur->m_ptr;
        if (derived){
            FREE(derived->m_strName);
            derived->m_strName=nullptr;
            FREE(derived->m_strValue);
            derived->m_strValue=nullptr;
            derived = nullptr;
        }
        FREE(cur->m_ptr);
        cur->m_ptr = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

int Map_StringToPtr_Size(Map_StringToPtr msp)
{
    int size = 0;
    while (msp && msp->m_string && msp->m_ptr)
    {
        ++size;
        msp = msp->m_next;
    }
    return size;
}

void Map_StringToPtr_Print(Map_StringToPtr msp)
{
    Pair_StringToPtr *iter = msp;
    while (iter)
    {
        if (iter->m_string && iter->m_ptr)
        {
            PRINTF("(%s: %s)", iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
        }
        else
        {
            PRINTF("%s:%d nullptr detected!\n", __func__, __LINE__);
        }
        if (iter->m_next)
        {
            PRINTF(", ");
        }
        iter = iter->m_next;
    }
}

Pair_StringToPtr *Map_StringToPtr_Find(Map_StringToPtr msp, char *key)
{
    while (msp && msp->m_string && key)
    {
        if (strcmp(msp->m_string, key) == 0)
        {
            return msp;
        }
        msp = msp->m_next;
    }
    //PRINTF("%s: \"%s\" not found\n", __func__, key);
    return nullptr;
}

void Map_StringToPtr_PushBack(Map_StringToPtr msp, char *str, OptimizedParamIF *ptr)
{
    if (msp && str)
    {
        if (Map_StringToPtr_Find(msp, str) == nullptr) // key word not exsiting
        {
            if (msp->m_string && msp->m_ptr)
            {
                Pair_StringToPtr *p = (Pair_StringToPtr *)MALLOC(sizeof(Pair_StringToPtr));
                Pair_StringToPtr *end = msp;
                while (end && end->m_next)
                {
                    end = end->m_next;
                }
                if (p)
                {
                    p->m_string = (char *)MALLOC(strlen(str) + 1);
                    p->m_ptr = ptr;
                    p->m_next = nullptr;
                    if (p->m_string)
                    {
                        strcpy(p->m_string, str);
                        end->m_next = p;
                        return;
                    }
                }
                else
                {
                    PRINTF("%s:%d failed!\n", __func__, __LINE__);
                }
            }
            else
            {
                msp->m_string = (char *)MALLOC(strlen(str) + 1);
                msp->m_ptr = ptr;
                msp->m_next = nullptr;
                if (msp->m_string)
                {
                    strcpy(msp->m_string, str);
                    return;
                }
            }
        }
        else // key word exsiting
        {
            Map_StringToPtr_Visit(msp, str)->m_ptr = ptr;
            //PRINTF("%s:%d msp[%s] has been updated\n", __func__, __LINE__, str);
        }
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
}

Map_StringToPtr Map_StringToPtr_Erase(Map_StringToPtr msp, char *key)
{
    if (msp && key)
    {
        Pair_StringToPtr *p = Map_StringToPtr_Find(msp, key);
        if (p) // target found
        {
            if (p == msp) //first node
            {
                if (msp->m_next) // size > 1
                {
                    msp = msp->m_next;
                    FREE(p->m_string);
                    p->m_string = nullptr;
                    FREE(p->m_ptr);
                    p->m_ptr = nullptr;
                    p->m_next = nullptr;
                    FREE(p);
                    p = nullptr;
                    return msp;
                }
                else // size = 1
                {
                    FREE(msp->m_string);
                    msp->m_string = nullptr;
                    FREE(msp->m_ptr);
                    msp->m_ptr = nullptr;
                    return msp;
                }
            }
            else
            {
                Pair_StringToPtr *former = msp;
                while (former && former->m_next != p)
                {
                    former = former->m_next;
                }
                if (former)
                {
                    former->m_next = former->m_next->m_next;
                    FREE(p->m_string);
                    p->m_string = nullptr;
                    FREE(p->m_ptr);
                    p->m_ptr = nullptr;
                    FREE(p);
                    p->m_next = nullptr;
                    FREE(p);
                    p = nullptr;
                }
                return msp;
            }
        }
        else
        {
            PRINTF("%s: Target doesn't exisit.\n", __func__);
            return nullptr;
        }
    }
    else
    {
        PRINTF("%s: nullptr detected.\n", __func__);
        return nullptr;
    }
}

Pair_StringToPtr *Map_StringToPtr_Visit(Map_StringToPtr msp, char *key)
{
    while (msp && msp->m_string && key)
    {
        if (strcmp(msp->m_string, key) == 0)
        {
            return msp;
        }
        msp = msp->m_next;
    }
    //PRINTF("%s:%d \"%s\" not found\n", __func__, __LINE__, key);
    return nullptr;
}

bool_t Map_StringToPtr_IsSame(Map_StringToPtr m1, Map_StringToPtr m2)
{
    while (m1 && m2)
    {
        if (strcmp(m1->m_string, m2->m_string))
        {
            return false_t;
        }
        if (m1->m_ptr != m2->m_ptr)
        {
            return false_t;
        }
        m1 = m1->m_next;
        m2 = m2->m_next;
    }
    if (!m1 && !m2)
    {
        return true_t;
    }
    else
    {
        return false_t;
    }
}

void Map_StringToPtr_Assign(Map_StringToPtr m1, Map_StringToPtr m2)
{
    Pair_StringToPtr *iter;
    Map_StringToPtr_Dtor(m1->m_next);
    m1->m_next = nullptr;
    FREE(m1->m_string);
    m1->m_string = nullptr;
    FREE(m1->m_ptr);
    m1->m_ptr = nullptr;
    iter = m2;
    while (iter)
    {
        Map_StringToPtr_PushBack(m1, iter->m_string, iter->m_ptr->clone(iter->m_ptr));
        iter = iter->m_next;
    }
}
