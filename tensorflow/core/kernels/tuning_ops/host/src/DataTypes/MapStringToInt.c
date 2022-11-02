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

#include "MapStringToInt.h"

Map_StringToInt Map_StringToInt_Ctor()
{
    Pair_StringToInt *p = (Pair_StringToInt *)MALLOC(sizeof(Pair_StringToInt));
    if (p)
    {
        p->m_string = nullptr;
        p->m_val = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Map_StringToInt_Dtor(Map_StringToInt msi)
{
    while (msi)
    {
        Pair_StringToInt *cur = msi;
        msi = msi->m_next;
        FREE(cur->m_string);
        cur->m_string = nullptr;
        FREE(cur->m_val);
        cur->m_val = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Map_StringToInt_Print(Map_StringToInt msi)
{
    PRINTF("{");
    while (msi)
    {
        if (msi->m_string && msi->m_val)
        {
            PRINTF("\"%s\": %d", msi->m_string, *msi->m_val);
        }
        if (msi->m_next)
        {
            PRINTF(", ");
        }
        msi = msi->m_next;
    }
    PRINTF("}\n");
}

Pair_StringToInt *Map_StringToInt_Find(Map_StringToInt msi, char *key)
{
    while (msi && msi->m_string && key)
    {
        if (strcmp(msi->m_string, key) == 0)
        {
            return msi;
        }
        msi = msi->m_next;
    }
    PRINTF("%s: \"%s\" not found\n", __func__, key);
    return nullptr;
}

void Map_StringToInt_PushBack(Map_StringToInt msi, char *str, int val)
{
    if (msi && str)
    {
        if (msi->m_string && msi->m_val)
        {
            Pair_StringToInt *p = (Pair_StringToInt *)MALLOC(sizeof(Pair_StringToInt));
            Pair_StringToInt *end = msi;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            if (p)
            {
                p->m_string = (char *)MALLOC(strlen(str) + 1);
                p->m_val = (int *)MALLOC(sizeof(int));
                p->m_next = nullptr;
                if (p->m_string && p->m_val)
                {
                    strcpy(p->m_string, str);
                    *p->m_val = val;
                    end->m_next = p;
                    return;
                }
            }
        }
        else if (!msi->m_string && !msi->m_val)
        {
            msi->m_string = (char *)MALLOC(strlen(str) + 1);
            msi->m_val = (int *)MALLOC(sizeof(int));
            if (msi->m_string && msi->m_val)
            {
                strcpy(msi->m_string, str);
                *msi->m_val = val;
                return;
            }
        }
    }
}

Map_StringToInt Map_StringToInt_Erase(Map_StringToInt msi, char *key)
{
    if (msi && key)
    {
        Pair_StringToInt *p = Map_StringToInt_Find(msi, key);
        if (p) // target found
        {
            if (p == msi) // first node
            {
                if (p->m_next) // size > 1
                {
                    msi = p->m_next;
                    FREE(p->m_string);
                    p->m_string = nullptr;
                    FREE(p->m_val);
                    p->m_val = nullptr;
                    FREE(p);
                    p = nullptr;
                    return msi;
                }
                else // size = 1
                {
                    FREE(msi->m_string);
                    msi->m_string = nullptr;
                    FREE(msi->m_val);
                    msi->m_val = nullptr;
                    return msi;
                }
            }
            else
            {
                Pair_StringToInt *former = msi;
                while (former && former->m_next != p)
                {
                    former = former->m_next;
                }
                if (former)
                {
                    former->m_next = p->m_next;
                    FREE(p->m_string);
                    p->m_string = nullptr;
                    FREE(p->m_val);
                    p->m_val = nullptr;
                    p->m_next = nullptr;
                    FREE(p);
                    p = nullptr;
                    return msi;
                }
                else
                {
                    PRINTF("%s: Targer not found.\n", __func__);
                    return nullptr;
                }
            }
        }
        else
        {
            PRINTF("%s: Targer doesn't exsit.\n", __func__);
            return nullptr;
        }
    }
    else
    {
        PRINTF("%s: nullptr detected.\n", __func__);
        return nullptr;
    }
}

bool_t Map_StringToInt_Cmp(Map_StringToInt m1, Map_StringToInt m2)
{
    while (m1 && m2)
    {
        if (m1->m_string && m1->m_val && m2->m_string && m2->m_val)
        {
            if (strcmp(m1->m_string, m2->m_string) == 0 && *(m1->m_val) == *(m2->m_val))
            {
                m1 = m1->m_next;
                m2 = m2->m_next;
            }
            else
            {
                return false_t;
            }
        }
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

Pair_StringToInt *Map_StringToInt_Visit(Map_StringToInt msi, char *str)
{
    if (msi && str)
    {
        while (msi && msi->m_string)
        {
            if (strcmp(str, msi->m_string) == 0)
            {
                return msi;
            }
            msi = msi->m_next;
        }
        PRINTF("%s not found\n", str);
    }
    return nullptr;
}
