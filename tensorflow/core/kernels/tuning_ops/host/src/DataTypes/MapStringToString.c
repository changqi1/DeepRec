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

#include "MapStringToString.h"

Map_StringToString Map_StringToString_Ctor()
{
    Pair_StringToString *p = (Pair_StringToString *)MALLOC(sizeof(Pair_StringToString));
    if (p)
    {
        p->m_key = nullptr;
        p->m_value = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Map_StringToString_Dtor(Map_StringToString mss)
{
    while (mss)
    {
        Pair_StringToString *cur = mss;
        mss = mss->m_next;
        FREE(cur->m_key);
        cur->m_key = nullptr;
        FREE(cur->m_value);
        cur->m_value = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

int Map_StringToString_Size(Map_StringToString mss)
{
    int size = 0;
    while (mss && mss->m_key && mss->m_value)
    {
        ++size;
        mss = mss->m_next;
    }
    return size;
}

void Map_StringToString_Print(Map_StringToString mss)
{
    PRINTF("{");
    while (mss)
    {
        if (mss->m_key && mss->m_value)
        {
            PRINTF("\"%s\": \"%s\"", mss->m_key, mss->m_value);
        }
        if (mss->m_next)
        {
            PRINTF(", ");
        }
        mss = mss->m_next;
    }
    PRINTF("}\n");
}

Pair_StringToString *Map_StringToString_Find(Map_StringToString mss, char *key)
{
    while (mss && mss->m_key && key)
    {
        if (strcmp(mss->m_key, key) == 0)
        {
            return mss;
        }
        mss = mss->m_next;
    }
    return nullptr;
}

Pair_StringToString *Map_StringToString_Visit(Map_StringToString mss, char *key)
{
    while (mss && mss->m_key && key)
    {   
        if (strcmp(mss->m_key, key) == 0)
        {
            return mss;
        }
        mss = mss->m_next;
    }
    PRINTF("%s: \"%s\" not found\n", __func__, key);
    return nullptr;
}

void Map_StringToString_PushBack(Map_StringToString mss, char *key, char *value)
{
    if (mss && key && value)
    {
        if (Map_StringToString_Find(mss, key) == nullptr) // key word not exsiting
        {
            if (mss->m_key && mss->m_value)
            {
                Pair_StringToString *p = (Pair_StringToString *)MALLOC(sizeof(Pair_StringToString));
                Pair_StringToString *end = mss;
                while (end && end->m_next)
                {
                    end = end->m_next;
                }
                if (p)
                {
                    p->m_key = (char *)MALLOC(strlen(key) + 1);
                    p->m_value = (char *)MALLOC(strlen(value) + 1);
                    p->m_next = nullptr;
                    if (p->m_key && p->m_value && end)
                    {
                        strcpy(p->m_key, key);
                        strcpy(p->m_value, value);
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
                mss->m_key = (char *)MALLOC(strlen(key) + 1);
                mss->m_value = (char *)MALLOC(strlen(value)+ 1);
                if (mss->m_key && mss->m_value)
                {
                    strcpy(mss->m_key, key);
                    strcpy(mss->m_value, value);
                    return;
                }
            }
        }
        else // key word exsiting
        {
            FREE(Map_StringToString_Visit(mss, key)->m_value);
            Map_StringToString_Visit(mss, key)->m_value = MALLOC(strlen(value) + 1);
            strcpy(Map_StringToString_Visit(mss, key)->m_value, value);
            //PRINTF("%s:%d mss[%s] has been updated\n", __func__, __LINE__, key);
        }
    }
    else
    {
        PRINTF("%s failed!\n", __func__);
    }
}
