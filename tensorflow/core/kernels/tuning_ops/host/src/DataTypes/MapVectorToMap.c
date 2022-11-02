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

#include "MapVectorToMap.h"

Map_VectorToMap Map_VectorToMap_Ctor()
{
    Pair_VectorToMap *mvm = (Pair_VectorToMap *)MALLOC(sizeof(Pair_VectorToMap));
    if (mvm)
    {
        mvm->m_vs = nullptr;
        mvm->m_mss = nullptr;
        mvm->m_next = nullptr;
    }
    else
    {
        PRINTF("%s failed!\n", __func__);
    }
    return mvm;
}

void Map_VectorToMap_Dtor(Map_VectorToMap mvm)
{
    while (mvm)
    {
        Pair_VectorToMap *cur = mvm;
        mvm = mvm->m_next;
        Vector_String_Dtor(cur->m_vs);
        cur->m_vs = nullptr;
        Map_StringToString_Dtor(cur->m_mss);
        cur->m_mss = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

int Map_VectorToMap_Size(Map_VectorToMap mvm)
{
    int size = 0;
    while (mvm && mvm->m_vs && mvm->m_mss)
    {
        ++size;
        mvm = mvm->m_next;
    }
    return size;
}

void Map_VectorToMap_Print(Map_VectorToMap mvm)
{
    PRINTF("{");
    while (mvm)
    {
        if (mvm->m_vs && mvm->m_mss)
        {
            Vector_String_Print(mvm->m_vs);
            PRINTF(": ");
            Map_StringToString_Print(mvm->m_mss);
        }
        if (mvm->m_next)
        {
            PRINTF(", ");
        }
        mvm = mvm->m_next;
    }
    PRINTF("}\n");
}

Pair_VectorToMap *Map_VectorToMap_Find(Map_VectorToMap mvm, Vector_String vs)
{
    while (vs && mvm)
    {
        if (Vector_String_Cmp(mvm->m_vs, vs))
        {
            return mvm;
        }
        mvm = mvm->m_next;
    }
    PRINTF("%s: not found\n", __func__);
    return mvm;
}

void Map_VectorToMap_PushBack(Map_VectorToMap mvm, Vector_String vs, Map_StringToString mss)
{
    if (mvm && vs && mss)
    {
        if (mvm->m_mss && mvm->m_mss->m_key && mvm->m_mss->m_value && mvm->m_vs && mvm->m_vs->m_string)
        {
            Pair_VectorToMap *p = (Pair_VectorToMap *)MALLOC(sizeof(Pair_VectorToMap));
            if (p)
            {
                Pair_VectorToMap *end = mvm;
                while (end && end->m_next)
                {
                    end = end->m_next;
                }
                p->m_vs = Vector_String_Ctor();
                p->m_mss = Map_StringToString_Ctor();
                p->m_next = nullptr;
                if (p->m_vs && p->m_mss && end)
                {
                    while (vs)
                    {
                        Vector_String_PushBack(p->m_vs, vs->m_string);
                        vs = vs->m_next;
                    }
                    while (mss)
                    {
                        Map_StringToString_PushBack(p->m_mss, mss->m_key, mss->m_value);
                        mss = mss->m_next;
                    }
                    end->m_next = p;
                    return;
                }
            }
        }
        else if (!mvm->m_vs && !mvm->m_mss)
        {
            mvm->m_vs = Vector_String_Ctor();
            mvm->m_mss = Map_StringToString_Ctor();
            if (mvm->m_vs)
            {
                while (vs)
                {
                    Vector_String_PushBack(mvm->m_vs, vs->m_string);
                    vs = vs->m_next;
                }
            }
            if (mvm->m_mss)
            {
                while (mss)
                {
                    Map_StringToString_PushBack(mvm->m_mss, mss->m_key, mss->m_value);
                    mss = mss->m_next;
                }
            }
            return;
        }
    }
}
