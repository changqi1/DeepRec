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

#include "MapMapToFloat.h"

Map_MapToFloat Map_MapToFloat_Ctor()
{
    Pair_MapToFloat *mmf = (Pair_MapToFloat *)MALLOC(sizeof(Pair_MapToFloat));
    if (mmf)
    {
        mmf->m_msf = nullptr;
        mmf->m_val = nullptr;
        mmf->m_next = nullptr;
    }
    else
    {
        PRINTF("%s failed!\n", __func__);
    }
    return mmf;
}

void Map_MapToFloat_Dtor(Map_MapToFloat mmf)
{
    while (mmf)
    {
        Pair_MapToFloat *cur = mmf;
        mmf = mmf->m_next;
        Map_StringToFloat_Dtor(cur->m_msf);
        cur->m_msf = nullptr;
        FREE(cur->m_val);
        cur->m_val = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Map_MapToFloat_Print(Map_MapToFloat mmf)
{
    PRINTF("{");
    while (mmf)
    {
        if (mmf->m_msf && mmf->m_val)
        {
            Map_StringToFloat_Print(mmf->m_msf);
            PRINTF(": %f", *mmf->m_val);
        }
        if (mmf->m_next)
        {
            PRINTF(", ");
        }
        mmf = mmf->m_next;
    }
    PRINTF("}\n");
}

Pair_MapToFloat *Map_MapToFloat_Find(Map_MapToFloat mmf, Map_StringToFloat msf)
{
    if (msf)
    {
        while (mmf)
        {
            if (Map_StringToFloat_Cmp(mmf->m_msf, msf))
            {
                return mmf;
            }
            mmf = mmf->m_next;
        }
        return nullptr;
    }
    return nullptr;
}
void Map_MapToFloat_PushBack(Map_MapToFloat mmf, Map_StringToFloat msf, float val)
{
    if (mmf && msf)
    {
        Map_StringToFloat msf_start = msf;
        if (mmf->m_msf && mmf->m_val)
        {
            Pair_MapToFloat *p = (Pair_MapToFloat *)MALLOC(sizeof(Pair_MapToFloat));
            Pair_MapToFloat *end = mmf;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            if (p)
            {
                p->m_msf = Map_StringToFloat_Ctor();
                p->m_val = (float *)MALLOC(sizeof(float));
                p->m_next = nullptr;
                if (p->m_msf && p->m_val)
                {
                    while (msf)
                    {
                        Map_StringToFloat_PushBack(p->m_msf, msf->m_string, *(msf->m_val));
                        msf = msf->m_next;
                    }
                    *(p->m_val) = val;
                    end->m_next = p;
                    Map_StringToFloat_Dtor(msf_start);
                    msf_start= nullptr;
                    msf = nullptr;
                    return;
                }
            }
        }
        else if (!mmf->m_msf && !mmf->m_val)
        {
            mmf->m_msf = Map_StringToFloat_Ctor();
            mmf->m_val = (float *)MALLOC(sizeof(float));
            if (mmf->m_msf && mmf->m_val)
            {
                while (msf)
                {
                    Map_StringToFloat_PushBack(mmf->m_msf, msf->m_string, *(msf->m_val));
                    msf = msf->m_next;
                }
                *(mmf->m_val) = val;
                Map_StringToFloat_Dtor(msf_start);
                msf_start= nullptr;
                msf = nullptr;
                return;
            }
        }
    }
}
