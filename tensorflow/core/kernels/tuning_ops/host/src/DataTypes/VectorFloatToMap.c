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

#include "VectorFloatToMap.h"

Vector_FloatToMap Vector_FloatToMap_Ctor()
{
    Pair_FloatToMap *vfm = (Pair_FloatToMap *)MALLOC(sizeof(Pair_FloatToMap));
    if (vfm)
    {
        vfm->m_val = nullptr;
        vfm->m_msf = nullptr;
        vfm->m_next = nullptr;
    }
    else
    {
        PRINTF("%s failed!\n", __func__);
    }
    return vfm;
}

void Vector_FloatToMap_Dtor(Vector_FloatToMap vfm)
{
    while (vfm)
    {
        Pair_FloatToMap *cur = vfm;
        vfm = vfm->m_next;
        FREE(cur->m_val);
        cur->m_val = nullptr;
        Map_StringToFloat_Dtor(cur->m_msf);
        cur->m_msf = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Vector_FloatToMap_Print(Vector_FloatToMap vfm)
{
    PRINTF("[");
    while (vfm)
    {
        PRINTF("(");
        if (vfm->m_val)
        {
            PRINTF("%f: ", *vfm->m_val);
        }
        Map_StringToFloat_Print(vfm->m_msf);
        PRINTF(")");
        if (vfm->m_next)
        {
            PRINTF(", ");
        }
        vfm = vfm->m_next;
    }
    PRINTF("]\n");
}

int Vector_FloatToMap_Size(Vector_FloatToMap vfm)
{
    int n = 0;
    while (vfm && vfm->m_val && vfm->m_msf)
    {
        ++n;
        vfm = vfm->m_next;
    }
    return n;
}

Pair_FloatToMap *Vector_FloatToMap_Visit(Vector_FloatToMap vfm, unsigned int i)
{
    while (vfm && i)
    {
        vfm = vfm->m_next;
        --i;
    }
    if (i)
    {
        PRINTF("%s:out of range!\n", __func__);
        return nullptr;
    }
    else
    {
        return vfm;
    }
}

void Vector_FloatToMap_PushBack(Vector_FloatToMap vfm, float val, Map_StringToFloat msf)
{
    if (vfm && msf)
    {
        if (vfm->m_msf && vfm->m_msf->m_string && vfm->m_msf->m_val && vfm->m_val)
        {
            Pair_FloatToMap *p = (Pair_FloatToMap *)MALLOC(sizeof(Pair_FloatToMap));
            if (p)
            {
                Pair_FloatToMap *end = vfm;
                while (end && end->m_next)
                {
                    end = end->m_next;
                }
                p->m_msf = Map_StringToFloat_Ctor();
                p->m_val = (float *)MALLOC(sizeof(float));
                p->m_next = nullptr;
                if (p->m_msf && p->m_val && end)
                {
                    while (msf)
                    {
                        Map_StringToFloat_PushBack(p->m_msf, msf->m_string, *msf->m_val);
                        msf = msf->m_next;
                    }
                    *(p->m_val) = val;
                    end->m_next = p;
                    return;
                }
            }
        }
        else if (!vfm->m_msf && !vfm->m_val)
        {
            vfm->m_msf = Map_StringToFloat_Ctor();
            vfm->m_val = (float *)MALLOC(sizeof(float));
            if (vfm->m_val)
            {
                *(vfm->m_val) = val;
            }
            if (vfm->m_msf)
            {
                while (msf)
                {
                    Map_StringToFloat_PushBack(vfm->m_msf, msf->m_string, *msf->m_val);
                    msf = msf->m_next;
                }
            }
            return;
        }
    }
}
