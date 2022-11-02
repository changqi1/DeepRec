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

#include "MapStringToFloat.h"

Map_StringToFloat Map_StringToFloat_Ctor()
{
    Pair_StringToFloat *p = (Pair_StringToFloat *)MALLOC(sizeof(Pair_StringToFloat));
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

void Map_StringToFloat_Dtor(Map_StringToFloat msf)
{
    while (msf)
    {
        Pair_StringToFloat *cur = msf;
        msf = msf->m_next;
        FREE(cur->m_string);
        cur->m_string = nullptr;
        FREE(cur->m_val);
        cur->m_val = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Map_StringToFloat_Print(Map_StringToFloat msf)
{
    PRINTF("{");
    while (msf)
    {
        if (msf->m_string && msf->m_val)
        {
            PRINTF("\"%s\": %f", msf->m_string, *msf->m_val);
        }
        if (msf->m_next)
        {
            PRINTF(", ");
        }
        msf = msf->m_next;
    }
    PRINTF("}");
}

Pair_StringToFloat *Map_StringToFloat_Find(Map_StringToFloat msf, char *key)
{
    while (msf && msf->m_string && key)
    {
        if (strcmp(msf->m_string, key) == 0)
        {
            return msf;
        }
        msf = msf->m_next;
    }
    //PRINTF("%s: \"%s\" not found\n", __func__, key);
    return nullptr;
}

void Map_StringToFloat_PushBack(Map_StringToFloat msf, char *str, float val)
{
    if (msf && str)
    {
        if (msf->m_string && msf->m_val)
        {
            Pair_StringToFloat *p = (Pair_StringToFloat *)MALLOC(sizeof(Pair_StringToFloat));
            Pair_StringToFloat *end = msf;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            if (p)
            {
                p->m_string = (char *)MALLOC(strlen(str) + 1);
                p->m_val = (float *)MALLOC(sizeof(float));
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
        else if (!msf->m_string && !msf->m_val)
        {
            msf->m_string = (char *)MALLOC(strlen(str) + 1);
            msf->m_val = (float *)MALLOC(sizeof(float));
            if (msf->m_string && msf->m_val)
            {
                strcpy(msf->m_string, str);
                *msf->m_val = val;
                return;
            }
        }
    }
}

Map_StringToFloat Map_StringToFloat_Erase(Map_StringToFloat msf, char *key)
{
    if (msf && key)
    {
        Pair_StringToFloat *p = Map_StringToFloat_Find(msf, key);
        if (p) // target found
        {
            if (p == msf) // first node
            {
                if (p->m_next) // size > 1
                {
                    msf = p->m_next;
                    FREE(p->m_string);
                    p->m_string = nullptr;
                    FREE(p->m_val);
                    p->m_val = nullptr;
                    FREE(p);
                    p = nullptr;
                    return msf;
                }
                else // size = 1
                {
                    FREE(msf->m_string);
                    msf->m_string = nullptr;
                    FREE(msf->m_val);
                    msf->m_val = nullptr;
                    return msf;
                }
            }
            else
            {
                Pair_StringToFloat *former = msf;
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
                    return msf;
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

bool_t Map_StringToFloat_Cmp(Map_StringToFloat m1, Map_StringToFloat m2)
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

Pair_StringToFloat *Map_StringToFloat_Visit(Map_StringToFloat msf, char *str)
{
    if (msf && str)
    {
        while (msf && msf->m_string)
        {
            if (strcmp(str, msf->m_string) == 0)
            {
                return msf;
            }
            msf = msf->m_next;
        }
        PRINTF("%s not found\n", str);
    }
    return nullptr;
}
