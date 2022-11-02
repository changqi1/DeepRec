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

#include "VectorString.h"

Vector_String Vector_String_Ctor()
{
    Node_String *p = (Node_String *)MALLOC(sizeof(Node_String));
    if (p)
    {
        p->m_string = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Vector_String_Dtor(Vector_String vs)
{
    while (vs)
    {
        Node_String *cur = vs;
        vs = vs->m_next;
        FREE(cur->m_string);
        cur->m_string = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

int Vector_String_Size(Vector_String vs)
{
    int size = 0;
    while (vs && vs->m_string)
    {
        ++size;
        vs = vs->m_next;
    }
    return size;
}

void Vector_String_Print(Vector_String vs)
{
    PRINTF("[");
    while (vs)
    {
        if (vs->m_string)
        {
            PRINTF("\"%s\"", vs->m_string);
        }
        if (vs->m_next)
        {
            PRINTF(", ");
        }
        vs = vs->m_next;
    }
    PRINTF("]\n");
}

void Vector_String_PushBack(Vector_String vs, char *str)
{
    if (vs && str)
    {
        if (vs->m_string)
        {
            Node_String *p;
            Node_String *end = vs;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            p = (Node_String *)MALLOC(sizeof(Node_String));
            if (p)
            {
                p->m_string = (char *)MALLOC(strlen(str) + 1);
                p->m_next = nullptr;
                if (p->m_string && end)
                {
                    strcpy(p->m_string, str);
                    end->m_next = p;
                    return;
                }
            }
        }
        else
        {
            vs->m_string = (char *)MALLOC(strlen(str) + 1);
            if (vs->m_string)
            {
                strcpy(vs->m_string, str);
                return;
            }
        }
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
}

bool_t Vector_String_Cmp(Vector_String vs1, Vector_String vs2)
{
    while (vs1 && vs2)
    {
        if (vs1->m_string && vs2->m_string && strcmp(vs1->m_string, vs2->m_string) == 0)
        {
            vs1 = vs1->m_next;
            vs2 = vs2->m_next;
        }
        else
        {
            return false_t;
        }
    }
    if (!vs1 & !vs2)
    {
        return true_t;
    }
    else
    {
        return false_t;
    }
}

Node_String *Vector_String_Visit(Vector_String vs, unsigned int i)
{
    while (vs && i)
    {
        vs = vs->m_next;
        --i;
    }
    if (i)
    {
        PRINTF("out of range\n");
        return nullptr;
    }
    else
    {
        return vs;
    }
}
