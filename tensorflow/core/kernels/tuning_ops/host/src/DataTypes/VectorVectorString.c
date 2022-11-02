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

#include "VectorVectorString.h"

Vector_Vector_String Vector_Vector_String_Ctor()
{
    Node_Vector_String *p = (Node_Vector_String *)MALLOC(sizeof(Node_Vector_String));
    if (p)
    {
        p->m_vs = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Vector_Vector_String_Dtor(Vector_Vector_String vvs)
{
    while (vvs)
    {
        Node_Vector_String *cur = vvs;
        vvs = vvs->m_next;
        Vector_String_Dtor(cur->m_vs);
        cur->m_vs = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Vector_Vector_String_Print(Vector_Vector_String vvs)
{
    PRINTF("[");
    while (vvs)
    {
        if (vvs->m_vs)
        {
            Vector_String_Print(vvs->m_vs);
        }
        if (vvs->m_next)
        {
            PRINTF(", ");
        }
        vvs = vvs->m_next;
    }
    PRINTF("]\n");
}

void Vector_Vector_String_PushBack(Vector_Vector_String vvs, Vector_String vs)
{
    if (vvs && vs)
    {
        if (vvs->m_vs)
        {
            Node_Vector_String *p = (Node_Vector_String *)MALLOC(sizeof(Node_Vector_String));
            if (p)
            {
                Node_Vector_String *end = vvs;
                while (end && end->m_next)
                {
                    end = end->m_next;
                }
                p->m_vs = Vector_String_Ctor();
                p->m_next = nullptr;
                if (p->m_vs && end)
                {
                    while (vs)
                    {
                        Vector_String_PushBack(p->m_vs, vs->m_string);
                        vs = vs->m_next;
                    }
                    end->m_next = p;
                    return;
                }
            }
        }
        else
        {
            vvs->m_vs = Vector_String_Ctor();
            while (vs)
            {
                Vector_String_PushBack(vvs->m_vs, vs->m_string);
                vs = vs->m_next;
            }
        }
    }
}
