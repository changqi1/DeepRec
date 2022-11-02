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

#include "VectorVectorInt.h"

Vector_Vector_Int Vector_Vector_Int_Ctor()
{
    Node_Vector_Int *p = (Node_Vector_Int *)MALLOC(sizeof(Node_Vector_Int));
    if (p)
    {
        p->m_vi = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Vector_Vector_Int_Dtor(Vector_Vector_Int vvi)
{
    while (vvi)
    {
        Node_Vector_Int *cur = vvi;
        vvi = vvi->m_next;
        Vector_Int_Dtor(cur->m_vi);
        cur->m_vi = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Vector_Vector_Int_Print(Vector_Vector_Int vvi)
{
    PRINTF("[");
    while (vvi)
    {
        if (vvi->m_vi)
        {
            Vector_Int_Print(vvi->m_vi);
        }
        if (vvi->m_next)
        {
            PRINTF(", ");
        }
        vvi = vvi->m_next;
    }
    PRINTF("]\n");
}

int Vector_Vector_Int_Size(Vector_Vector_Int vvi)
{
    int i = 0;
    while (vvi && vvi->m_vi)
    {
        ++i;
        vvi = vvi->m_next;
    }
    return i;
}

void Vector_Vector_Int_PushBack(Vector_Vector_Int vvi, Vector_Int vi)
{
    if (vvi && vi)
    {
        if (vvi->m_vi)
        {
            Node_Vector_Int *p = (Node_Vector_Int *)MALLOC(sizeof(Node_Vector_Int));
            if (p)
            {
                Node_Vector_Int *end = vvi;
                while (end && end->m_next)
                {
                    end = end->m_next;
                }
                p->m_vi = Vector_Int_Ctor();
                p->m_next = nullptr;
                if (p->m_vi && end)
                {
                    while (vi)
                    {
                        Vector_Int_PushBack(p->m_vi, *vi->m_val);
                        vi = vi->m_next;
                    }
                    end->m_next = p;
                    return;
                }
            }
        }
        else
        {
            vvi->m_vi = Vector_Int_Ctor();
            while (vi)
            {
                Vector_Int_PushBack(vvi->m_vi, *vi->m_val);
                vi = vi->m_next;
            }
        }
    }
}

Node_Vector_Int *Vector_Vector_Int_Visit(Vector_Vector_Int vvi, unsigned int i)
{
    while (vvi && i)
    {
        vvi = vvi->m_next;
        --i;
    }
    if (i)
    {
        PRINTF("out of range\n");
        return nullptr;
    }
    else
    {
        return vvi;
    }
}

void Vector_Vector_Int_Resize(Vector_Vector_Int vvi, int n)
{
    int vvi_size = Vector_Vector_Int_Size(vvi);
    if (n < vvi_size)
    {
        Vector_Vector_Int target = Vector_Vector_Int_Visit(vvi, n - 1);
        Vector_Vector_Int_Dtor(target->m_next);
        target->m_next = nullptr;
        return;
    }
    else if (n > vvi_size)
    {
        int i;
        Vector_Int vi = Vector_Int_Ctor();
        Vector_Int_Resize(vi, 1);
        for (i = 0; i < (n - vvi_size); ++i)
        {
            Vector_Vector_Int_PushBack(vvi, vi);
        }
        Vector_Int_Dtor(vi);
        vi = nullptr;
        return;
    }
    else
    {
        return;
    }
}
