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

#include "VectorFloat.h"

Vector_Float Vector_Float_Ctor()
{
    Node_Float *p = (Node_Float *)MALLOC(sizeof(Node_Float));
    if (p)
    {
        p->m_val = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Vector_Float_Dtor(Vector_Float vf)
{
    while (vf)
    {
        Node_Float *cur = vf;
        vf = vf->m_next;
        FREE(cur->m_val);
        cur->m_val = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

int Vector_Float_Size(Vector_Float vf)
{
    int size = 0;
    while (vf && vf->m_val)
    {
        ++size;
        vf = vf->m_next;
    }
    return size;
}

void Vector_Float_Print(Vector_Float vf)
{
    PRINTF("[");
    while (vf)
    {
        if (vf->m_val)
        {
            PRINTF("%f", *vf->m_val);
        }
        if (vf->m_next)
        {
            PRINTF(", ");
        }
        vf = vf->m_next;
    }
    PRINTF("]");
}

void Vector_Float_PushBack(Vector_Float vf, float val)
{
    if (vf)
    {
        if (vf->m_val)
        {
            Node_Float *end = vf;
            Node_Float *p;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            p = (Node_Float *)MALLOC(sizeof(Node_Float));
            if (p)
            {
                p->m_val = (float *)MALLOC(sizeof(float));
                if (p->m_val)
                {
                    *p->m_val = val;
                }
                p->m_next = nullptr;
                end->m_next = p;
                return;
            }
        }
        else
        {
            vf->m_val = (float *)MALLOC(sizeof(float));
            if (vf->m_val)
            {
                *vf->m_val = val;
            }
        }
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
}

Node_Float *Vector_Float_Visit(Vector_Float vf, unsigned int i)
{
    while (vf && i)
    {
        vf = vf->m_next;
        --i;
    }
    if (i)
    {
        PRINTF("%s:out of range!\n", __func__);
        return nullptr;
    }
    else
    {
        return vf;
    }
}

void Vector_Float_Resize(Vector_Float vf, int n)
{
    int vf_size = Vector_Float_Size(vf);
    if (n < vf_size)
    {
        Node_Float *target = Vector_Float_Visit(vf, n - 1);
        Vector_Float_Dtor(target->m_next);
        target->m_next = nullptr;
        return;
    }
    else if (n > vf_size)
    {
        int i;
        for (i = 0; i < (n - vf_size); ++i)
        {
            Vector_Float_PushBack(vf, 0.0);
        }
        return;
    }
    else
    {
        return;
    }
}

void Vector_Float_Assign(Vector_Float vf, Vector_Float v)
{
    Node_Float *iter;
    Vector_Float_Dtor(vf->m_next);
    vf->m_next = nullptr;
    FREE(vf->m_val);
    vf->m_val = nullptr;
    iter = v;
    while (iter)
    {
        if (iter->m_val)
        {
            Vector_Float_PushBack(vf, *iter->m_val);
            iter = iter->m_next;
        }
        else
        {
            break;
        }
    }
}

void Vector_Float_Clear(Vector_Float vf)
{
    if (vf)
    {
        Vector_Float_Dtor(vf->m_next);
        vf->m_next = nullptr;
        FREE(vf->m_val);
        vf->m_val = nullptr;
    }
}