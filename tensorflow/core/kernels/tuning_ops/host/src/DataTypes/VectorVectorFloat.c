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

#include "VectorVectorFloat.h"

Vector_Vector_Float Vector_Vector_Float_Ctor()
{
    Node_Vector_Float *p = (Node_Vector_Float *)MALLOC(sizeof(Node_Vector_Float));
    if (p)
    {
        p->m_vf = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Vector_Vector_Float_Dtor(Vector_Vector_Float vvf)
{
    while (vvf)
    {
        Node_Vector_Float *cur = vvf;
        vvf = vvf->m_next;
        Vector_Float_Dtor(cur->m_vf);
        cur->m_vf = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Vector_Vector_Float_Print(Vector_Vector_Float vvf)
{
    PRINTF("[");
    while (vvf)
    {
        if (vvf->m_vf)
        {
            Vector_Float_Print(vvf->m_vf);
        }
        if (vvf->m_next)
        {
            PRINTF(", ");
        }
        vvf = vvf->m_next;
    }
    PRINTF("]\n");
}

int Vector_Vector_Float_Size(Vector_Vector_Float vvf)
{
    int i = 0;
    while (vvf && vvf->m_vf)
    {
        ++i;
        vvf = vvf->m_next;
    }
    return i;
}

void Vector_Vector_Float_PushBack(Vector_Vector_Float vvf, Vector_Float vf)
{
    if (vvf && vf)
    {
        if (vvf->m_vf)
        {
            Node_Vector_Float *p = (Node_Vector_Float *)MALLOC(sizeof(Node_Vector_Float));
            if (p)
            {
                Node_Vector_Float *end = vvf;
                while (end && end->m_next)
                {
                    end = end->m_next;
                }
                p->m_vf = Vector_Float_Ctor();
                p->m_next = nullptr;
                if (p->m_vf && end)
                {
                    while (vf)
                    {
                        Vector_Float_PushBack(p->m_vf, *vf->m_val);
                        vf = vf->m_next;
                    }
                    end->m_next = p;
                    return;
                }
            }
        }
        else
        {
            vvf->m_vf = Vector_Float_Ctor();
            while (vf)
            {
                Vector_Float_PushBack(vvf->m_vf, *vf->m_val);
                vf = vf->m_next;
            }
        }
    }
}

Node_Vector_Float *Vector_Vector_Float_Visit(Vector_Vector_Float vvf, unsigned int i)
{
    while (vvf && i)
    {
        vvf = vvf->m_next;
        --i;
    }
    if (i)
    {
        PRINTF("out of range\n");
        return nullptr;
    }
    else
    {
        return vvf;
    }
}

void Vector_Vector_Float_Resize(Vector_Vector_Float vvf, int n)
{
    int vvf_size = Vector_Vector_Float_Size(vvf);
    if (n < vvf_size)
    {
        Vector_Vector_Float target = Vector_Vector_Float_Visit(vvf, n - 1);
        Vector_Vector_Float_Dtor(target->m_next);
        target->m_next = nullptr;
        return;
    }
    else if (n > vvf_size)
    {
        int i;
        Vector_Float vf = Vector_Float_Ctor();
        Vector_Float_Resize(vf, 1);
        for (i = 0; i < (n - vvf_size); ++i)
        {
            Vector_Vector_Float_PushBack(vvf, vf);
        }
        Vector_Float_Dtor(vf);
        vf = nullptr;
        return;
    }
    else
    {
        return;
    }
}
