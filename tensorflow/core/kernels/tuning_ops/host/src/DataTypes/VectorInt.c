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

#include "VectorInt.h"

Vector_Int Vector_Int_Ctor()
{
    Node_Int *p = (Node_Int *)MALLOC(sizeof(Node_Int));
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

void Vector_Int_Dtor(Vector_Int vi)
{
    while (vi)
    {
        Node_Int *cur = vi;
        vi = vi->m_next;
        FREE(cur->m_val);
        cur->m_val = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

int Vector_Int_Size(Vector_Int vi)
{
    int size = 0;
    while (vi && vi->m_val)
    {
        ++size;
        vi = vi->m_next;
    }
    return size;
}

void Vector_Int_Print(Vector_Int vi)
{
    PRINTF("[");
    while (vi)
    {
        if (vi->m_val)
        {
            PRINTF("%d", *vi->m_val);
        }
        if (vi->m_next)
        {
            PRINTF(", ");
        }
        vi = vi->m_next;
    }
    PRINTF("]");
}

void Vector_Int_PushBack(Vector_Int vi, int val)
{
    if (vi)
    {
        if (vi->m_val)
        {
            Node_Int *end = vi;
            Node_Int *p;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            p = (Node_Int *)MALLOC(sizeof(Node_Int));
            if (p)
            {
                p->m_val = (int *)MALLOC(sizeof(int));
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
            vi->m_val = (int *)MALLOC(sizeof(int));
            if (vi->m_val)
            {
                *vi->m_val = val;
            }
        }
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
}

Node_Int *Vector_Int_Visit(Vector_Int vi, unsigned int i)
{
    while (vi && i)
    {
        vi = vi->m_next;
        --i;
    }
    if (i)
    {
        PRINTF("%s:out of range!\n", __func__);
        return nullptr;
    }
    else
    {
        return vi;
    }
}

void Vector_Int_RandomShuffle(Vector_Int vi)
{
    int n = Vector_Int_Size(vi), i;
    for (i = n - 1; i > 0; --i)
    {
        int j = RAND_FUNC % (i + 1);
        int temp = *(Vector_Int_Visit(vi, j)->m_val);
        *(Vector_Int_Visit(vi, j)->m_val) = *(Vector_Int_Visit(vi, i)->m_val);
        *(Vector_Int_Visit(vi, i)->m_val) = temp;
    }
}

Node_Int *Vector_Int_Find(Vector_Int vi, int val)
{
    while (vi && vi->m_val)
    {
        if (*(vi->m_val) == val)
        {
            return vi;
        }
        vi = vi->m_next;
    }
    return nullptr;
}

Node_Int *Vector_Int_Erase(Vector_Int vi, Node_Int *p)
{
    if (vi && p)
    {
        if (vi == p) // first node
        {
            if (vi->m_next) // size > 1
            {
                vi = vi->m_next;
                FREE(p->m_val);
                p->m_val = nullptr;
                p->m_next = nullptr;
                FREE(p);
                p = nullptr;
                return vi;
            }
            else // size = 1
            {
                FREE(vi->m_val);
                vi->m_val = nullptr;
                return vi;
            }
        }
        else
        {
            Node_Int *former = vi;
            while (former && former->m_next != p)
            {
                former = former->m_next;
            }
            if (former) // target found
            {
                Node_Int *next = former->m_next->m_next;
                FREE(former->m_next->m_val);
                former->m_next->m_val = nullptr;
                FREE(former->m_next);
                former->m_next = next;
                return vi;
            }
            else
            {
                PRINTF("%s: Target doesn't exsit.\n", __func__);
                return nullptr;
            }
        }
    }
    else
    {
        PRINTF("%s: nullptr detected.\n", __func__);
        return nullptr;
    }
}

void Vector_Int_Clear(Vector_Int vi)
{
    if (vi)
    {
        Vector_Int_Dtor(vi->m_next);
        vi->m_next = nullptr;
        FREE(vi->m_val);
        vi->m_val = nullptr;
    }
}

bool_t Vector_Int_Next_Permutation(Vector_Int vi)
{
    int size = Vector_Int_Size(vi);
    int i = size - 2, j, temp, l, r;
    while (i >= 0 && *Vector_Int_Visit(vi, i)->m_val >= *Vector_Int_Visit(vi, i + 1)->m_val)
    {
        --i;
    }
    if (i < 0)
    {
        return false_t;
    }
    j = size - 1;
    while (*Vector_Int_Visit(vi, j)->m_val <= *Vector_Int_Visit(vi, i)->m_val)
    {
        --j;
    }
    temp = *Vector_Int_Visit(vi, i)->m_val;
    *Vector_Int_Visit(vi, i)->m_val = *Vector_Int_Visit(vi, j)->m_val;
    *Vector_Int_Visit(vi, j)->m_val = temp;
    l = i + 1;
    r = size - 1;
    while (l < r)
    {
        temp = *Vector_Int_Visit(vi, l)->m_val;
        *Vector_Int_Visit(vi, l)->m_val = *Vector_Int_Visit(vi, r)->m_val;
        *Vector_Int_Visit(vi, r)->m_val = temp;
        ++l;
        --r;
    }
    return true_t;
}

void Vector_Int_Resize(Vector_Int vi, int n)
{
    int vi_size = Vector_Int_Size(vi);
    if (n < vi_size)
    {
        Node_Int *target = Vector_Int_Visit(vi, n - 1);
        Vector_Int_Dtor(target->m_next);
        target->m_next = nullptr;
        return;
    }
    else if (n > vi_size)
    {
        int i;
        for (i = 0; i < (n - vi_size); ++i)
        {
            Vector_Int_PushBack(vi, 0.0);
        }
        return;
    }
    else
    {
        return;
    }
}

void Vector_Int_Assign(Vector_Int vi, Vector_Int v)
{
    Node_Int *iter;
    Vector_Int_Dtor(vi->m_next);
    vi->m_next = nullptr;
    FREE(vi->m_val);
    vi->m_val = nullptr;
    iter = v;
    while (iter)
    {
        if (iter->m_val)
        {
            Vector_Int_PushBack(vi, *iter->m_val);
            iter = iter->m_next;
        }
        else
        {
            break;
        }
    }
}