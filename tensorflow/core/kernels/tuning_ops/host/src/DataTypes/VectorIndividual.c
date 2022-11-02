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

#include "VectorIndividual.h"

Vector_Individual Vector_Individual_Ctor()
{
    Node_Individual *self = (Node_Individual *)MALLOC(sizeof(Node_Individual));
    self->m_indi = nullptr;
    self->m_next = nullptr;
    return self;
}

void Vector_Individual_Dtor(Vector_Individual vi)
{
    while (vi)
    {
        Vector_Individual cur = vi;
        vi = vi->m_next;
        Individual_Dtor(cur->m_indi);
        cur->m_indi = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

Node_Individual *Vector_Individual_Visit(Vector_Individual vi, unsigned int n)
{
    Vector_Individual iter = vi;
    while (iter && n)
    {
        iter = iter->m_next;
        --n;
    }
    if (n)
    {
        PRINTF("%s:out of range!\n", __func__);
        return nullptr;
    }
    else
    {
        return iter;
    }
}

int Vector_Individual_Size(Vector_Individual vi)
{
    int n = 0;
    while (vi && vi->m_indi)
    {
        ++n;
        vi = vi->m_next;
    }
    return n;
}

void Vector_Individual_RandomShuffle(Vector_Individual vi)
{
    int n = Vector_Individual_Size(vi), i;
    for (i = n - 1; i > 0; --i)
    {
        int j = RAND_FUNC % (i + 1);
        Individual *temp = Vector_Individual_Visit(vi, j)->m_indi;
        Vector_Individual_Visit(vi, j)->m_indi = Vector_Individual_Visit(vi, i)->m_indi;
        Vector_Individual_Visit(vi, i)->m_indi = temp;
    }
    return;
}

void Vector_Individual_PushBack(Vector_Individual vi, Individual *i)
{
    if (vi)
    {
        if (vi->m_indi)
        {
            Node_Individual *end = vi;
            Node_Individual *p;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            p = (Node_Individual *)MALLOC(sizeof(Node_Individual));
            if (p)
            {
                p->m_indi = Individual_Copy_Ctor(i);
                p->m_next = nullptr;
                end->m_next = p;
                return;
            }
        }
        else
        {
            vi->m_indi = Individual_Copy_Ctor(i);
        }
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
}

void Vector_Individual_Resize(Vector_Individual vi, int n)
{
    int vi_size = Vector_Individual_Size(vi), i;
    if (n < vi_size)
    {
        Node_Individual *target = Vector_Individual_Visit(vi, n - 1);
        Vector_Individual_Dtor(target->m_next);
        target->m_next = nullptr;
        return;
    }
    else if (n > vi_size)
    {
        for (i = 0; i < (n - vi_size); ++i)
        {   Individual *tmp_individual = Individual_Ctor();
            Vector_Individual_PushBack(vi, tmp_individual);
            Individual_Dtor(tmp_individual);
            tmp_individual = nullptr;
        }
        return;
    }
    else
    {
        return;
    }
}

void Vector_Individual_Erase(Vector_Individual* p_vi, Individual *indi)
{
    if (NULL == p_vi) {
        return;
    }

    if (*p_vi)
    {
        //Find the node to be erased
        Node_Individual *node = *p_vi;
        Node_Individual *prev = nullptr;
        while (node && node->m_indi != indi) {
            prev = node;
            node = node->m_next;
        }

        if (node == *p_vi) { //erase head of the list
            if (nullptr == node->m_next) {
                //vector only contains one node, do not free node
                Individual_Dtor(node->m_indi);
                node->m_indi = nullptr;
                return;
            }
            *p_vi = node->m_next;
        }
        else {
            prev->m_next = node->m_next;
        }

        Individual_Dtor(node->m_indi);
        node->m_indi = nullptr;
        node->m_next = nullptr;
        FREE(node);
        node = nullptr;
    }
    else
    {
        PRINTF("%s failed!\n", __func__);
    }
}

void Vector_Individual_Print(Vector_Individual vi)
{
    Node_Individual *iter = vi;
    int i = 0;
    PRINTF("[\n");
    while (iter)
    {
        PRINTF("Individual[%d] = ", i++);
        Individual_Print(iter->m_indi);
        PRINTF("\n");
        iter = iter->m_next;
    }
    PRINTF("]\n");
}
