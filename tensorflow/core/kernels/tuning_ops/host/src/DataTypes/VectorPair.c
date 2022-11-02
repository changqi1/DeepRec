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

#include "VectorPair.h"

Vector_Pair_StringToInt Vector_Pair_StringToInt_Ctor()
{
    Node_Pair_StringToInt *self = (Node_Pair_StringToInt *)MALLOC(sizeof(Node_Pair_StringToInt));
    self->m_next = nullptr;
    self->m_pair = nullptr;
    return self;
}
void Vector_Pair_StringToInt_Dtor(Vector_Pair_StringToInt vpsi)
{
    while (vpsi)
    {
        Vector_Pair_StringToInt cur = vpsi;
        vpsi = vpsi->m_next;
        if (cur->m_pair)
        {
            FREE(cur->m_pair->m_string);
            cur->m_pair->m_string = nullptr;
        }
        FREE(cur->m_pair);
        cur->m_pair = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
    return;
}

int Vector_Pair_StringToInt_Size(Vector_Pair_StringToInt vpsi)
{
    int n = 0;
    while (vpsi && vpsi->m_pair)
    {
        ++n;
        vpsi = vpsi->m_next;
    }
    return n;
}

void Vector_Pair_StringToInt_PushBack(Vector_Pair_StringToInt vpsi, ps2i *psi)
{
    if (vpsi && psi->m_string)
    {
        if (vpsi->m_pair == nullptr) // first node, empty container
        {
            vpsi->m_pair = (ps2i *)MALLOC(sizeof(ps2i));
            vpsi->m_pair->m_string = (char *)MALLOC(strlen(psi->m_string) + 1);
            strcpy(vpsi->m_pair->m_string, psi->m_string);
            vpsi->m_pair->m_val = psi->m_val;
            return;
        }
        else
        {
            Vector_Pair_StringToInt end = vpsi;
            Node_Pair_StringToInt *temp;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            temp = (Node_Pair_StringToInt *)MALLOC(sizeof(Node_Pair_StringToInt));
            temp->m_pair = (ps2i *)MALLOC(sizeof(ps2i));
            temp->m_pair->m_string = (char *)MALLOC(strlen(psi->m_string) + 1);
            strcpy(temp->m_pair->m_string, psi->m_string);
            temp->m_pair->m_val = psi->m_val;
            temp->m_next = nullptr;
            end->m_next = temp;
            return;
        }
    }
    else
    {
        PRINTF("%s:%d failed.", __func__, __LINE__);
    }
}

void Vector_Pair_StringToInt_PushBack_param(Vector_Pair_StringToInt vpsi, char *key, int value)
{
    if (vpsi && key)
    {
        if (vpsi->m_pair == nullptr) // first node, empty container
        {
            vpsi->m_pair = (ps2i *)MALLOC(sizeof(ps2i));
            vpsi->m_pair->m_string = (char *)MALLOC(strlen(key) + 1);
            strcpy(vpsi->m_pair->m_string, key);
            vpsi->m_pair->m_val = value;
            return;
        }
        else
        {
            Node_Pair_StringToInt *temp;
            Vector_Pair_StringToInt end = vpsi;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            temp = (Node_Pair_StringToInt *)MALLOC(sizeof(Node_Pair_StringToInt));
            temp->m_pair = (ps2i *)MALLOC(sizeof(ps2i));
            temp->m_pair->m_string = (char *)MALLOC(strlen(key) + 1);
            strcpy(temp->m_pair->m_string, key);
            temp->m_pair->m_val = value;
            temp->m_next = nullptr;
            end->m_next = temp;
            return;
        }
    }
    else
    {
        PRINTF("%s:%d failed.\n", __func__, __LINE__);
    }
}

Vector_Pair_StringToInt Vector_Pair_StringToInt_Erase(Vector_Pair_StringToInt vpsi, Node_Pair_StringToInt *p)
{
    if (vpsi && p)
    {
        if (vpsi == p) // first node
        {
            if (vpsi->m_next) // size > 1
            {
                vpsi = vpsi->m_next;
                FREE(p->m_pair->m_string);
                p->m_pair->m_string = nullptr;
                FREE(p->m_pair);
                p->m_pair = nullptr;
                p->m_next = nullptr;
                FREE(p);
                p = nullptr;
                return vpsi;
            }
            else // size = 1
            {
                FREE(vpsi->m_pair->m_string);
                vpsi->m_pair->m_string = nullptr;
                FREE(vpsi->m_pair);
                vpsi->m_pair = nullptr;
                return vpsi;
            }
        }
        else
        {
            Node_Pair_StringToInt *former = vpsi;
            while (former && former->m_next != p)
            {
                former = former->m_next;
            }
            if (former) // target found
            {
                Node_Pair_StringToInt *next = former->m_next->m_next;
                FREE(former->m_next->m_pair->m_string);
                former->m_next->m_pair->m_string = nullptr;
                FREE(former->m_next->m_pair);
                former->m_next->m_pair = nullptr;
                FREE(former->m_next);
                former->m_next = next;
                return vpsi;
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

void Vector_Pair_StringToInt_Print(Vector_Pair_StringToInt vpsi)
{
    PRINTF("[");
    while (vpsi && vpsi->m_pair)
    {
        PRINTF("%s: %d", vpsi->m_pair->m_string, vpsi->m_pair->m_val);
        if (vpsi->m_next)
        {
            PRINTF(", ");
        }
        vpsi = vpsi->m_next;
    }
    PRINTF("]\n");
}
