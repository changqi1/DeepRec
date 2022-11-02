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

#include "VectorMap.h"

Vector_Map_StringToPtr Vector_Map_StringToPtr_Ctor()
{
    Node_Map_StringToPtr *p = (Node_Map_StringToPtr *)MALLOC(sizeof(Node_Map_StringToPtr));
    if (p)
    {
        p->m_msp = nullptr;
        p->m_next = nullptr;
    }
    else
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
    }
    return p;
}

void Vector_Map_StringToPtr_Dtor(Vector_Map_StringToPtr vmsp)
{
    while (vmsp)
    {
        Node_Map_StringToPtr *cur = vmsp;
        vmsp = vmsp->m_next;
        Map_StringToPtr_Dtor(cur->m_msp);
        cur->m_msp = nullptr;
        cur->m_next = nullptr;
        FREE(cur);
        cur = nullptr;
    }
}

void Vector_Map_StringToPtr_PushBack(Vector_Map_StringToPtr vmsp, Map_StringToPtr msp)
{
    if (vmsp)
    {
        if (vmsp->m_msp)
        {
            Node_Map_StringToPtr *p = (Node_Map_StringToPtr *)MALLOC(sizeof(Node_Map_StringToPtr));
            Node_Map_StringToPtr *end = vmsp;
            while (end && end->m_next)
            {
                end = end->m_next;
            }
            if (p)
            {
                p->m_msp = Map_StringToPtr_Ctor();
                p->m_next = nullptr;
                if (p->m_msp)
                {
                    if (msp)
                    {
                        Map_StringToPtr_PushBack(p->m_msp, msp->m_string, msp->m_ptr);
                    }
                    else
                    {
                        Map_StringToPtr_PushBack(p->m_msp, "", nullptr);
                    }
                    end->m_next = p;
                }
            }
            else
            {
                PRINTF("%s:%d failed!\n", __func__, __LINE__);
            }
        }
        else
        {
            vmsp->m_msp = Map_StringToPtr_Ctor();
            vmsp->m_next = nullptr;
            if (vmsp->m_msp)
            {
                if (msp)
                {
                    Map_StringToPtr_PushBack(vmsp->m_msp, msp->m_string, msp->m_ptr);
                }
                else
                {
                    Map_StringToPtr_PushBack(vmsp->m_msp, "", nullptr);
                }
                return;
            }
        }
    }
    else
    {
        PRINTF("%s failed!\n", __func__);
    }
}

int Vector_Map_StringToPtr_Size(Vector_Map_StringToPtr vmsp)
{
    int n = 0;
    while (vmsp && vmsp->m_msp)
    {
        ++n;
        vmsp = vmsp->m_next;
    }
    return n;
}

Node_Map_StringToPtr *Vector_Map_StringToPtr_Visit(Vector_Map_StringToPtr vmsp, unsigned int i)
{
    while (vmsp && i)
    {
        vmsp = vmsp->m_next;
        --i;
    }
    if (i)
    {
        PRINTF("%s:out of range!\n", __func__);
        return nullptr;
    }
    else
    {
        return vmsp;
    }
}

void Vector_Map_StringToPtr_Resize(Vector_Map_StringToPtr vmsp, int n)
{
    int vmsp_size = Vector_Map_StringToPtr_Size(vmsp);
    if (n < vmsp_size)
    {
        Node_Map_StringToPtr *target = Vector_Map_StringToPtr_Visit(vmsp, n - 1);
        Vector_Map_StringToPtr_Dtor(target->m_next);
        target->m_next = nullptr;
        return;
    }
    else if (n > vmsp_size)
    {
        int i;
        for (i = 0; i < (n - vmsp_size); ++i)
        {
            Vector_Map_StringToPtr_PushBack(vmsp, vmsp->m_msp);
        }
        return;
    }
    else
    {
        return;
    }
}

void Vector_Map_StringToPtr_Erase(Vector_Map_StringToPtr* p_vmsp, Map_StringToPtr msp)
{
    if (nullptr == p_vmsp) {
        return;
    }

    if (*p_vmsp)
    {
        //Find the node to be erased
        Node_Map_StringToPtr *node = *p_vmsp;
        Node_Map_StringToPtr *prev = nullptr;
        while (node && node->m_msp != msp) {
            prev = node;
            node = node->m_next;
        }

        if (node == *p_vmsp) { //erase head of the list
            if (nullptr == node->m_next) {
                //vector only contains one node, do not free node
                Map_StringToPtr_Dtor(node->m_msp);
                node->m_msp = nullptr;
                return;
            }
            *p_vmsp = node->m_next;
        }
        else {
            prev->m_next = node->m_next;
        }

        Map_StringToPtr_Dtor(node->m_msp);
        node->m_msp = nullptr;
        node->m_next = nullptr;
        FREE(node);
        node = nullptr;
    }
    else
    {
        PRINTF("%s failed!\n", __func__);
    }
}

void Vector_Map_StringToPtr_Print(Vector_Map_StringToPtr vmsp)
{
    Node_Map_StringToPtr * node = vmsp;
    int i = 0;
    while (node) {
        PRINTF("vmsp[%d] = ", i);
        Map_StringToPtr_Print(node->m_msp);
        node = node->m_next;
        ++i;
    }
}
