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

#include "Sampling.h"

#ifdef KERNEL_MODULE
extern double pow(double, double);
#endif

void Sampling_MonteCarloSampling(Vector_Vector_Float p, int sample_iter)
{
    int i,j;
    for (i = 0; i < Vector_Vector_Float_Size(p); ++i)
    {
        for (j = 0; j < Vector_Float_Size(Vector_Vector_Float_Visit(p, i)->m_vf); ++j)
        {
            *(Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, j)->m_val) = randomFloat(0.0, 1.0);
        }
    }
}

void Sampling_LatinHypercubeSampling(Vector_Vector_Float p, int sample_iter)
{
    Vector_Int sequence = Vector_Int_Ctor();
    Vector_Int result = Vector_Int_Ctor();
    Vector_Int temp = Vector_Int_Ctor();
    int min_distance = -1;
    int best_distance = -1;
    int distance = 0;
    int pop_size = Vector_Vector_Float_Size(p);
    int var_num;
    int i, iter, row, col, var;
    if (pop_size == 0)
    {
        return;
    }

    var_num = Vector_Float_Size(Vector_Vector_Float_Visit(p, 0)->m_vf);

    for (i = 0; i < pop_size; ++i)
    {
        Vector_Int_PushBack(sequence, i);
    }
    for (iter = 0; iter < sample_iter; ++iter)
    {
        Vector_Int_Clear(temp);
        for (i = 0; i < var_num; ++i)
        {
            Vector_Int sequence_temp;
            Vector_Int_RandomShuffle(sequence);
            sequence_temp = sequence;
            while (sequence_temp)
            {
                Vector_Int_PushBack(temp, *(sequence_temp->m_val));
                sequence_temp = sequence_temp->m_next;
            }
        }
        min_distance = -1;
        for (row = 0; row < Vector_Vector_Float_Size(p); ++row)
        {
            for (col = row + 1; col < Vector_Vector_Float_Size(p); ++col)
            {
                distance = 0;
                for (var = 0; var < var_num; ++var)
                {
                    distance += pow(*(Vector_Int_Visit(temp, row + var * pop_size)->m_val) - *(Vector_Int_Visit(temp, col + var * pop_size)->m_val), 2);
                }
                if (min_distance == 1 || min_distance > distance)
                {
                    min_distance = distance;
                }
            }
        }
        if (best_distance == -1)
        {
            Vector_Int temp1;
            best_distance = min_distance;
            Vector_Int_Dtor(result);
            result = Vector_Int_Ctor();
            temp1 = temp;
            while (temp1)
            {
                Vector_Int_PushBack(result, *(temp1->m_val));
                temp1 = temp1->m_next;
            }
        }
        if (best_distance > min_distance)
        {
            best_distance = min_distance;
            Vector_Int_Assign(result, temp);
        }
    }
    for (i = 0; i < Vector_Vector_Float_Size(p); ++i)
    {
        for (var = 0; var < var_num; ++var)
        {
            *(Vector_Float_Visit(Vector_Vector_Float_Visit(p, i)->m_vf, var)->m_val) = *(Vector_Int_Visit(result, i + var * pop_size)->m_val) * 1.0 / pop_size;
        }
    }
    Vector_Int_Dtor(sequence);
    sequence = nullptr;
    Vector_Int_Dtor(temp);
    temp = nullptr;
    Vector_Int_Dtor(result);
    result = nullptr;
}
