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

#include "Crossover.h"
#include "Optimizer.h"

extern double pow(double, double);

void Crossover_adaptive_simulatedBinaryCrossover(Individual *p1, Individual *p2, float eta)
{
    int len = Map_StringToPtr_Size(p1->m_mapPopParam);
    Map_StringToPtr iter1 = p1->m_mapPopParam;
    Map_StringToPtr iter2 = p2->m_mapPopParam;
    float betaq = 0.0;
    float alpha = 0.0;
    float beta = 0.0;
    float r = 0.0, c1, c2;
    int i = 0;
    for (; i < len; ++i)
    {
        float d_p1 = iter1->m_ptr->dcur(iter1->m_ptr);
        float d_p2 = iter2->m_ptr->dcur(iter2->m_ptr);
        float max = iter1->m_ptr->dmax(iter1->m_ptr);
        float min = iter1->m_ptr->dmin(iter1->m_ptr);
        float y1 = d_p1 > d_p2 ? d_p2 : d_p1;
        float y2 = d_p1 > d_p2 ? d_p1 : d_p2;
        if (fabsf(y1 - y2) < 1e-5)
        {
            continue; // no crossover needed
        }
        r = randomFloat(0.0, 1.0);
        beta = 1.0 + (2.0 * (y1 - min) / (y2 - y1));
        alpha = 2.0 - pow(beta, -(eta + 1));
        if (r <= 1.0 / alpha)
        {
            betaq = pow(alpha * r, 1.0 / (eta + 1));
        }
        else
        {
            betaq = pow(1.0 / (2.0 - r * alpha), 1.0 / (eta + 1)); // delta too small?
        }
        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

        beta = 1.0 + (2.0 * (max - y2) / (y2 - y1));
        alpha = 2.0 - pow(beta, -(eta + 1));
        if (r <= 1.0 / alpha)
        {
            betaq = pow(alpha * r, 1.0 / (eta + 1));
        }
        else
        {
            betaq = pow(1.0 / (2.0 - r * alpha), 1.0 / (eta + 1)); // delta too small?
        }
        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
        if (randomFloat(0.0, 1.0) <= 0.5)
        {
            d_p1 = c1;
            d_p2 = c2;
        }
        else
        {
            d_p1 = c2;
            d_p2 = c1;
        }
        float_to_string(p1->m_valueStr,d_p1);
        iter1->m_ptr->set(iter1->m_ptr, p1->m_valueStr, false_t, nullptr);
        float_to_string(p2->m_valueStr,d_p2);
        iter2->m_ptr->set(iter2->m_ptr, p2->m_valueStr, false_t, nullptr);
        iter1 = iter1->m_next;
        iter2 = iter2->m_next;
    }
    return;
}

void Crossover_simulatedBinaryCrossover(Individual *p1, Individual *p2, float eta)
{
    int len = Map_StringToPtr_Size(p1->m_mapPopParam);
    Map_StringToPtr iter1 = p1->m_mapPopParam;
    Map_StringToPtr iter2 = p2->m_mapPopParam;
    float delta = 0.0;
    float r = 0.0;
    int i = 0;
    for (; i < len; i++)
    {
        float d_p1, d_p2;
        d_p1 = iter1->m_ptr->dcur(iter1->m_ptr);
        d_p2 = iter2->m_ptr->dcur(iter2->m_ptr);
        r = randomFloat(0.0, 1.0);
        if (r <= 0.5)
        {
            delta = pow(2 * r, 1.0 / (eta + 1));
        }
        else
        {
            delta = pow(1.0 / (2 * (1 - r)), 1.0 / (eta + 1)); // delta too small?
        }
        d_p1 = 0.5 * ((1 + delta) * d_p1 + (1 - delta) * d_p2);
        d_p2 = 0.5 * ((1 - delta) * d_p1 + (1 + delta) * d_p2);
        float_to_string(p1->m_valueStr,d_p1);
        iter1->m_ptr->set(iter1->m_ptr, p1->m_valueStr, false_t, nullptr);
        float_to_string(p2->m_valueStr,d_p2);
        iter2->m_ptr->set(iter2->m_ptr, p2->m_valueStr, false_t, nullptr);
        iter1 = iter1->m_next;
        iter2 = iter2->m_next;
    }
    return;
}

void Crossover_pointwiseCrossover(Individual *p1, Individual *p2, float eta)
{
    int len = Map_StringToPtr_Size(p1->m_mapPopParam);
    int pos1 = randomInt(0, len - 1);
    int pos2 = randomInt(0, len - 1);
    if (pos1 > pos2)
    {
        SWAPInt(pos1, pos2);
    }
    if (pos1 != pos2 && !Map_StringToPtr_IsSame(p1->m_mapPopParam, p2->m_mapPopParam))
    {
        OptimizedParamIF *temp;
        Map_StringToPtr iter1 = p1->m_mapPopParam;
        Map_StringToPtr iter2 = p2->m_mapPopParam;
        int idx = 0;
        do
        {
            if (idx >= pos1 && idx <= pos2)
            {
                temp = iter1->m_ptr;
                iter1->m_ptr = iter2->m_ptr;
                iter2->m_ptr = temp;
            }
            idx++;
            iter1 = iter1->m_next;
            iter2 = iter2->m_next;
        } while (iter1 && iter2);
    }
    return;
}
