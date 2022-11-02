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

#include "Mutate.h"
#include "Optimizer.h"

extern double pow(double, double);
//0.1 5.0
void Mutate_adaptive_polynomialMutate(Individual *p, float mutp, float eta)
{
    //float betaq = 0.0;
    float beta, alpha, d, y, max, min;
    float r = 0.0;
    Map_StringToPtr iter = p->m_mapPopParam;
    while (iter)
    {
        if (randomFloat(0.0, 1.0) >= mutp)//0-1 间随机数>= mutp，跳过当前参数 ，不进行参数漂移，转到下一个参数
        {
            iter = iter->m_next;
            continue;
        }
        y = iter->m_ptr->dcur(iter->m_ptr);
        max = iter->m_ptr->dmax(iter->m_ptr);
        min = iter->m_ptr->dmin(iter->m_ptr);
        r = randomFloat(0.0, 1.0);
        if (r <= 0.5)
        {
            beta = 1.0 - (y - min) / (max - min);
            alpha = 2 * r + (1.0 - 2.0 * r) * pow(beta, eta + 1);
            d = pow(alpha, 1.0 / (eta + 1)) - 1;
        }
        else
        {
            beta = 1.0 - (max - y) / (max - min);
            alpha = 2 * (1 - r) + (2.0 * r - 1.0) * pow(beta, eta + 1);
            d = 1.0 - pow(alpha, 1.0 / (eta + 1));
        }
        y = y + d * ((max - min) > 0 ? max - min : 1);
        float_to_string(p->m_valueStr,y);
        iter->m_ptr->set(iter->m_ptr, p->m_valueStr, false_t, nullptr);
        iter = iter->m_next;
    }
}

void Mutate_polynomialMutate(Individual *p, float mutp, float eta)
{
    float delta = 0.0, d_b, range;
    float r = 0.0;
    Map_StringToPtr iter = p->m_mapPopParam;
    while (iter)
    {
        if (randomFloat(0.0, 1.0) >= mutp)
            continue;
        d_b = iter->m_ptr->dcur(iter->m_ptr);
        range = iter->m_ptr->dmax(iter->m_ptr) - iter->m_ptr->dmin(iter->m_ptr);
        r = randomFloat(0.0, 1.0);
        if (r <= 0.5)
        {
            delta = pow(2 * r, 1.0 / (eta + 1)) - 1;
        }
        else
        {
            delta = 1.0 - pow(2 * (1 - r), 1.0 / (eta + 1)); // delta too small?
        }
        d_b = d_b + delta * (range > 0 ? range : 1);
        float_to_string(p->m_valueStr,d_b);
        iter->m_ptr->set(iter->m_ptr, p->m_valueStr, false_t, nullptr);
        iter = iter->m_next;
    }
}
