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

#include "Suite.h"

#define PI 3.14159265358979323846

Vector_Float Suite_target_func(Suite *self)
{
    return self->fitness;
}

bool_t Suite_repair(struct Suite *self, float *x) {
    return false_t;
}

void Suite_Ctor(Suite *self) {
    if (nullptr == self) {
        return;
    }
    self->name = SUITE_NOT_SPECIFIED;
    self->var = Vector_String_Ctor();
    self->var_min = Vector_Float_Ctor();
    self->var_max = Vector_Float_Ctor();
    self->fitness = Vector_Float_Ctor();
    self->target_func = Suite_target_func;
    self->get_var = nullptr;
    self->evaluate = nullptr;
    self->repair = Suite_repair;
    self->parse = nullptr;
}

void Suite_Dtor(Suite *self) {
    if (nullptr == self) {
        return;
    }
    self->name = SUITE_NOT_SPECIFIED;
    Vector_String_Dtor(self->var);
    self->var = nullptr;
    Vector_Float_Dtor(self->var_min);
    self->var_min = nullptr;
    Vector_Float_Dtor(self->var_max);
    self->var_max = nullptr;
    Vector_Float_Dtor(self->fitness);
    self->fitness = nullptr;
    self->get_var = nullptr;
    self->evaluate = nullptr;
    self->repair = nullptr;
    self->parse = nullptr;
}
