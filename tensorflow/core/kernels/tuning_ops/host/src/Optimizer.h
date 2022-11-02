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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "OptimizerIF.h"

void int_to_string(char* str,int x);
void float_to_string(char* str,float x);

typedef struct ParamOptimizer
{
    ParamOptimizerIF base;

    void (*pca_analysis)(struct ParamOptimizer *self, Map_StringToPtr bestParam);
    void (*append_sample)(struct ParamOptimizer *self, Map_StringToPtr param, Vector_Float fitness);
    void (*update_intern)(struct ParamOptimizer *self);
    void (*update_intern_param)(struct ParamOptimizer *self, Map_StringToString param, Vector_Float result);
    void (*dump_csv)(struct ParamOptimizer *self);
    bool_t (*isInHistory)(struct ParamOptimizer *self, Map_StringToPtr p_param);
    bool_t (*isSame)(struct ParamOptimizer *self, Map_StringToPtr p1, Map_StringToPtr p2);
#if FLOAT_PARAM
    float (*optimize)(struct ParamOptimizer *self, char *key, float min, float max);
    void (*set_value)(struct ParamOptimizer *self, char *key, float value);
#else
    int (*optimize)(struct ParamOptimizer *self, char *key, int min, int max);
    void (*set_value)(struct ParamOptimizer *self, char *key, int value);
#endif

    Map_StringToPtr m_mapParam;
    Suite *suite;
    Vector_Float m_prevTarget;
    bool_t m_initGlog;
    char *m_logname;
    char *m_strvalue;
    int m_windowSize;
    Vector_FloatToMap m_sampleWindow;
#ifndef KERNEL_MODULE
    FILE *m_csvOut;
#endif
    Map_MapToFloat m_mapHist;
} ParamOptimizer;
void ParamOptimizer_Ctor(ParamOptimizer *op, OptParam *p);
void ParamOptimizer_Dtor(ParamOptimizer *self);
void ParamOptimizer_update(ParamOptimizerIF *self);
char* strValue_Ctor();
void strValue_Dtor();
typedef struct OptimizedParamIF
{
    char *(*getName)(struct OptimizedParamIF *self);
    void (*update)(struct OptimizedParamIF *self, ParamOptimizer *p);
    void (*set)(struct OptimizedParamIF *self, char *value, bool_t update, ParamOptimizer *p);
    void *(*cur)(struct OptimizedParamIF *self);
    void *(*max)(struct OptimizedParamIF *self);
    void *(*min)(struct OptimizedParamIF *self);
    float (*dcur)(struct OptimizedParamIF *self);
    float (*dmax)(struct OptimizedParamIF *self);
    float (*dmin)(struct OptimizedParamIF *self);
    struct OptimizedParamIF *(*clone)(struct OptimizedParamIF *self);
    char *(*to_string)(struct OptimizedParamIF *self);
} OptimizedParamIF;
void OptimizedParamIF_Ctor(OptimizedParamIF *self);
void OptimizedParamIF_Dtor(OptimizedParamIF *self);

typedef struct OptimizedParam
{
    OptimizedParamIF base;

    char *m_strName;
    char *m_strValue;
#if FLOAT_PARAM
    float m_curValue;
    float m_maxValue;
    float m_minValue;
    int (*m_funcUpdate)(char *, float);
    float (*m_funcOptimize)(ParamOptimizer *, char *, float, float);
    void (*m_funcSet)(ParamOptimizer *, char *, float);
#else
    int m_curValue;
    int m_maxValue;
    int m_minValue;
    int (*m_funcUpdate)(char *, int);
    int (*m_funcOptimize)(ParamOptimizer *, char *, int, int);
    void (*m_funcSet)(ParamOptimizer *, char *, int);
#endif
} OptimizedParam;
#if FLOAT_PARAM
OptimizedParam *OptimizedParam_Ctor(char *name, float initValue, float maxValue, float minValue,
    int (*update_func)(char *, float), float (*optimize_func)(ParamOptimizer *, char *, float, float), void (*set_func)(ParamOptimizer *, char *, float));
#else
OptimizedParam *OptimizedParam_Ctor(char *name, int initValue, int maxValue, int minValue,
    int (*update_func)(char *, int), int (*optimize_func)(ParamOptimizer *, char *, int, int), void (*set_func)(ParamOptimizer *, char *, int));
#endif
void OptimizedParam_Dtor(OptimizedParam *self);

#endif
