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

#include "Optimizer.h"
#include "PSOOptimizer.h"
#include "GAOptimizer.h"
#include "DEOptimizer.h"
#include "BayesianOptimizer.h"
#ifdef KERNEL_MODULE
extern double atof(const char *);
#endif

#define HELP_INFO \
"\
usage: pta_launcher [options]\n\
  basic user options for pta_launcher, with defaults in [] when omitted:\n\
    -help                 show this message\n\
    -algo PSO|GA|DE|BO    specify the tuning algo [PSO]\n\
    -suite Ackley|NetworkAI|ExpSinFunc    specify test suite [Ackley]\n\
                          if NetworkAI is specified, user has to further specify -exe <exe_path> -xml <xml_path>\n\
    -gen <number>         specify test generation [100]\n\
    -pop <number>         specify test pop size [60]\n\
  examples:\n\
    ./pta_launcher\n\
    ./pta_launcher -algo BO -suite ExpSinFunc -pop 100 -gen 10\n\
    ./pta_launcher -suite NetworkAI -exe ./networkai_related/Release/benchmark_app -xml ./networkai_related/pan.xml\n\
\n"

void int_to_string(char* str,int x)
{
    sprintf(str, "%d", x);
}

void float_to_string(char* str,float x)
{
    sprintf(str, "%f", x);
}

// member func of ParamOptimizerIF

ParamOptimizerIF *getParamOptimizer(OptParam *p)
{
    switch (p->algorithm)
    {
    case PSO:
        return (ParamOptimizerIF *)PSOOptimizer_Ctor(p);
    case GA:
        return (ParamOptimizerIF *)GAOptimizer_Ctor(p);
    case DE:
        return (ParamOptimizerIF *)DEOptimizer_Ctor(p);
    case BO:
        return (ParamOptimizerIF *)BayesianOptimizer_Ctor(p);
    default:
        PRINTF("No support for optimizer by algorithm:%d\n", p->algorithm);
        break;
    }
    return nullptr;
}

void ParamOptimizerIF_Dtor(ParamOptimizerIF *self)
{
    Algorithm alg = self->getAlgorithm(self);
    switch (alg)
    {
    case PSO:
        PSOOptimizer_Dtor((PSOOptimizer *)self);
        break;
    case GA:
        GAOptimizer_Dtor((GAOptimizer *)self);
        break;
    case DE:
        DEOptimizer_Dtor((DEOptimizer *)self);
        break;
    case BO:
        BayesianOptimizer_Dtor((BayesianOptimizer *)self);
        break;
    default:
        PRINTF("No support for optimizer by algorithm:%d\n", alg);
        break;
    }
}

// member func of OptimizedParamIF
void OptimizedParamIF_Ctor(OptimizedParamIF *self)
{
    if (self)
    {
        self->getName = nullptr;
        self->update = nullptr;
        self->set = nullptr;
        self->cur = nullptr;
        self->max = nullptr;
        self->min = nullptr;
        self->dcur = nullptr;
        self->dmax = nullptr;
        self->dmin = nullptr;
        self->clone = nullptr;
        self->to_string = nullptr;
    }
}

void OptimizedParamIF_Dtor(OptimizedParamIF *self)
{
    if (self)
    {
        self->getName = nullptr;
        self->update = nullptr;
        self->set = nullptr;
        self->cur = nullptr;
        self->max = nullptr;
        self->min = nullptr;
        self->dcur = nullptr;
        self->dmax = nullptr;
        self->dmin = nullptr;
        self->clone = nullptr;
        self->to_string = nullptr;
        FREE(self);
    }
}

// member func of OptimizedParam
char *OptimizedParam_getName(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
        return derived->m_strName;
    }
    else
    {
        return nullptr;
    }
}

void *OptimizedParam_cur(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
        return (void *)&(derived->m_curValue);
    }
    else
    {
        return nullptr;
    }
}

void *OptimizedParam_max(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
        return (void *)&(derived->m_maxValue);
    }
    else
    {
        return nullptr;
    }
}

void *OptimizedParam_min(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
        return (void *)&(derived->m_minValue);
    }
    else
    {
        return nullptr;
    }
}

float OptimizedParam_dcur(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
        return (float)(derived->m_curValue);
    }
    else
    {
        return -32768.0;
    }
}

float OptimizedParam_dmax(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
        return (float)(derived->m_maxValue);
    }
    else
    {
        return -1.0;
    }
}

float OptimizedParam_dmin(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
        return (float)(derived->m_minValue);
    }
    else
    {
        return -1.0;
    }
}

void OptimizedParam_update(OptimizedParamIF *self, ParamOptimizer *p)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
#if FLOAT_PARAM
        float val = derived->m_funcOptimize(p, derived->m_strName, derived->m_minValue, derived->m_maxValue);
#else
        int val = derived->m_funcOptimize(p, derived->m_strName, derived->m_minValue, derived->m_maxValue);
#endif
        if (derived->m_funcUpdate)
        {   
            derived->m_curValue = derived->m_funcUpdate(derived->m_strName, val);
        }
        else
        {
            derived->m_curValue = val;
        }
    }
}

void OptimizedParam_set(OptimizedParamIF *self, char *value, bool_t update, ParamOptimizer *p)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived && value)
    {
#if FLOAT_PARAM
        derived->m_curValue = atof(value);
#else
        derived->m_curValue = atoi(value);
#endif
        if (derived->m_curValue < derived->m_minValue)
        {
            derived->m_curValue = derived->m_minValue;
        }
        if (derived->m_curValue > derived->m_maxValue)
        {
            derived->m_curValue = derived->m_maxValue;
        }
        if (update)
        {
            derived->m_funcSet(p, derived->m_strName, derived->m_curValue);
        }
    }
}

OptimizedParamIF *OptimizedParam_clone(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    OptimizedParamIF *copy = (OptimizedParamIF *)OptimizedParam_Ctor(
        derived->m_strName,
        derived->m_curValue,
        derived->m_maxValue,
        derived->m_minValue,
        derived->m_funcUpdate,
        derived->m_funcOptimize,
        derived->m_funcSet);
    return copy;
}

char *OptimizedParam_to_string(OptimizedParamIF *self)
{
    OptimizedParam *derived = (OptimizedParam *)self;
    if (derived)
    {
#if FLOAT_PARAM
        sprintf(derived->m_strValue, "%f", derived->m_curValue);
#else
        sprintf(derived->m_strValue, "%d", derived->m_curValue);
#endif
        return derived->m_strValue;
    }
    else
    {
        return nullptr;
    }
}

OptimizedParam *OptimizedParam_Ctor(
    char *name,
#if FLOAT_PARAM
    float initValue,
    float maxValue,
    float minValue,
    int (*update_func)(char *, float),
    float (*optimize_func)(ParamOptimizer *, char *, float, float),
    void (*set_func)(ParamOptimizer *, char *, float))
#else
    int initValue,
    int maxValue,
    int minValue,
    int (*update_func)(char *, int),
    int (*optimize_func)(ParamOptimizer *, char *, int, int),
    void (*set_func)(ParamOptimizer *, char *, int))
#endif
{
    OptimizedParam *self = (OptimizedParam *)MALLOC(sizeof(OptimizedParam));
    if (self)
    {
        OptimizedParamIF_Ctor((OptimizedParamIF *)&(self->base));
        self->base.getName = OptimizedParam_getName;
        self->base.cur = OptimizedParam_cur;
        self->base.max = OptimizedParam_max;
        self->base.min = OptimizedParam_min;
        self->base.dmax = OptimizedParam_dmax;
        self->base.dmin = OptimizedParam_dmin;
        self->base.dcur = OptimizedParam_dcur;
        self->base.update = OptimizedParam_update;
        self->base.set = OptimizedParam_set;
        self->base.clone = OptimizedParam_clone;
        self->base.to_string = OptimizedParam_to_string;
        self->m_strName = (char *)MALLOC(strlen(name) + 1);
        self->m_strValue = strValue_Ctor();
        if (self->m_strName)
        {
            strcpy(self->m_strName, name);
        }
        self->m_curValue = initValue;
        self->m_maxValue = maxValue;
        self->m_minValue = minValue;
        self->m_funcUpdate = update_func;
        self->m_funcOptimize = optimize_func;
        self->m_funcSet = set_func;
    }
    return self;
}

void OptimizedParam_Dtor(OptimizedParam *self)
{
    if (self)
    {
        self->base.getName = nullptr;
        self->base.cur = nullptr;
        self->base.max = nullptr;
        self->base.min = nullptr;
        self->base.dmax = nullptr;
        self->base.dmin = nullptr;
        self->base.dcur = nullptr;
        self->base.update = nullptr;
        self->base.set = nullptr;
        self->base.clone = nullptr;
        self->base.to_string = nullptr;
        OptimizedParamIF_Dtor(&(self->base));

        FREE(self->m_strName);
        self->m_strName = nullptr;
        strValue_Dtor(self->m_strValue);
        self->m_strValue=nullptr;
        self->m_funcUpdate = nullptr;
        self->m_funcOptimize = nullptr;
        self->m_funcSet = nullptr;
    }
    FREE(self);
}

// member func of ParamOptimizer
#if FLOAT_PARAM
void ParamOptimizer_regist(ParamOptimizerIF *self, char *key, float min, float max, int (*update)(char *, float))
#else
void ParamOptimizer_regist(ParamOptimizerIF *self, char *key, int min, int max, int (*update)(char *, int))
#endif
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    OptimizedParamIF *p = (OptimizedParamIF *)OptimizedParam_Ctor(key, min, max, min, update, derived->optimize, derived->set_value);
    Map_StringToPtr_Visit(derived->m_mapParam, key)->m_ptr = p;
}

void ParamOptimizer_unregist(ParamOptimizerIF *self, char *key)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    if (derived && key)
    {
        Map_StringToPtr_Erase(derived->m_mapParam, key);
    }
}

void ParamOptimizer_getOptimizedParam(ParamOptimizerIF *self, Map_StringToString param)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    Map_StringToPtr iter = derived->m_mapParam;
    while (iter)
    {
        OptimizedParam *derived_m_ptr = (OptimizedParam *)iter->m_ptr;
#if FLOAT_PARAM
        float_to_string(derived_m_ptr->m_strValue,derived_m_ptr->m_curValue);
        Map_StringToString_PushBack(param, iter->m_string, derived_m_ptr->m_strValue);
#else
        int_to_string(derived_m_ptr->m_strValue,derived_m_ptr->m_curValue);
        Map_StringToString_PushBack(param, iter->m_string, derived_m_ptr->m_strValue);
#endif
        iter = iter->m_next;
    }
}

void ParamOptimizer_getOptimizedParams(ParamOptimizerIF *self, Map_VectorToMap param)
{
    return;
}

char *ParamOptimizer_getOptimizedTarget(ParamOptimizerIF *self)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    if (derived && derived->m_prevTarget && derived->m_prevTarget->m_val)
    {
        sprintf(derived->m_strvalue, "%f", *(derived->m_prevTarget->m_val));
        return derived->m_strvalue;
    }
    else
    {
        return nullptr;
    }
}

Vector_Vector_String ParamOptimizer_getOptimizedTargets(ParamOptimizerIF *self)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    if (derived)
    {
        Vector_Vector_String result = Vector_Vector_String_Ctor();
        Vector_String ep = Vector_String_Ctor();
        Vector_Float iter = derived->m_prevTarget;
        while (iter)
        {
            float_to_string(derived->m_strvalue,*iter->m_val);
            Vector_String_PushBack(ep, derived->m_strvalue);
            iter = iter->m_next;
        }
        Vector_Vector_String_PushBack(result, ep);
        return result;
    }
    else
    {
        return nullptr;
    }
}

void ParamOptimizer_getCurrentParam(ParamOptimizerIF *self, Map_StringToString param)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    if (derived && param)
    {
        Map_StringToPtr iter = derived->m_mapParam;
        while (iter)
        {
            if (Map_StringToString_Find(param, iter->m_string) == nullptr)
            {
                Map_StringToString_PushBack(param, iter->m_string, iter->m_ptr->to_string(iter->m_ptr));
            }
            else
            {
                FREE(Map_StringToString_Visit(param, iter->m_string)->m_value);
                Map_StringToString_Visit(param, iter->m_string)->m_value = (char *)MALLOC(strlen(iter->m_ptr->to_string(iter->m_ptr)) + 1);
                strcpy(Map_StringToString_Visit(param, iter->m_string)->m_value, iter->m_ptr->to_string(iter->m_ptr));
            }
            iter = iter->m_next;
        }
    }
}

void ParamOptimizer_calibrateParam(ParamOptimizerIF *self, Map_StringToString param)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    if (self && param)
    {
        Map_StringToPtr iter = derived->m_mapParam;
        Map_StringToString iterParam = param;
        while (iter)
        {
            while (iterParam)
            {
                if (iterParam->m_key == iter->m_string)
                {
                    iter->m_ptr->set(iter->m_ptr, iterParam->m_value, true_t, derived);
                    iterParam = iterParam->m_next;
                }
            }
            iter = iter->m_next;
        }
    }
}

void ParamOptimizer_update(ParamOptimizerIF *self)
{
    Map_StringToPtr iter;
    ParamOptimizer *derived = (ParamOptimizer *)self;  
    if (derived && derived->suite->target_func)
    {
        Vector_Float suite_fitness = derived->suite->target_func(derived->suite);
        if (suite_fitness && suite_fitness->m_val) {
            Vector_Float_Assign(derived->m_prevTarget, suite_fitness);
        } else {
            Vector_Float_PushBack(derived->m_prevTarget, -32768.0);
        }
        derived->append_sample(derived, derived->m_mapParam, derived->m_prevTarget);  
        derived->update_intern(derived);
 
        iter = derived->m_mapParam;
        while (iter)
        {
            iter->m_ptr->update(iter->m_ptr, derived);
            iter = iter->m_next;
        }
    }
}

void ParamOptimizer_completeTrial(ParamOptimizerIF *self, Map_StringToString param, Vector_Float result)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    if (derived && param && result)
    {
        derived->update_intern_param(derived, param, result);
    }
}

void ParamOptimizer_update_intern_param(ParamOptimizer *self, Map_StringToString param, Vector_Float result)
{
    return;
}

bool_t ParamOptimizer_getTrial(ParamOptimizerIF *self, Map_StringToString param)
{
    return false_t;
}

enum Algorithm ParamOptimizer_getAlgorithm(ParamOptimizerIF *self)
{
    return EMPTY;
}

#if FLOAT_PARAM
float ParamOptimizer_optimize(struct ParamOptimizer *self, char *key, float min, float max)
#else
int ParamOptimizer_optimize(struct ParamOptimizer *self, char *key, int min, int max)
#endif
{
    return min;
}
#if FLOAT_PARAM
void ParamOptimizer_set_value(struct ParamOptimizer *self, char *key, float value)
#else
void ParamOptimizer_set_value(struct ParamOptimizer *self, char *key, int value)
#endif
{
    return;
}

bool_t ParamOptimizer_isTrainingEnd(ParamOptimizerIF *self)
{
    return false_t;
}
/*
static void SignalHandle(char *data, int size)
{
    FILE *fp = nullptr;
    fp = fopen("glog_dump.log", "w+");
    int sz = size < strlen(data) ? size : strlen(data);
    char *str = (char *)MALLOC(sz + 1);
    int i = 0;

    PRINTF(" Signal start \n");

    for (i = 0; i < sz; ++i)
    {
        str[i] = data[i];
    }
    str[sz] = '\0';
    fprintf(fp, "%s", str);
    fclose(fp);
    PRINTF("%s\n", str);
    FREE(str);
    str = nullptr;
}
*/
void ParamOptimizer_initLogging(ParamOptimizerIF *self, char *argv)
{
    return;
}

void ParamOptimizer_setPCAWindow(ParamOptimizerIF *self, int size)
{
    ParamOptimizer *derived = (ParamOptimizer *)self;
    if (derived)
    {
        derived->m_windowSize = size;
    }
}

void ParamOptimizer_append_sample(ParamOptimizer *self, Map_StringToPtr param, Vector_Float fitness)
{
    Map_StringToFloat temp;
    Map_StringToPtr iter;
    if (!fitness->m_val)
    {
        return;
    }
    temp = Map_StringToFloat_Ctor();
    iter = param;
    while (iter)
    {   
        Map_StringToFloat_PushBack(temp, iter->m_string, iter->m_ptr->dcur(iter->m_ptr));
        iter = iter->m_next;
    }
    Map_MapToFloat_PushBack(self->m_mapHist, temp, *fitness->m_val);
    if (self->m_windowSize <= 0)
    {   
        return;
    }
    if (Vector_FloatToMap_Size(self->m_sampleWindow) == self->m_windowSize)
    {
        float small = 0;
        int index = 0;
        int i;
        for (i = 0; i < Vector_FloatToMap_Size(self->m_sampleWindow); ++i)
        {
            if (i == 0)
            {
                small = *(Vector_FloatToMap_Visit(self->m_sampleWindow, i)->m_val);
            }
            else
            {
                if (small > *(Vector_FloatToMap_Visit(self->m_sampleWindow, i)->m_val))
                {
                    small = *(Vector_FloatToMap_Visit(self->m_sampleWindow, i)->m_val);
                    index = i;
                }
            }
        }
        if (*fitness->m_val > small)
        {
            Map_StringToPtr iter_param;
            *(Vector_FloatToMap_Visit(self->m_sampleWindow, index)->m_val) = *(fitness->m_val);
            iter_param = param;
            while (iter_param)
            {
                *(Map_StringToFloat_Visit(Vector_FloatToMap_Visit(self->m_sampleWindow, index)->m_msf, iter_param->m_string)->m_val) = iter_param->m_ptr->dcur(iter_param->m_ptr);
                iter_param = iter_param->m_next;
            }
        }
    }
    else
    {
        Map_StringToFloat b = Map_StringToFloat_Ctor();
        Map_StringToPtr iter_param = param;
        while (iter_param)
        {
            Map_StringToFloat_PushBack(b, iter_param->m_string, iter_param->m_ptr->dcur(iter_param->m_ptr));
            iter_param = iter_param->m_next;
        }
        Vector_FloatToMap_PushBack(self->m_sampleWindow, *(fitness->m_val), b);
    }
}

#ifdef ENABLE_PCA
static void printMat(CvMat *p)
{
    int row, col;
    for (row = 0; row < p->rows; row++)
    {
        for (col = 0; col < p->cols; col++)
        {
            PRINTF("%f  ", cvGet2D(p, row, col).val[0]);
        }
        PRINTF("\n");
    }
}

template <typename TA>
void ParamOptimizer<TA>::updateBestParam(CvMat *data, std::vector<int> &dims, std::map<std::string, OptimizedParamIF::Ptr> &param)
{
    if (dims.size() == 0)
        return;
    for (auto i : dims)
    {
        std::map<std::string, OptimizedParamIF::Ptr>::iterator iter;
        int j = 0;
        for (iter = param.begin(); iter != param.end(); ++iter, j++)
        {
            if (j == i)
            {
                iter->second->set(std::to_string(cvGet2D(data, 0, j).val[0]));
                break;
            }
        }
    }
}
#endif
void ParamOptimizer_pca_analysis(ParamOptimizer *self, Map_StringToPtr bestParam)
{
#ifdef ENABLE_PCA
    if (m_sampleWindow.size() == 0)
        return;
    int sampleSize = m_sampleWindow.size();
    int dim = bestParam.size();
    vector<int> updateDims;
    CvMat *data = cvCreateMat(sampleSize, dim, CV_32FC1);
    CvMat *mean = cvCreateMat(1, dim, CV_32FC1);
    CvMat *min = cvCreateMat(1, dim, CV_32FC1);
    CvMat *max = cvCreateMat(1, dim, CV_32FC1);
    CvMat *eigenVal = cvCreateMat(1, MIN(sampleSize, dim), CV_32FC1);
    CvMat *eigenVec = cvCreateMat(MIN(sampleSize, dim), dim, CV_32FC1);
    int row = 0;
    int col = 0;

    std::map<std::string, OptimizedParamIF::Ptr>::iterator iter;
    for (iter = bestParam.begin(); iter != bestParam.end(); ++iter)
#ifdef USE_GLOG
        VLOG(STATISTIC) << "[before PCA] best key: " << iter->first << "  value: " << iter->second->to_string();
#else
        std::cout << "[before PCA] best key: " << iter->first << "  value: " << iter->second->to_string() << std::endl;
#endif

    for (row = 0; row < sampleSize; row++)
    {
        std::map<std::string, double>::iterator iter;
        iter = m_sampleWindow[row].second.begin();
        for (col = 0; col < dim; col++)
        {
            cvSet2D(data, row, col, {iter->second, 0, 0});
            iter++;
        }
        //PRINTF("sample %d fitness %lf\n", row, m_sampleWindow[row].first);
    }
    cvReduce(data, mean, 0, CV_REDUCE_AVG);
    cvReduce(data, min, 0, CV_REDUCE_MIN);
    cvReduce(data, max, 0, CV_REDUCE_MAX);
    //PRINTF("=========mean===========\n");
    //printMat(mean);
    //PRINTF("=========data===========\n");
    //printMat(data);
    for (row = 0; row < data->rows; row++)
    {
        for (col = 0; col < data->cols; col++)
        {
            float a = cvGet2D(data, row, col).val[0];
            float b = cvGet2D(max, 0, col).val[0];
            float c = cvGet2D(min, 0, col).val[0];
            cvSet2D(data, row, col, {(b - a) / (b - c) - 0.5, 0, 0});
        }
    }
    cvCalcPCA(data, mean, eigenVal, eigenVec, CV_PCA_DATA_AS_ROW);
    //    PRINTF("=========eigenVal==========\n");
    //    printMat(eigenVal);
    //    PRINTF("=========eigenVec==========\n");
    //    printMat(eigenVec);
    for (col = 0; col < eigenVal->cols - 1; col++)
    {
        if (cvGet2D(eigenVal, 0, col).val[0] > 0.01)
        {
            int row = col;
            int t = 0;
            for (t = 0; t < eigenVec->cols; t++)
            {
                if (cvGet2D(eigenVec, row, t).val[0] > 0.7)
                    updateDims.push_back(t);
            }
        }
    }
    updateBestParam(min, updateDims, bestParam);
    cvReleaseMat(&data);
    cvReleaseMat(&mean);
    cvReleaseMat(&min);
    cvReleaseMat(&eigenVal);
    cvReleaseMat(&eigenVec);
#else
    return;
#endif
}

bool_t ParamOptimizer_isInHistory(ParamOptimizer *self, Map_StringToPtr p_param)
{
    Map_StringToFloat temp = Map_StringToFloat_Ctor();
    Map_StringToPtr iter = p_param;
    while (iter)
    {
        if (Map_StringToFloat_Find(temp, iter->m_string))
        {
            *(Map_StringToFloat_Visit(temp, iter->m_string)->m_val) = iter->m_ptr->dcur(iter->m_ptr);
        }
        else
        {
            Map_StringToFloat_PushBack(temp, iter->m_string, iter->m_ptr->dcur(iter->m_ptr));
        }
        iter = iter->m_next;
    }
    if (Map_MapToFloat_Find(self->m_mapHist, temp))
    {
        Map_StringToFloat_Dtor(temp);
        temp = nullptr;
        return true_t;
    }
    Map_StringToFloat_Dtor(temp);
    temp = nullptr;
    return false_t;
}

bool_t ParamOptimizer_isSame(ParamOptimizer *self, Map_StringToPtr p1, Map_StringToPtr p2)
{
    if (Map_StringToPtr_Size(p1) != Map_StringToPtr_Size(p2))
    {
        return false_t;
    }
    while (p1)
    {
        OptimizedParamIF *temp = Map_StringToPtr_Visit(p2, p1->m_string)->m_ptr;
        if (temp->dcur(temp) != p1->m_ptr->dcur(p1->m_ptr))
        {
            return false_t;
        }
        p1 = p1->m_next;
    }
    return true_t;
}
char* strValue_Ctor()
{
    char *p = (char*)MALLOC(128*sizeof(char));
    if (!p)
    {
        PRINTF("%s:%d failed!\n", __func__, __LINE__);
        return NULL;
    }
    strcpy(p,"none");
    return p;
}
void strValue_Dtor(char *p) {
    if (p) {
        FREE(p);
    }
}
void ParamOptimizer_Ctor(ParamOptimizer *op, OptParam *p)
{
    ParamOptimizerIF *oif = &(op->base);
    oif->update = ParamOptimizer_update;
    oif->completeTrial = ParamOptimizer_completeTrial;
    oif->getTrial = ParamOptimizer_getTrial;
    oif->getAlgorithm = ParamOptimizer_getAlgorithm;
    oif->regist = ParamOptimizer_regist;
    oif->unregist = ParamOptimizer_unregist;
    oif->getOptimizedParam = ParamOptimizer_getOptimizedParam;
    oif->getOptimizedParams = ParamOptimizer_getOptimizedParams;
    oif->getOptimizedTarget = ParamOptimizer_getOptimizedTarget;
    oif->getOptimizedTargets = ParamOptimizer_getOptimizedTargets;
    oif->getCurrentParam = ParamOptimizer_getCurrentParam;
    oif->calibrateParam = ParamOptimizer_calibrateParam;
    oif->isTrainingEnd = ParamOptimizer_isTrainingEnd;
    oif->initLogging = ParamOptimizer_initLogging;
    oif->setPCAWindow = ParamOptimizer_setPCAWindow;

    //static struct ParamOptimizerIF *getParamOptimizer(OptParam<TA> &param);

    op->pca_analysis = ParamOptimizer_pca_analysis;
    op->append_sample = ParamOptimizer_append_sample;
    op->update_intern = nullptr;
    op->update_intern_param = ParamOptimizer_update_intern_param;
    op->dump_csv = nullptr;
    op->isInHistory = ParamOptimizer_isInHistory;
    op->isSame = ParamOptimizer_isSame;
    op->optimize = ParamOptimizer_optimize;
    op->set_value = ParamOptimizer_set_value;

    op->m_mapParam = Map_StringToPtr_Ctor();
    op->suite = p->suite;
    op->m_prevTarget = Vector_Float_Ctor();
    op->m_initGlog = false_t;
    op->m_logname = nullptr;
    op->m_strvalue = strValue_Ctor();
    op->m_windowSize = 0;
    op->m_sampleWindow = Vector_FloatToMap_Ctor();
    op->m_mapHist = Map_MapToFloat_Ctor();
#ifndef KERNEL_MODULE
    op->m_csvOut = nullptr;

    char *envVar = getenv("DUMP_TO_CSV");
    if (envVar)
    {
        op->m_csvOut = fopen(envVar, "w");
        if (!op->m_csvOut)
        {
            PRINTF("open %s fail/n", envVar);
        }
    }
#endif
}

void ParamOptimizer_Dtor(ParamOptimizer *self)
{
    if (self)
    {
        self->base.update = nullptr;
        self->base.completeTrial = nullptr;
        self->base.getTrial = nullptr;
        self->base.getAlgorithm = nullptr;
        self->base.regist = nullptr;
        self->base.unregist = nullptr;
        self->base.getOptimizedParam = nullptr;
        self->base.getOptimizedParams = nullptr;
        self->base.getOptimizedTarget = nullptr;
        self->base.getOptimizedTargets = nullptr;
        self->base.getCurrentParam = nullptr;
        self->base.calibrateParam = nullptr;
        //static struct ParamOptimizerIF *getParamOptimizer(OptParam<TA> &param);
        self->base.isTrainingEnd = nullptr;
        self->base.initLogging = nullptr;
        self->base.setPCAWindow = nullptr;
        self->pca_analysis = nullptr;
        self->append_sample = nullptr;
        self->update_intern = nullptr;
        self->update_intern_param = nullptr;
        self->dump_csv = nullptr;
        self->isInHistory = nullptr;
        self->isSame = nullptr;
        self->optimize = nullptr;
        self->set_value = nullptr;
        strValue_Dtor(self->m_strvalue);
        self->m_strvalue=nullptr;
        Map_StringToPtr_Dtor(self->m_mapParam);
        self->m_mapParam = nullptr;
        self->suite = nullptr;
        Vector_Float_Dtor(self->m_prevTarget);
        self->m_prevTarget = nullptr;
        Vector_FloatToMap_Dtor(self->m_sampleWindow);
        self->m_sampleWindow = nullptr;
#ifndef KERNEL_MODULE
        if (self->m_csvOut)
        {
            fclose(self->m_csvOut);
        }
#endif
        Map_MapToFloat_Dtor(self->m_mapHist);
        self->m_mapHist = nullptr;
    }
}

int getOptParam(Algorithm algo, Suite* p_suite, int gen, int pop, OptParam** pp_OptParam)
{
    if (nullptr == pp_OptParam || nullptr == p_suite) {
        PRINTF("Invalid parameter for getOptParam()!");
        return -1;
    }
    switch (algo) {
        case PSO:
        {
            PSOOptParam *p_PSOOptParam = (PSOOptParam *)MALLOC(sizeof(PSOOptParam));
            if (p_PSOOptParam) {
                p_PSOOptParam->base.algorithm = PSO;
                p_PSOOptParam->base.suite = p_suite;
                p_PSOOptParam->iter_num = gen;
                p_PSOOptParam->swarm_num = pop;
                p_PSOOptParam->update_weight = 0.8;
                p_PSOOptParam->update_cg = 0.5;
                p_PSOOptParam->update_cp = 0.5;
                *pp_OptParam = (OptParam *)p_PSOOptParam;
            }
            else {
                PRINTF("Fail to malloc memory for PSOOptParam!\n");
                return -1;
            }
            break;
        }
        case GA:
        {
            GAOptParam *p_GAOptParam = (GAOptParam *)MALLOC(sizeof(GAOptParam));
            if (p_GAOptParam) {
                p_GAOptParam->base.algorithm = GA;
                p_GAOptParam->base.suite = p_suite;
                p_GAOptParam->gen_num = gen;
                p_GAOptParam->pop_size = pop;
                p_GAOptParam->mutp = 0.1;
                *pp_OptParam = (OptParam *)p_GAOptParam;
            }
            else {
                PRINTF("Fail to malloc memory for GAOptParam!\n");
                return -1;
            }
            break;
        }
        case DE:
        {
            DEOptParam *p_DEOptParam = (DEOptParam *)MALLOC(sizeof(DEOptParam));
            if (p_DEOptParam) {
                p_DEOptParam->base.algorithm = DE;
                p_DEOptParam->base.suite = p_suite;
                p_DEOptParam->gen_num = gen;
                p_DEOptParam->pop_size = pop;
                *pp_OptParam = (OptParam *)p_DEOptParam;
            }
            else {
                PRINTF("Fail to malloc memory for DEOptParam!\n");
                return -1;
            }
            break;
        }
        case BO:
        {
            BOOptParam *p_BOOptParam = (BOOptParam *)MALLOC(sizeof(BOOptParam));
            if (p_BOOptParam) {
                p_BOOptParam->base.algorithm = BO;
                p_BOOptParam->base.suite = p_suite;
                p_BOOptParam->random_state = pop;
                p_BOOptParam->iter_num = gen;
                p_BOOptParam->cfg = "Manta"; //Never used
                *pp_OptParam = (OptParam *)p_BOOptParam;
            }
            else {
                PRINTF("Fail to malloc memory for BOOptParam!\n");
                return -1;
            }
            break;
        }
        default:
            //Should not come here since we've already checked supported algos in parseAlgoRelated().
            return -1;
    }
    return 0;
}

void OptParam_Dtor(OptParam* p_OptParam) {
    if (p_OptParam) {
        FREE(p_OptParam);
        p_OptParam = nullptr;
    }
}

int parseAlgoRelated(int argc, char **argv, Algorithm* p_algo, int* p_gen, int* p_pop)
{
    int offset = 0;
    int i = 0;

    if (nullptr == argv || nullptr == p_algo || nullptr == p_gen || nullptr == p_pop) {
        PRINTF("%s:%d Invalid input parameter.\n", __func__, __LINE__);
        return -1;
    }

    // Parse -algo, -pop, -gen, skip all others
    for (i = 1; i < argc; i += offset) {
        offset = 1;
        if (strcmp(argv[i], "-algo") == 0) {
            if (i == argc - 1) {
                PRINTF("Algorithm not specified after -algo.\n");
                return -1;
            }
            if (strcmp(argv[i + 1], "PSO") == 0) {
                *p_algo = PSO;
            }
            else if (strcmp(argv[i + 1], "GA") == 0) {
                *p_algo = GA;
            }
            else if (strcmp(argv[i + 1], "DE") == 0) {
                *p_algo = DE;
            }
            else if (strcmp(argv[i + 1], "BO") == 0) {
                *p_algo = BO;
            }
            else {
                PRINTF("%s:%d Unspported algorithm %s.\n", __func__, __LINE__, argv[i + 1]);
                PRINTF("Currently support PSO, GA, DE and BO.\n");
                return -1;
            }
            offset = 2;
        }
        else if (strcmp(argv[i], "-pop") == 0) {
            if (i == argc - 1) {
                PRINTF("Number not specified after -pop.\n");
                return -1;
            }
            if ((*p_pop = atoi(argv[i + 1])) == 0) {
                PRINTF("Wrong parameter for -pop!\n");
                return -1;
            }
            offset = 2;
        }
        else if (strcmp(argv[i], "-gen") == 0) {
            if (i == argc - 1) {
                PRINTF("Number not specified after -gen.\n");
                return -1;
            }
            if ((*p_gen = atoi(argv[i + 1])) == 0) {
                PRINTF("Wrong parameter for -gen!\n");
                return -1;
            }
            offset = 2;
        }
    }
    return 0;
}

int registOptimizer(ParamOptimizerIF **p_ParamOptimizer, Suite *p_Suite, OptParam *p_OptParam, Map_StringToString *mss,Map_StringToString *curBestmss)
{
    int n_var, i;
    if (nullptr == p_Suite || nullptr == p_OptParam) {
        PRINTF("Invalid parameter for tune()!\n");
        return -1;
    }

    ParamOptimizerIF *optimizerIF = getParamOptimizer(p_OptParam);
    Map_StringToString map_s2s = Map_StringToString_Ctor();
    Map_StringToString curbest_map_s2s = Map_StringToString_Ctor();

    if (nullptr == optimizerIF || nullptr == map_s2s || nullptr == map_s2s) {
        PRINTF("Fail to get Param Optimizer or construct Map_StringToString!\n");
        if (map_s2s) {
            Map_StringToString_Dtor(map_s2s);
            map_s2s = nullptr;
        }
        if (curbest_map_s2s) {
            Map_StringToString_Dtor(curbest_map_s2s);
            curbest_map_s2s = nullptr;
        }
        if (optimizerIF) {
            ParamOptimizerIF_Dtor(optimizerIF);
            optimizerIF = nullptr;
        }
        return -1;
    }

    p_Suite->get_var(p_Suite);
    n_var = Vector_String_Size(p_Suite->var);
    for (i = 0; i < n_var; ++i)
    {
#if FLOAT_PARAM
        optimizerIF->regist(optimizerIF, Vector_String_Visit(p_Suite->var, i)->m_string, *(Vector_Float_Visit(p_Suite->var_min, i)->m_val), *(Vector_Float_Visit(p_Suite->var_max, i)->m_val), nullptr);
#else
        optimizerIF->regist(optimizerIF, Vector_String_Visit(p_Suite->var, i)->m_string, *(Vector_Int_Visit(p_Suite->var_min, i)->m_val), *(Vector_Int_Visit(p_Suite->var_max, i)->m_val), nullptr);
#endif
    }

    *p_ParamOptimizer = optimizerIF;
    *mss = map_s2s;
    *curBestmss = curbest_map_s2s;
    return 0;
}

bool tuneOneIteration(ParamOptimizerIF *p_ParamOptimizer, Suite *p_Suite, Map_StringToString mss,Map_StringToString curBestmss,float* pbest_fitness){
    if (p_ParamOptimizer->isTrainingEnd(p_ParamOptimizer) == false_t)
    {   
        p_ParamOptimizer->update(p_ParamOptimizer);
        p_ParamOptimizer->getCurrentParam(p_ParamOptimizer, mss);
        int i = 0;
        int size = Map_StringToString_Size(mss);
#if FLOAT_PARAM
        float input[size];
#else
        int input[size];
#endif
        Map_StringToString iter = mss;
        while (iter)
        {
            float tmpVal = atof(iter->m_value);
            input[i] = tmpVal;
            iter = iter->m_next;
            ++i;
        }
        p_Suite->fitness = p_Suite->evaluate(p_Suite, input);
        p_ParamOptimizer->getOptimizedParam(p_ParamOptimizer, curBestmss);
        float best_fitness = atof(p_ParamOptimizer->getOptimizedTarget(p_ParamOptimizer));
        *pbest_fitness=best_fitness;
        return false;
    }
    p_ParamOptimizer->update(p_ParamOptimizer);
    float best_fitness = atof(p_ParamOptimizer->getOptimizedTarget(p_ParamOptimizer));
    *pbest_fitness=best_fitness;
    p_ParamOptimizer->getOptimizedParam(p_ParamOptimizer, curBestmss);
    return true;
}

void freeSpace(ParamOptimizerIF *p_ParamOptimizer, Map_StringToString mss, Map_StringToString curBestmss,
               Suite *ppSuite, OptParam* mOptParam) {
    Algorithm alg = p_ParamOptimizer->getAlgorithm(p_ParamOptimizer);
    if (mss) {
        Map_StringToString_Dtor(mss);
        mss = nullptr;
    }
    if (curBestmss) {
        Map_StringToString_Dtor(curBestmss);
        curBestmss = nullptr;
    }
    if (ppSuite) {
        Suite_Dtor(ppSuite);
        ppSuite = nullptr;
    }
    if (p_ParamOptimizer) {
        ParamOptimizerIF_Dtor(p_ParamOptimizer);
        p_ParamOptimizer = nullptr;
    }
    if (mOptParam) {
        switch (alg) {
        case PSO: {
            PSOOptParam *psomOptParam = (PSOOptParam *)mOptParam;
            FREE(psomOptParam);
            psomOptParam = nullptr;
            mOptParam = nullptr;
        } break;
        case GA: {
            GAOptParam *gamOptParam = (GAOptParam *)mOptParam;
            FREE(gamOptParam);
            gamOptParam = nullptr;
            mOptParam = nullptr;
        } break;
        case DE: {
            DEOptParam *demOptParam = (DEOptParam *)mOptParam;
            FREE(demOptParam);
            demOptParam = nullptr;
            mOptParam = nullptr;
        } break;
        case BO: {
            BOOptParam *bomOptParam = (BOOptParam *)mOptParam;
            FREE(bomOptParam);
            bomOptParam = nullptr;
            mOptParam = nullptr;
        } break;
        default:
            PRINTF("No support for optimizer by algorithm:%d\n", alg);
            break;
        }
    }
}

int tuneSuiteWithOptParam(Suite *p_Suite, OptParam *p_OptParam)
{   
    ParamOptimizerIF *p_ParamOptimizer;
    Map_StringToString mss;
    Map_StringToString iter;
    int n_var, i, size;
    char *best_fitness;
    if (nullptr == p_Suite || nullptr == p_OptParam) {
        PRINTF("Invalid parameter for tune()!\n");
        return -1;
    }

    p_ParamOptimizer = getParamOptimizer(p_OptParam);
    mss = Map_StringToString_Ctor();

    if(nullptr == p_ParamOptimizer || nullptr == mss) {
        PRINTF("Fail to get Param Optimizer or construct Map_StringToString!\n");
        if (mss) {
            Map_StringToString_Dtor(mss);
            mss = nullptr;
        }
        if (p_ParamOptimizer) {
            ParamOptimizerIF_Dtor(p_ParamOptimizer);
            p_ParamOptimizer = nullptr;
        }
        return -1;
    }

    p_Suite->get_var(p_Suite);
    n_var = Vector_String_Size(p_Suite->var);
    for (i = 0; i < n_var; ++i)
    {
#if FLOAT_PARAM
        p_ParamOptimizer->regist(p_ParamOptimizer, Vector_String_Visit(p_Suite->var, i)->m_string, *(Vector_Float_Visit(p_Suite->var_min, i)->m_val), *(Vector_Float_Visit(p_Suite->var_max, i)->m_val), nullptr);
#else
        p_ParamOptimizer->regist(p_ParamOptimizer, Vector_String_Visit(p_Suite->var, i)->m_string, *(Vector_Int_Visit(p_Suite->var_min, i)->m_val), *(Vector_Int_Visit(p_Suite->var_max, i)->m_val), nullptr);
#endif
    }

    while (p_ParamOptimizer->isTrainingEnd(p_ParamOptimizer) == false_t)
    {   
        p_ParamOptimizer->update(p_ParamOptimizer);
        p_ParamOptimizer->getCurrentParam(p_ParamOptimizer, mss);
        i = 0;
        size = Map_StringToString_Size(mss);
#if FLOAT_PARAM
        float input[size];
#else
        int input[size];
#endif
        Map_StringToString iter = mss;
        while (iter)
        {
            float tmpVal = atof(iter->m_value);
            input[i] = tmpVal;
            iter = iter->m_next;
            ++i;
        }
        p_Suite->fitness = p_Suite->evaluate(p_Suite, input);
    }
    p_ParamOptimizer->update(p_ParamOptimizer);
    best_fitness = p_ParamOptimizer->getOptimizedTarget(p_ParamOptimizer);
    p_ParamOptimizer->getOptimizedParam(p_ParamOptimizer, mss);

    iter = mss;
    PRINTF("best_fitness = %s\n", best_fitness);
    while (iter)
    {
        PRINTF("item: %s = %s\n", iter->m_key, iter->m_value);
        iter = iter->m_next;
    }

    if (mss) {
        Map_StringToString_Dtor(mss);
        mss = nullptr;
    }
    if (p_ParamOptimizer) {
        ParamOptimizerIF_Dtor(p_ParamOptimizer);
        p_ParamOptimizer = nullptr;
    }
    return 0;
}

int tune(int argc, char *argv[], Suite* p_suite) {

    // default parameters
    Algorithm algo = PSO;
    int gen = 60;
    int pop = 100;
    OptParam * p_OptParam = nullptr;
    int res = 0;

    do {
        res = parseAlgoRelated(argc, argv, &algo, &gen, &pop);
        if (res < 0) {
            PRINTF("Fail to parse algo Related parameters! Check the usage by \"pta_launcher -help\".\n");
            break;
        }

        res = getOptParam(algo, p_suite, gen, pop, &p_OptParam);
        if (res < 0) {
            PRINTF("Fail to get OptParam!\n");
            break;
        }

        res = tuneSuiteWithOptParam(p_suite, p_OptParam);
        if (res < 0) {
            PRINTF("Fail to tune the specified suite!\n");
            break;
        }
    } while (false_t);

    // Release resources
    OptParam_Dtor(p_OptParam);
    return res;
}

int checkHelp(int argc, char **argv) {
    int i = 0;

    if (nullptr == argv) {
        PRINTF("%s:%d Invalid parameter!", __func__, __LINE__);
        return -1;
    }

    // Parse -help, skip others
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-help") == 0) {
            PRINTF(HELP_INFO);
            if ((i == 1) && (argc == 2)) {
                // -help is the only option
                // just display help message
                return 1;
            }
            break;
        }
    }
    return 0;
}