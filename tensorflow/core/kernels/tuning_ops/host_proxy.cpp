#include "host_proxy.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "MapStringToInt.h"
#include "OptimizerIF.h"
#include "Suite.h"
#include "VectorFloat.h"
#include "VectorInt.h"
#include "VectorString.h"
#ifdef __cplusplus
}
#endif

#include <iostream>
#include <stdlib.h>

struct SuiteWrapper {
    Suite base;
    HostProxy *proxy;
};

void SuiteWrapperGetVar(Suite *self) {
    SuiteWrapper *wrapper = (SuiteWrapper *)self;

    auto params = wrapper->proxy->getParamters();

    for (auto param : params) {
        Vector_String_PushBack(self->var, (char *)param.name.c_str());
        Vector_Float_PushBack(self->var_min, param.min_max_val.first);
        Vector_Float_PushBack(self->var_max, param.min_max_val.second);
    }
}

Vector_Float SuiteWrapperEvaluate(Suite *self, float *x) {
    SuiteWrapper *wrapper = (SuiteWrapper *)self;
    auto cust_func = wrapper->proxy->getEvaluateFunc();
    std::vector<int> cur_params;
    cur_params.push_back(x[0]);
    cur_params.push_back(x[1]);
    cur_params.push_back(x[2]);
    const float curiter_avg_latency = cust_func(cur_params);
    float latency = (-1.0) * curiter_avg_latency;
    if (Vector_Float_Size(self->fitness) > 0) {
        *Vector_Float_Visit(self->fitness, 0)->m_val = latency;
    } else {
        Vector_Float_PushBack(self->fitness, latency);
    }
    return self->fitness;
}

HostProxy::HostProxy(const char *name) : mName(name) {
}

HostProxy::~HostProxy() {
    if (mSuiteBase) {
        free(mSuiteBase);
    }
}

void HostProxy::SetHaltTuingConditions(const int max_tuning_time_ms, const int max_tuning_rounds) {
}

void HostProxy::SetEvaluateFunc(EvaluateFunc evaluate) {
    mEvaluateFunc = evaluate;
}
HostProxy::EvaluateFunc &HostProxy::getEvaluateFunc() {
    return mEvaluateFunc;
}

void HostProxy::SetParamter(const char *var_name, int var_min, int var_max) {
    HostParam param{var_name, std::make_pair(var_min, var_max)};
    mTuneParams.push_back(param);
}

std::vector<HostParam> &HostProxy::getParamters() {
    return mTuneParams;
}

std::map<std::string, int> HostProxy::GetTunedResult() {
    return mTunedResult;
}

bool HostProxy::Start() {
    if (mState == State::RUNNING) {
        std::cout << "Host proxy: " << mName << " is already running!" << std::endl;
        return true;
    }

    if (mSuiteBase == nullptr) {
        SuiteWrapper *suite_wrapper = (SuiteWrapper *)malloc(sizeof(SuiteWrapper));
        if (suite_wrapper == nullptr) {
            std::string msg = "Cannot allocate memory for proxy: " + mName;
            // throw std::runtime_error(msg); todo(marvin): show the error info within tf.
        }

        Suite_Ctor(&(suite_wrapper->base));
        suite_wrapper->proxy = this;
        suite_wrapper->base.get_var = SuiteWrapperGetVar;
        suite_wrapper->base.evaluate = SuiteWrapperEvaluate;

        Algorithm algo = PSO;
        int gen = 10;
        int pop = 10;
        OptParam *p_OptParam = nullptr;

        Suite *pp_Suite = &(suite_wrapper->base);
        getOptParam(algo, pp_Suite, gen, pop, &p_OptParam);
        tuneSuiteWithOptParam(pp_Suite, p_OptParam);
    }
    mState = State::RUNNING;
    return true;
}

bool HostProxy::Stop() {
    if (mSuiteBase) {
        free(mSuiteBase);
        mSuiteBase = nullptr;
    }
    return true;
}

bool HostProxy::Suspend() {
    return true;
}
