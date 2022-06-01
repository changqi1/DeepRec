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
    auto params = wrapper->proxy->GetParamters();

    for (auto param : params) {
        Vector_String_PushBack(self->var, (char *)param.name.c_str());
#if FLOAT_PARAM
        Vector_Float_PushBack(self->var_min, param.min_max_val.first);
        Vector_Float_PushBack(self->var_max, param.min_max_val.second);
#else
        Vector_Int_PushBack(self->var_min, param.min_max_val.first);
        Vector_Int_PushBack(self->var_max, param.min_max_val.second);
#endif
    }
}

Vector_Float SuiteWrapperEvaluate(Suite *self, int *x) {
    SuiteWrapper *wrapper = (SuiteWrapper *)self;
    auto cust_func = wrapper->proxy->GetEvaluateFunc();
    auto size = wrapper->proxy->GetParamsNums();
    std::vector<int> cur_params;
    for (int i = 0; i < size; i++) {
        cur_params.push_back(x[i]);
    }
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
    mCondition.notify_one();
    if (mTuneThread.joinable()) {
        mTuneThread.join();
    }
}

void HostProxy::SetConditionFunc(ConditionFunc condition) {
    std::lock_guard<std::mutex> lk(mMutex);
    mConditionFunc = condition;
}
HostProxy::ConditionFunc &HostProxy::GetConditionFunc() {
    return mConditionFunc;
}

void HostProxy::SetAlgorithm(const char *algo,int gens,int pops) {
    std::lock_guard<std::mutex> lk(mMutex);
    if (0 == strcmp("PSO",algo)) {
        mAlgo = PSO;
    } else if (0 == strcmp("GA",algo)) {
        mAlgo = GA;
    } else if (0 == strcmp("DE",algo)) {
        mAlgo = DE;
    } else if (0 == strcmp("BO",algo)) {
        mAlgo = BO;
    } else {
        std::string msg = "There is no algorithm named " + (std::string)algo + " in host";
        // throw std::runtime_error(msg);
    }
    mIterManger.gens = gens;
    mIterManger.pops = pops;
}

void HostProxy::SetEvaluateFunc(EvaluateFunc evaluate) {
    std::lock_guard<std::mutex> lk(mMutex);
    mEvaluateFunc = evaluate;
}
HostProxy::EvaluateFunc &HostProxy::GetEvaluateFunc() {
    std::lock_guard<std::mutex> lk(mMutex);
    return mEvaluateFunc;
}

void HostProxy::SetParamter(const char *var_name, int var_min, int var_max) {
    std::lock_guard<std::mutex> lk(mMutex);
    HostParam param{var_name, std::make_pair(var_min, var_max)};
    mTuneParams.push_back(param);
}

std::vector<HostParam> &HostProxy::GetParamters() {
    return mTuneParams;
}

int HostProxy::GetParamsNums() {
    return mTuneParams.size();
}

float HostProxy::GetBestCost() {
    return -mBestFitness;
}

std::map<std::string, int> HostProxy::GetTunedResult() {
    return mTunedResult;
}

TuningContext HostProxy::GetTuningContext() {
    return mTuningContext;
}

HostProxy::State HostProxy::GetProxyState(){
    return mState;
  }

void HostProxy::Regist() {
    std::lock_guard<std::mutex> lk(mMutex);
    SuiteWrapper *suite_wrapper = (SuiteWrapper *)malloc(sizeof(SuiteWrapper));
    if (suite_wrapper == nullptr) {
        std::string msg = "Cannot allocate memory for proxy: " + mName;
        // throw std::runtime_error(msg);
    }
    Suite_Ctor(&(suite_wrapper->base));
    suite_wrapper->proxy = this;
    suite_wrapper->base.get_var = SuiteWrapperGetVar;
    suite_wrapper->base.evaluate = SuiteWrapperEvaluate;

    Algorithm algo = Algorithm(mAlgo);
    OptParam *p_OptParam = nullptr;
    Suite *pp_Suite = &(suite_wrapper->base);
    getOptParam(algo, pp_Suite, mIterManger.gens, mIterManger.pops, &p_OptParam);
    ParamOptimizerIF *mp_ParamOptimizer = nullptr;
    Map_StringToString mss = nullptr;
    Map_StringToString curBestmss = nullptr;
    registOptimizer(&mp_ParamOptimizer, pp_Suite, p_OptParam, &mss, &curBestmss);
    mOptimizerIF = (ParamOptimizerIF *)mp_ParamOptimizer;
    mParamsMapStringToString = (Map_StringToString)mss;
    mBestParamsMapStringToString = (Map_StringToString)curBestmss;
    mSuiteBase = (Suite *)pp_Suite;

    for(auto hostParam:mTuneParams){
        mTunedResult[hostParam.name] = -std::numeric_limits<int>::max();
        mTuningContext.cur_tune_params[hostParam.name] = -std::numeric_limits<int>::max();
        mTuningContext.cur_best_params[hostParam.name] = -std::numeric_limits<int>::max();
    }
}

void HostProxy::PushBestTunedResult() {
    std::lock_guard<std::mutex> lk(mMutex);
    Map_StringToString bestiter = (Map_StringToString)mBestParamsMapStringToString;
    if (bestiter->m_key) {
        mTuningContext.cur_best_fitness = -mBestFitness;
        while (bestiter) {
            std::string str(bestiter->m_key);
            mTunedResult[str] = atof(bestiter->m_value);
            mTuningContext.cur_best_params[str] = atof(bestiter->m_value);
            bestiter = bestiter->m_next;
        }
    }

}

void HostProxy::PushCurTunedResult(std::map<std::string, int> &tmpParams, float &tmpFitness) {
    std::lock_guard<std::mutex> lk(mMutex);
    if (!tmpParams.empty()) {
        std::map<std::string, int>::iterator lastiter;
        for (lastiter = tmpParams.begin(); lastiter != tmpParams.end(); lastiter++) {
            mTuningContext.cur_tune_params[lastiter->first] = lastiter->second;
        }
        mTuningContext.cur_tune_fitness = -tmpFitness;
    }

    Map_StringToString curiter = (Map_StringToString)mParamsMapStringToString;
    if (curiter->m_key) {
        while (curiter) {
            std::string str(curiter->m_key);
            tmpParams[str] = atof(curiter->m_value);
            tmpFitness = *(Vector_Float_Visit(((Suite *)mSuiteBase)->fitness, 0)->m_val);
            curiter = curiter->m_next;
        }
    }
}

void HostProxy::UpdateTuneIndexs(){
    std::lock_guard<std::mutex> lk(mMutex);
    auto pops_done = mIterManger.curPop;
    auto iter_done = mIterManger.curIter;
    mTuningContext.cur_iter = iter_done;
    mTuningContext.cur_point = pops_done;
    mTuningContext.tune_percentage = (float)(iter_done * mIterManger.pops + pops_done+1) / (mIterManger.gens * mIterManger.pops); 
    ++mIterManger.curPop;
    if (mIterManger.curPop == mIterManger.pops) {
        mIterManger.curPop = 0;
        ++mIterManger.curIter;
    }
}
void HostProxy::Tune() {
    if (mSuiteBase == nullptr) {
        Regist();
    }

    std::cout << "Start tune!" << std::endl;
    std::map<std::string, int> tmpParams;
    float tmpFitness = -std::numeric_limits<float>::max();
    while (!tuneOneIteration((ParamOptimizerIF *)mOptimizerIF, (Suite *)mSuiteBase,
                             (Map_StringToString)mParamsMapStringToString,
                             (Map_StringToString)mBestParamsMapStringToString, &mBestFitness)) {
        {
            if (mStopRequest) {
                mStopRequest = false;
                std::cout << "Stop requested!" << std::endl;
                break;
            }
        }
        PushCurTunedResult(tmpParams, tmpFitness);
        UpdateTuneIndexs();
        if (mTuningContext.cur_point == mIterManger.pops - 1) {
            PushBestTunedResult();
        }
        if (mConditionFunc && mConditionFunc(mTuningContext)) {
            std::lock_guard<std::mutex> lk(mMutex);
            mState = State::SUSPENDED;
        }
        std::unique_lock<std::mutex> suspendLk(mMutex);
        mCondition.wait(suspendLk, [this] { return (!(mState == State::SUSPENDED) || mStopRequest); });
    }
    PushCurTunedResult(tmpParams, tmpFitness);
    UpdateTuneIndexs();
    PushBestTunedResult();

    freeSpace((ParamOptimizerIF *)mOptimizerIF, (Map_StringToString)mParamsMapStringToString,
              (Map_StringToString)mBestParamsMapStringToString, (Suite *)mSuiteBase);
    std::lock_guard<std::mutex> lk(mMutex);
    mState = State::STOPPED;
    std::cout << "Tune over" << std::endl;
}

bool HostProxy::Start() {
    if (mState == State::RUNNING) {
        std::cout << "Host proxy: " << mName << " is already running!" << std::endl;
        return true;
    }
    if (mState == State::SUSPENDED) {
        std::cout << "Host proxy: " << mName << " has been suspended before,continue run now" << std::endl;
        std::lock_guard<std::mutex> lk(mMutex);
        mState = State::RUNNING;
        mCondition.notify_one();
        return true;
    }
    if (mState == State::STOPPED) {
        std::cout << "Tune has stopped,do you wan to restart tune?" << std::endl;
        return true;
    }
    mTuneThread = std::thread(&HostProxy::Tune, this);
    mState = State::RUNNING;
    return true;
}

bool HostProxy::Stop() {
    if (mState == State::STOPPED) {
        std::cout << "Tune has over" << std::endl;
        return true;
    }
    mStopRequest = true;
    mCondition.notify_one();
    if (mTuneThread.joinable()) {
        mTuneThread.join();
    }
    return true;
}

bool HostProxy::Suspend() {
    if (mState == State::STOPPED) {
        std::cout << "Tune has finished, so suspend is useless" << std::endl;
        return true;
    }
    std::lock_guard<std::mutex> lk(mMutex);
    mState = State::SUSPENDED;
    std::cout << "Notice tune process to suspend" << std::endl;
    mCondition.notify_one();
    return true;
}
