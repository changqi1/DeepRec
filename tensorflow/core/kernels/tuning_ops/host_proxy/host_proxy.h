#ifndef __HOST_PROXY_H__
#define __HOST_PROXY_H__

#include <functional>
#include <map>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

struct HostParam {
    const std::string name;
    std::pair<int, int> min_max_val;
};

struct TuningContext {
    int cur_iter;
    int cur_point;
    float tune_percentage;
    float cur_tune_fitness = std::numeric_limits<float>::max();
    float cur_best_fitness = std::numeric_limits<float>::max();
    std::map<std::string, int> cur_best_params;
    std::map<std::string, int> cur_tune_params;
};

struct IterManger {
    int gens = 10;
    int pops = 10;
    int curIter = 0;
    int curPop = -1;
};

class HostProxy {
  public:
    using Ptr = std::shared_ptr<HostProxy>;

    enum class State {
        UNINITIALIZED = -1, /**< Default state */
        INITIALIZED = 0,    /**< Ready to run  */
        RUNNING = 1,        /**< Tuning in progress */
        SUSPENDED = 2,      /**< Halt during tuning, can be resumed */
        STOPPED = 3         /**< Tuning completed */
    };

    typedef std::function<const float(std::string const &name,std::vector<int> const &param)> EvaluateFunc;
    typedef std::function<const bool(TuningContext &context)> ConditionFunc;
    HostProxy(const char *name);
    virtual ~HostProxy();

    HostProxy(HostProxy &) = delete;
    HostProxy(HostProxy &&) = delete;
    void operator=(const HostProxy &) = delete;

    void SetParamter(const char *var_name, const int var_min, const int var_max);
    std::vector<HostParam> &GetParamters();
    int GetParamsNums();

    State GetProxyState();

    void SetEvaluateFunc(EvaluateFunc evaluate);
    EvaluateFunc &GetEvaluateFunc();
    void SetAlgorithm(const char *algo,int gens,int pops);
    void SetConditionFunc(ConditionFunc condition);
    ConditionFunc &GetConditionFunc();
    std::string GetName();
    void UpdateTuneIndexs();
    void PushCurTunedResult(std::map<std::string, int> &tmpParams, float &tmpFitness);
    void PushBestTunedResult();
    std::map<std::string, int> GetTunedResult();
    float GetBestCost();
    TuningContext GetTuningContext();

    void Tune();

    void Regist();

    bool Start();

    bool Stop();

    bool Suspend();



  private:
    int paramNums;
    int mTotalIters;
    float mBestFitness;
    std::string mName;
    std::chrono::milliseconds mMaxTimeCostMs = std::chrono::milliseconds(-1);
    EvaluateFunc mEvaluateFunc;
    ConditionFunc mConditionFunc = NULL;
    TuningContext mTuningContext;
    int mAlgo;
    std::vector<HostParam> mTuneParams;
    std::map<std::string, int> mTunedResult;
    std::thread mTuneThread;
    State mState = State::UNINITIALIZED;
    std::condition_variable mCondition;
    std::mutex mMutex;
    std::atomic<bool> mStopRequest{false};
    IterManger mIterManger;
    void *mSuiteBase = nullptr;
    void *mOptimizerIF = nullptr;
    void *mParamsMapStringToString = nullptr;
    void *mBestParamsMapStringToString = nullptr;
};

#endif