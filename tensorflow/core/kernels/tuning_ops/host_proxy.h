#ifndef __HOST_PROXY_H__
#define __HOST_PROXY_H__

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct HostParam {
    const std::string name;
    std::pair<int, int> min_max_val;
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

    typedef std::function<const float(std::vector<int> const &param)> EvaluateFunc;

    HostProxy(const char *name);
    virtual ~HostProxy();

    HostProxy(HostProxy &) = delete;
    HostProxy(HostProxy &&) = delete;
    void operator=(const HostProxy &) = delete;

    void SetParamter(const char *var_name, const int var_min, const int var_max);
    std::vector<HostParam> &getParamters();

    void SetEvaluateFunc(EvaluateFunc evaluate);
    EvaluateFunc &getEvaluateFunc();

    void SetHaltTuingConditions(const int max_tuning_time_ms, const int max_tuning_rounds);

    std::map<std::string, int> GetTunedResult();

    bool Start();

    bool Stop();

    bool Suspend();

  private:
    void *mSuiteBase;
    float mBestFitness;
    State mState = State::UNINITIALIZED;
    std::string mName;

    std::chrono::milliseconds mMaxTimeCostMs = std::chrono::milliseconds(-1);
    EvaluateFunc mEvaluateFunc;
    std::vector<HostParam> mTuneParams;
    std::map<std::string, int> mTunedResult;
};

#endif