#include "benchmark/core/metrics.h"

namespace benchmark {

bool Metrics::Init() {
  auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
  if (!reporter_) {
    reporter_.reset(new cppmetrics::core::ConsoleReporter(
        registry, std::cout, boost::chrono::seconds(1)));
  }
  return true;
}

void Metrics::Start() { reporter_->start(boost::chrono::seconds(3)); }

void Metrics::Stop() {
  std::unique_lock <std::mutex> l(mu_);
  if (!stopped_) {
    stopped_ = true;
    reporter_->stop();
  }
}

void Metrics::UpdateLatency(const std::string& name, int latency) {
  cppmetrics::core::HistogramPtr lat;
  cppmetrics::core::MeterPtr thr;
  {
    std::unique_lock <std::mutex> l(mu_);
    if (latencies_.find(name) == latencies_.end()) {
      auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
      latencies_[name] = registry->histogram(name + "_latency");
      throughputs_[name] = registry->meter(name + "_throughput");
    }
    lat = latencies_[name];
    thr = throughputs_[name];
  }
  lat->update(latency);
  thr->mark();
}

void Metrics::UpdateFailures(const std::string& name) {
  cppmetrics::core::MeterPtr fail;
  {
    std::unique_lock <std::mutex> l(mu_);
    if (failures_.find(name) == failures_.end()) {
      auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
      failures_[name] = registry->meter(name + "_failures");
    }
    fail = failures_[name];
  }
  fail->mark();
}

bool Metrics::IsStopped() {
  std::unique_lock <std::mutex> l(mu_);
  return stopped_;
}

}  // namespace benchmark
