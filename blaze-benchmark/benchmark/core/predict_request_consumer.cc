#include "benchmark/core/predict_request_consumer.h"
#include <string>
#include <fstream>

using namespace tensorflow;
namespace benchmark {

PredictRequestConsumer::PredictRequestConsumer(
    benchmark::ModelSelector *model_selector,
    benchmark::PredictRequestQueue *predict_queue,
    benchmark::Metrics *metrics, int max_queue_size) {
  model_selector_ = model_selector;
  predict_queue_ = predict_queue;
  metrics_ = metrics;
  max_queue_size_ = max_queue_size;
}

void PredictRequestConsumer::Start() {
  while (!metrics_->IsStopped()) {
    PredictRequest *predict_request = predict_queue_->Dequeue();
    if (!predict_request) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }

    int model_idx = predict_request->model_idx;
    std::shared_ptr<Model> model = model_selector_->GetModel(model_idx);
    if (!model) {
      LOG(ERROR) << "model_idx: " << model_idx << " out of range.";
      return;
    }
    if (max_queue_size_ > 0 && predict_queue_->size() > max_queue_size_) {
      metrics_->UpdateFailures(model->name());
      VLOG(2) << "Drop request: number of outstanding requests exceeds max_queue_size.";
      continue;
    }
    Model::PredictContext *predict_context = model->Borrow();
    if (!predict_context) {
      predict_queue_->Enqueue(predict_request);
      continue;
    }
    auto bef = std::chrono::high_resolution_clock::now();
    this->PredictImpl(predict_context);
    auto aft = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(aft - bef).count();
    metrics_->UpdateLatency(model->name(), dur);
    model->Return(predict_context);
  }
}

bool PredictRequestConsumer::PredictImpl(
    benchmark::Model::PredictContext *predict_context) {
  auto session = predict_context->session;
  std::vector<Tensor> outputs;
  RunMetadata meta;
  auto s = session->Run(predict_context->parent->run_options(), predict_context->parent->inputs(),
                        predict_context->parent->output_names(), {}, &outputs, &meta);
  if (!s.ok()) {
    LOG(ERROR) << "Session::Run failed: " << s.ToString();
    return false;
  }
  std::vector<std::string> output_names = predict_context->parent->output_names();
  if (outputs.size() != output_names.size()) {
    LOG(ERROR) << "Error: output numbers mismatch.";
    return false;
  }
  for (int i = 0; i < outputs.size(); i++) {
    TensorProto proto;
    outputs[i].AsProtoField(&proto);
    VLOG(1) << "Output " << output_names[i] << " (output of session::run): "<< proto.DebugString();
  }
  {
    static int _count = 1;
    if(_count == 2500){
      std::string outfile = "serialized";
      meta.step_stats().SerializeToString(&outfile);
      std::ofstream ofs("timeline" + std::to_string(_count));
      ofs << outfile;
      ofs.close();
    }
    _count++;
  }
  return true;
}

}  // namespace benchmark
