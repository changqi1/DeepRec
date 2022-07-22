#include "benchmark/core/model.h"
#include "benchmark/core/device_util.h"
#include "benchmark/core/graph_util.h"

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

using namespace tensorflow;
namespace benchmark {

Model::PredictContext *Model::Borrow() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& context : predict_contexts_) {
    if (!context.borrowed) {
      context.borrowed = true;
      return &context;
    }
  }
  auto timeout = std::chrono::microseconds(500);
  PredictContext *res = nullptr;
  auto pred = [&]() {
    for (auto& context : predict_contexts_) {
      if (!context.borrowed) {
        context.borrowed = true;
        res = &context;
        return true;
      }
    }
    return false;
  };
  if (cond_.wait_for(lock, timeout, pred)) {
    return res;
  }
  return nullptr;
}

void Model::Return(PredictContext *predict_context) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& context : predict_contexts_) {
    if (context.session == predict_context->session) {
      context.borrowed = false;
    }
  }
  cond_.notify_one();
}

bool Model::ParseRunOptions(const std::string& run_options) {
  if (run_options.empty()) {
    LOG(WARNING) << "No run_options path configured: " << name()
                 << ", use default RunOptions.";
    return false;
  }
  Status s = ReadTextProto(Env::Default(), run_options.c_str(), &run_options_);
  if (!s.ok()) {
    s = ReadBinaryProto(Env::Default(), run_options.c_str(), &run_options_);
    if (!s.ok()) {
      LOG(ERROR) << "Read run options failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  VLOG(1) << "Read run options, " << run_options << ": " << run_options_.DebugString();
  return true;
}

bool Model::ParseSignatureDef(const std::string& meta_graph, const std::string& signature_key,
                              std::unordered_set<std::string>* existing_inputs) {
  if (meta_graph.empty()) {
    LOG(ERROR) << "No meta graph path configured: " << name();
    return false;
  }
  MetaGraphDef meta;
  Status s = ReadTextProto(Env::Default(), meta_graph.c_str(), &meta);
  if (!s.ok()) {
    s = ReadBinaryProto(Env::Default(), meta_graph.c_str(), &meta);
    if (!s.ok()) {
      LOG(ERROR) << "Parse meta graph failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  const auto& sig = meta.signature_def();
  if (sig.size() == 0) {
    LOG(ERROR) << "MetaGraphDef does not contain signature_def.";
    return false;
  }

  // if a key is not given, use the first signature in meta graph
  auto sig_iter = sig.begin();
  if (!signature_key.empty()) {
    sig_iter = sig.find(signature_key);
    if (sig_iter == sig.end()) {
      LOG(ERROR) << "Signature key " << signature_key << " not found.";
      return false;
    }
  }
  const auto& signature_def = sig_iter->second;
  for (auto iter = signature_def.outputs().begin();
       iter != signature_def.outputs().end(); ++iter) {
    const std::string& name = iter->second.name();
    output_names_.emplace_back(name);
  }
  for (auto iter = signature_def.inputs().begin();
       iter != signature_def.inputs().end(); ++iter) {
    const std::string& name = iter->second.name();
    const DataType& dtype = iter->second.dtype();
    if (dtype == DT_INVALID) {
      LOG(ERROR) << "Dtype of tensor " << name << " is not set.";
      return false;
    }
    TensorShape shape = iter->second.tensor_shape();
    Tensor tensor(dtype, shape);
    const void* data = tensor.tensor_data().data();
    memset(const_cast<void*>(data), 0, tensor.AllocatedBytes());
    inputs_.push_back({name, tensor});
    existing_inputs->emplace(name);
  }

  return true;
}

bool Model::InitSession(const std::string& config_proto) {
  if (VLOG_IS_ON(1)) DumpGraphDefToFile(name() + ".before_strip", gdef_, "dump");
  Status s = StripUnusedNodes(gdef_, output_names_, &inputs_, &gdef_);
  if (!s.ok()) {
    LOG(ERROR) << "Strip graph failed: " << name() << ", " << s.ToString();
    return false;
  }
  if (VLOG_IS_ON(1)) DumpGraphDefToFile(name() + ".after_strip", gdef_, "dump");

  // Prepare Session ConfigProto
  SessionOptions session_options = SessionOptions();
  ConfigProto* config = &session_options.config;
  if (!config_proto.empty()) {
    s = ReadTextProto(Env::Default(), config_proto.c_str(), config);
    if (!s.ok()) {
      s = ReadBinaryProto(Env::Default(), config_proto.c_str(), config);
      if (!s.ok()) {
        LOG(ERROR) << "Read config proto failed: " << name() << ", " << s.ToString()
                   << ". Use default ConfigProto.";
      }
    }
    VLOG(1) << "Read config proto: " << name() << ", " << config->DebugString();
  }
  config->set_allow_soft_placement(true);
  BlazeConfSingleton::GetInstance()->Set(*config);
  s = SetBlazeOpAttributes(graph_path_, *config, &gdef_);

  // Get cpu/gpu device counts
  int cpu_count = 1, gpu_count = 0;
  s = GetDeviceCount(*config, &cpu_count, &gpu_count);
  if (!s.ok()) {
    LOG(ERROR) << "GetDeviceCount failed: " << name() << ", " << s.ToString();
    return false;
  }
  LOG(INFO) << "GetDeviceCount: cpu_count =  " << cpu_count << ", gpu_count = " << gpu_count;

  for (int i = 0; i < this->predictor_num_; ++i) {
    // Set device id for nodes in GraphDef
    int cpuid = 0, gpuid = 0;
    if (cpu_count <= 0) {
      cpuid = 0;
    } else {
      cpuid = i % cpu_count;
    }
    if (gpu_count <= 0) {
      gpuid = -1;
    } else {
      gpuid = i % gpu_count;
    }

    Session* session = nullptr;
    std::string session_key = name() + "/CPU:" + std::to_string(cpuid) +
                              "/GPU:" + std::to_string(gpuid);
    auto iter = sessions_.find(session_key);
    if (iter == sessions_.end()) {
      if (VLOG_IS_ON(1)) DumpGraphDefToFile(name() + ".before_setdevice", gdef_, "dump");
      s = SetGraphDevice(gdef_, cpuid, gpuid);
      if (VLOG_IS_ON(1)) DumpGraphDefToFile(name() + ".after_setdevice", gdef_, "dump");
      if (!s.ok()) {
        LOG(ERROR) << "Set device failed: " << name() << ", " << s.ToString();
        return false;
      }
      s = NewSession(session_options, &session);
      if (!s.ok()) {
        LOG(ERROR) << "New session failed: " << name() << ", " << s.ToString();
        return false;
      }
      s = session->Create(gdef_);
      if (!s.ok()) {
        LOG(ERROR) << "Create session failed: " << name() << ", " << s.ToString();
        return false;
      }
      sessions_[session_key] = session;
    } else {
      session = iter->second;
    }
    PredictContext context{session, false, this};
    predict_contexts_.push_back(context);
    LOG(INFO) << "Predictor " << i << " uses session " << session_key;
  }
  return true;
}

static void NodeNameFromNamedTensor(std::string* tensor_name, std::string* node_name, int* index) {
  static const std::string arg_prefix = "_arg_";
  if (tensor_name->find(arg_prefix) == 0) {
    // name of placeholder's traced output: _arg_xxx_x_x:0, change it to xxx:0
    std::string temp = tensor_name->substr(arg_prefix.size(), tensor_name->find_last_of("_") - arg_prefix.size());
    temp = temp.substr(0, temp.find_last_of("_"));
    *tensor_name = temp + ":0";
    *node_name = temp;
    *index = 0;
    return;
  }
  std::size_t pos = tensor_name->find(":");
  *node_name = tensor_name->substr(0, pos);
  *index = atoi(tensor_name->substr(pos + 1).c_str());
}

bool Model::ParseRunmeta(const std::string& runmeta,
                         const std::unordered_set<std::string>& existing_inputs) {
  if (runmeta.empty()) {
    LOG(ERROR) << "No runmeta path configured: " << name();
    return false;
  }
  RunMetadata meta;
  Status s = ReadBinaryProto(Env::Default(), runmeta.c_str(), &meta);
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), runmeta.c_str(), &meta);
    if (!s.ok()) {
      LOG(ERROR) << "Read runmeta failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  VLOG(2) << "Runmeta: " << meta.DebugString();

  std::set<std::string> input_names;
  for (int i = 0; i < gdef_.node_size(); ++i) {
    const auto& node = gdef_.node(i);
    if (kDenseInputs.count(node.op()) != 0) {
      input_names.emplace(node.name());
    }
  }

  const auto& infos = meta.tensor_infos();
  for (int i = 0; i < infos.name_tensors_size(); ++i) {
    const auto& name_tensor = infos.name_tensors(i);
    std::string tensor_name = name_tensor.name();
    std::string node_name;
    int index = 0;
    NodeNameFromNamedTensor(&tensor_name, &node_name, &index);

    // Parse output tensors from runmeta for debug
    if (std::find(std::begin(output_names_), std::end(output_names_), tensor_name) != std::end(output_names_) ||
        std::find(std::begin(output_names_), std::end(output_names_), node_name  ) != std::end(output_names_)) {
      VLOG(1) << "Output " << tensor_name << " (traced tensor in runmeta): "
              << name_tensor.tensor().DebugString();
    }

    // Parse input tensors from runmeta
    if (existing_inputs.count(tensor_name) > 0) {
      LOG(INFO) << "Tensor " << tensor_name
                << " exists in SignatureDef, will not parse tensor from runmeta.";
      continue;
    }
    if (input_names.count(node_name) == 0) {
      continue;
    }
    if (name_tensor.tensor().dtype() == DT_VARIANT) {
      continue;
    }
    Tensor tensor;
    if (!tensor.FromProto(name_tensor.tensor())) {
      LOG(ERROR) << "Init tensor from proto failed.";
      return false;
    }
    inputs_.push_back({tensor_name, tensor});
  }

  int64 batchsize = 1;
  for (auto iter : inputs_) {
    const Tensor& t = iter.second;
    if (t.dims() >= 1) {
      int64 temp = t.dim_size(0);
      if (temp > batchsize) batchsize = temp;
    }
  }
  LOG(INFO) << "Inferred batchsize = " << batchsize << ", " << name();
  return true;
}

bool Model::LoadGraph(const std::string& frozen_graph) {
  if (frozen_graph.empty()) {
    LOG(ERROR) << "No graph path configured: " << name();
    return false;
  }
  Status s = ReadTextProto(Env::Default(), frozen_graph.c_str(), &gdef_);
  if (!s.ok()) {
    s = ReadBinaryProto(Env::Default(), frozen_graph.c_str(), &gdef_);
    if (!s.ok()) {
      LOG(ERROR) << "Read graph failed: " << name() << ", " << s.ToString();
      return false;
    }
  }

  graph_path_ = frozen_graph;
  size_t pos = graph_path_.find_last_of("/");
  if (pos != std::string::npos) {
    graph_path_ = graph_path_.substr(0, pos + 1);
  } else {
    graph_path_ = "./";
  }
  return true;
}

bool Model::Warmup() {
  for (auto context : predict_contexts_) {
    Session* session = context.session;
    std::vector<Tensor> outputs;
    RunMetadata meta;
    Status s = session->Run(run_options_, inputs_, output_names_, {}, &outputs, &meta);
    if (!s.ok()) {
      LOG(ERROR) << "Warmup: " << name() << ", Session::Run failed: " << s.ToString();
      return false;
    }
    if (outputs.size() > 0) {
      VLOG(1) << "output: " << outputs[0].DebugString();
    }
  }
  return true;
}

Model* ModelReloader::CreateObject() {
  Model *model= new Model(bench_model_config_.name(), bench_model_config_.predictor_num());
  // Load graph
  if (!model->LoadGraph(bench_model_config_.frozen_graph())) {
    LOG(ERROR) << "Load graph failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Prepare Session::Run outputs
  std::unordered_set<std::string> existing_inputs;
  if (!model->ParseSignatureDef(bench_model_config_.meta_graph(),
                                bench_model_config_.signature_key(), &existing_inputs)) {
    LOG(ERROR) << "Parse signature failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Prepare Session::Run inputs
  if (!model->ParseRunmeta(bench_model_config_.runmeta(), existing_inputs)) {
    LOG(ERROR) << "Read runmeta failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Init TensorFlow Session
  if (!model->InitSession(bench_model_config_.config_proto())) {
    LOG(ERROR) << "Init tensorflow session failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Prepare Session RunOptions
  if (!model->ParseRunOptions(bench_model_config_.run_options())) {
    LOG(ERROR) << "Parse run options failed: " << bench_model_config_.name()
               << ", use default RunOptions.";
  }

  // Warmup
  if (!model->Warmup()) {
    LOG(ERROR) << "Warmup failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  LOG(INFO) << "Init and warmup model complete: " << bench_model_config_.name();
  return model;
}

bool ModelSelector::InitModel(
    const benchmark::BenchModelConfig& bench_model_config) {
  std::shared_ptr<ModelReloader> model_reloader =
      std::make_shared<ModelReloader>(bench_model_config);
  bool success = model_reloader->Switch();
  if (!success) {
    return false;
  }
  model_reloaders_.emplace_back(model_reloader);
  switch_interval_.emplace_back(bench_model_config.switch_interval());
  return true;
}

std::shared_ptr<Model> ModelSelector::GetModel(int idx) const {
  auto model_reloader = model_reloaders_[idx];
  return model_reloader->Instance();
}

void ModelSelector::Start() {
  running_ = true;
  std::vector<int> left_time_to_switch(switch_interval_);
  while (running_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (int i = 0; i < left_time_to_switch.size(); ++i) {
      left_time_to_switch[i]--;
      if (left_time_to_switch[i] <= 0) {
        LOG(INFO) << "Begin switch model.";
        bool success = model_reloaders_[i]->Switch();
        if (!success) {
          LOG(ERROR) << "Switch model failed.";
          continue;
        }
        LOG(INFO) << "Switch model successfully.";
        left_time_to_switch[i] = switch_interval_[i];
      }
    }
  }
}

void ModelSelector::Stop() { running_ = false; }

}  // namespace benchmark
