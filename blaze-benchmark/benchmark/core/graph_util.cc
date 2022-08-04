#include "benchmark/core/graph_util.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"

#include "benchmark/core/node_util.h"

#if USE_CUDA
#include <nvml.h>
#endif  // USE_CUDA

using namespace tensorflow;

namespace benchmark {

const char* const kGPUDevice = "/device:GPU:";
const char* const kCPUDevice = "/device:CPU:";
const char* const kBlazeAttrConf = "blaze_option_path";
const char* const kBlazeAttrGraph = "graph_def";
const char* const kBlazeKernelName = "BlazeXlaOp";
const char* const kBlazeRealDevice = "_blaze_real_device";
const char* const kDeepCopySuffix = "_deep_copy";

template <class T>
inline void SetNodeAttr(const std::string& key, const T& value, NodeDef* node) {
  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  auto* attr_map = node->mutable_attr();
  (*attr_map)[key] = attr_value;
}


Status SetGraphDevice(GraphDef& graph_def, int cpuid, int gpuid) {
  if (cpuid < 0) {
    return errors::InvalidArgument("Invalid cpuid: ", cpuid);
  }
  for (int i = 0; i < graph_def.node_size() ;i++) {
    NodeDef* node = graph_def.mutable_node(i);
    std::string device_name;
    std::string cpu_device = kCPUDevice + std::to_string(cpuid);
    std::string gpu_device = kGPUDevice + std::to_string(gpuid);
    if (gpuid < 0) {
      // if gpu is not present, place all nodes to cpu
      device_name = cpu_device;
    } else if (node->device().empty()) {
      // place nodes to gpu by default
      device_name = gpu_device;
    } else {
      // change cpu/gpu id based on device numbers
      DeviceNameUtils::ParsedName device;
      if (!DeviceNameUtils::ParseFullName(node->device(), &device)) {
        return errors::InvalidArgument("invalid device ", node->name(), " ", node->device());
      }
      if (device.type == "GPU") {
        device.id = gpuid;
      } else if (device.type == "CPU") {
        device.id = cpuid;
      }
      device_name = DeviceNameUtils::ParsedNameToString(device);
    }
    node->set_device(device_name);
    if (node->op() == kBlazeKernelName) {
      std::string temp = gpu_device;
      if (gpuid == -1) temp = cpu_device;
      *(((*(node->mutable_attr()))[kBlazeRealDevice]).mutable_s()) = temp;
    }
  }
  return Status::OK();
}

Status SetBlazeOpAttributes(const std::string& folder_path, const ConfigProto& config, GraphDef* graph_def) {
  for (int i = 0; i < graph_def->node_size(); i++) {
    NodeDef* node = graph_def->mutable_node(i);
    if (node->op() == kBlazeKernelName) {
      auto attr = node->mutable_attr()->find(kBlazeAttrGraph);
      if (attr == node->mutable_attr()->end()) {
        return errors::Internal("Blaze node ", node->DebugString(),
                                " do not have attr ", kBlazeAttrGraph);
      }
      std::string graph_path = folder_path + "/" + node->attr().at(kBlazeAttrGraph).s();
      SetNodeAttr(kBlazeAttrGraph, graph_path, node);
      attr = node->mutable_attr()->find(kBlazeAttrConf);
      if (attr == node->mutable_attr()->end()) {
        return errors::Internal("Blaze node ", node->DebugString(),
                                " do not have attr ", kBlazeAttrConf);
      }
      auto options = config.blaze_options();
      SetNodeAttr(kBlazeAttrConf, options.DebugString(), node);
    }
  }
  return Status::OK();
}

void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def) {
  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    if (selector(node)) {
      *output_graph_def->mutable_node()->Add() = node;
    }
  }
}

void RedirectEdgesInGraphDef(const std::set<std::string>& input_nodes,
                             std::vector<std::pair<std::string, Tensor>>* inputs,
                             GraphDef* graph_def) {
  for (auto& iter : *inputs) {
    iter.first = NewNodeNameFromInput(iter.first);
  }
  for (NodeDef& node : *(graph_def->mutable_node())) {
    for (std::string& input_name : *(node.mutable_input())) {
      std::string input_node = NodeNameFromInput(input_name);
      // std::string input_copy = NewNodeNameFromInput(input_name) + kDeepCopySuffix;
      std::string input_copy = NewNodeNameFromInput(input_name);
      if (input_nodes.count(input_node) && node.name() != input_copy) {
        input_name = input_copy;
      }
    }
  }
}

Status AddSubGraphDef(const std::string& folder_path, const NodeDef& node, GraphDef* graph_def) {
  if (!HasNodeAttr(node, kBlazeAttrGraph)) {
    return errors::Internal("Blaze node ", node.DebugString(),
                            " do not have attr ", kBlazeAttrGraph);
  }

  GraphDef sub_graph_def_;
  std::string graph_path = folder_path + "/" + node.attr().at(kBlazeAttrGraph).s();
  if (!ReadTextProto(Env::Default(), graph_path, &sub_graph_def_).ok()) {
    if (!ReadBinaryProto(Env::Default(), graph_path, &sub_graph_def_).ok()) {
      LOG(ERROR) << "Parse graph from " << graph_path << " failed";
      return errors::Internal("Parse graph from ", graph_path, " failed");
    }
  }

  std::map<std::string, std::string> inputs_lookup;
  if (HasNodeAttr(node, "input_names")) {
    auto list_value = node.attr().at("input_names").list();
    if(list_value.s_size()!=node.input_size()){
      LOG(ERROR) << "Parse graph from " << graph_path << " failed, inputs number is unmatch" << "node: " << node.DebugString();
      return errors::Internal("Parse graph from ", graph_path, " failed, inputs number is unmatch");
    }

    for(int i=0; i<list_value.s_size(); ++i){
      inputs_lookup[list_value.s(i)] = node.input(i);
    }
  }

  for (NodeDef n : sub_graph_def_.node()) {
    for(int i=0; i<n.input_size(); ++i){
      if(inputs_lookup.count(n.input(i))<=0){
        continue;
      }
      n.set_input(i, inputs_lookup[n.input(i)]);
    }

    *(graph_def->mutable_node()->Add()) = n;
  }

  return Status::OK();
}

// replace BlazeKernel OP
Status ReplaceSubGraphDef(const std::string& folder_path, GraphDef* graph_def, std::vector<std::string>& output_names) {
  for (const NodeDef& node : graph_def->node()) {
  if (node.op() == kBlazeKernelName) {
      Status s = AddSubGraphDef(folder_path, node, graph_def);
      if (!s.ok()) {
        LOG(ERROR) << "replace BlazeXlaOP failed: " << s.ToString();
        return errors::Internal("replace BlazeXlaOP failed: ", s.ToString());
      }

      for(std::string& output_name : output_names){
        std::string prefix;
        std::string node_name;
        std::string suffix;
        NodeNamePartsFromInput(output_name, &prefix, &node_name, &suffix);

        if(node.name()==node_name){
          //TODO only one output now
          std::string output_node_name = node.attr().at("output_names").list().s(0);
          output_name = prefix + output_node_name + suffix;
        }
      }
    }
  }

  GraphDef filtered_graph_def;
  FilterGraphDef(*graph_def,
                 [&](const NodeDef& node) {
                   return node.op() != kBlazeKernelName;
                 },
                 &filtered_graph_def);

  graph_def->Clear();
  for (const NodeDef& node : filtered_graph_def.node()) {
    *(graph_def->mutable_node()->Add()) = node;
  }

  return Status::OK();
}

Status StripUnusedNodes(const GraphDef& input_graph_def,
                        const std::vector<std::string>& output_names,
                        std::vector<std::pair<std::string, Tensor>>* inputs,
                        GraphDef* output_graph_def) {
  std::set<std::string> required_nodes;
  std::set<std::string> input_nodes;
  std::unordered_map<std::string, std::vector<std::pair<std::string, DataType>>> node_tensor_map;
  for (auto& iter : *inputs) {
    std::string tensor_name = iter.first;
    std::string node_name = NodeNameFromInput(iter.first);
    required_nodes.insert(node_name);
    input_nodes.insert(node_name);
    if (node_tensor_map.find(node_name) == node_tensor_map.end()) {
      std::vector<std::pair<std::string, DataType>> temp;
      node_tensor_map[node_name] = temp;
    }
    node_tensor_map[node_name].emplace_back(std::make_pair(tensor_name, iter.second.dtype()));
  }
  for (const std::string& output : output_names) {
    required_nodes.insert(output);
  }

  std::map<std::string, const NodeDef*> node_lookup;
  MapNamesToNodes(input_graph_def, &node_lookup);

  std::vector<std::string> current_inputs;
  for (const std::string& output_name : output_names) {
    current_inputs.push_back(NodeNameFromInput(output_name));
  }

  while (!current_inputs.empty()) {
    std::set<std::string> next_inputs;
    for (const std::string& current_input : current_inputs) {
      required_nodes.insert(current_input);
      if (input_nodes.count(current_input)) {
        continue;
      }
      if (!node_lookup.count(current_input)) {
        return errors::InvalidArgument("Input node ", current_input,
                                       " not found in graph");
      }
      const NodeDef* current_node = node_lookup[current_input];
      for (const std::string& input_name : current_node->input()) {
        std::string input_node_name = NodeNameFromInput(input_name);
        if (!required_nodes.count(input_node_name)) {
          next_inputs.insert(input_node_name);
        }
      }
    }
    current_inputs =
        std::vector<std::string>(next_inputs.begin(), next_inputs.end());
  }

  GraphDef filtered_graph_def;

  FilterGraphDef(input_graph_def,
                 [&](const NodeDef& node) {
                   return required_nodes.count(node.name()) > 0;
                 },
                 &filtered_graph_def);

  output_graph_def->Clear();
  for (const NodeDef& node : filtered_graph_def.node()) {
    if (input_nodes.count(node.name())) {
      for (auto iter : node_tensor_map[node.name()]) {
        NodeDef placeholder_node;
        placeholder_node.set_op("Placeholder");
        placeholder_node.set_name(NewNodeNameFromInput(iter.first));
        SetNodeAttr("dtype", iter.second, &placeholder_node);
        *(output_graph_def->mutable_node()->Add()) = placeholder_node;

        // Optimize GPU memcpy: add DeepCopy after input to allocate pinned memory and do async h2d memcpy
        // NodeDef copy_node;
        // copy_node.set_op("DeepCopy");
        // copy_node.set_name(NewNodeNameFromInput(iter.first) + kDeepCopySuffix);
        // SetNodeAttr("T", iter.second, &copy_node);
        // copy_node.add_input(placeholder_node.name());
        // copy_node.set_device("/device:CPU:0");
        // *(output_graph_def->mutable_node()->Add()) = copy_node;
      }
    } else {
      *(output_graph_def->mutable_node()->Add()) = node;
    }
  }
  RedirectEdgesInGraphDef(input_nodes, inputs, output_graph_def);

  for (const NodeDef& node : output_graph_def->node()) {
    if (kDenseInputs.count(node.op()) != 0 && node.op() != "Placeholder") {
      return errors::Internal(node.op(),
          " is not stripped (may because runmeta is invalid and does not include the op's output tensors).");
    }
  }

  return Status::OK();
}

}  // namespace benchmark
