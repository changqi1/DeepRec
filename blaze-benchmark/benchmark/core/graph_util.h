#pragma once
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

using namespace tensorflow;
namespace benchmark {

const std::unordered_set<std::string> kDenseInputs =
    {"Placeholder", "EmbeddingLookupOp", "FeatureValueDenseOpV4", "QInfoFieldToTensorOpV2",
     "RapidEmbeddingOp", "RapidEmbeddingOpV2", "IndicatorOp"};

Status SetGraphDevice(GraphDef& gdef, int cpuid, int gpuid);

Status SetBlazeOpAttributes(const std::string& folder_path, const ConfigProto& config, GraphDef* graph_def);

Status StripUnusedNodes(const GraphDef& input_graph_def,
                        const std::vector<std::string>& output_names,
                        std::vector<std::pair<std::string, Tensor>>* inputs,
                        GraphDef* output_graph_def);

}  // namespace benchmark
