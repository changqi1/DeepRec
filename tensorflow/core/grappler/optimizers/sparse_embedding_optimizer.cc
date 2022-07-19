/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/optimizers/sparse_embedding_optimizer.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {


DataType GetInputType(GraphProperties& properties, const NodeDef* node_view, int i_th) {

    const string &name = node_view->name();
    const auto& input = properties.GetInputProperties(name);

    if (i_th + 1 > input.size()) {
        LOG(WARNING) << "Cannot get data type for input " << i_th << " of node " << name;
        return DT_INT64;
    }

    const DataType type = input[i_th].dtype();
    return type;
}

// Seems cannot get shape during the optimization, carefully to call it
int GetOutputDims(GraphProperties& properties, const string& node_name) {
  printf("Try to get shape\n");
  const std::vector<OpInfo::TensorProperties>& prop_list =
      properties.GetOutputProperties(node_name);
  const OpInfo::TensorProperties& props = prop_list[0];
  TensorShape shape(props.shape());
  printf("Dims is %d\n", shape.dims());
  return shape.dims();
}

string get_node_by_tensor(string tensor_name) {
    auto position = tensor_name.find(":");
    if ( position != string::npos )
        tensor_name.erase(tensor_name.begin() + position, tensor_name.end());
    if ( tensor_name[0] == '^' )
        tensor_name.erase(tensor_name.begin());

    return tensor_name;
}

NodeDef* skip_identity(std::unordered_map<string, NodeDef*>& node_mapping, string node_name) {
    NodeDef* node = node_mapping[node_name];
    while ( node->op() == "Identity" ) {
        node_name = get_node_by_tensor(node->input(0));
        node = node_mapping[node_name];
    }
    return node;
}

NodeDef* find_output_node(std::unordered_map<string, std::vector<string>>& node_outputs,
                    std::unordered_map<string, NodeDef*>& node_mapping,
                    string node_name, std::vector<string>& target_ops) {
    if( node_outputs.count(node_name) == 0 ) return NULL;

    auto nodes = node_outputs[node_name];
    for ( int i = 0; i < nodes.size(); i ++ ) {
        string outnode_name = nodes.at(i);
        NodeDef* tmp_outnode = node_mapping[outnode_name];
        if ( find(target_ops.begin(), target_ops.end(), tmp_outnode->op()) != target_ops.end() )
            return tmp_outnode;
    }
    return NULL;
}

Status SparseEmbeddingOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
    *optimized_graph = item.graph;
    const char* env_p = std::getenv("EC_FUSE");
    if (env_p != NULL && env_p[0] == '0') {
        LOG(INFO) << "Ignore the optimization as EC_FUSE=0";
        return Status::OK();
    }

    printf("\n [Begin] - SparseEmbeddingOptimizer.\n");
    std::vector<string> sparse_segment_nodes;
    string node_device = "";
    for (auto& node : *optimized_graph->mutable_node()) {
        string node_op = node.op();
        // Record all nodes with 'SparseSegmentSum' or 'SparseSegmentMean' type
        if ( node_op == "SparseSegmentSum" || node_op == "SparseSegmentMean" ) {
            string node_name = node.name();
            sparse_segment_nodes.push_back(node_name);
            node_device = node.device();
        }
    }
    if ( sparse_segment_nodes.size() == 0 ) {
        printf(" [Over] - Target node is not found.\n");
        return Status::OK();
    }

    printf("    [Running] - Discovered %d subgraphs to be merged.\n", sparse_segment_nodes.size());

    // collect the common information
    std::unordered_map<string, std::vector<string>> node_outputs;
    std::unordered_map<string, NodeDef*> node_mapping;
    for (auto& node : *optimized_graph->mutable_node()) {
        string node_name = node.name();
        node_mapping.insert(std::make_pair(node_name, &node));

        int input_num = node.input_size();
        if ( input_num == 0 ) continue;

        for ( int i = 0; i < input_num; i ++ ) {
            string prenode_name = get_node_by_tensor(node.input(i));

            if( node_outputs.count(prenode_name) > 0 ) {
                node_outputs[prenode_name].push_back(node_name);
            } else {
                std::vector<string> tmp_vector = {node_name};
                node_outputs.insert(std::make_pair(prenode_name, tmp_vector));
            }
        }
    }

    // fuse
    GraphProperties properties(item);
    properties.InferStatically(false);

    int optimizer_idx = 0;
    for ( int i = 0; i < sparse_segment_nodes.size(); i ++ ) {
        // Get the embedding node and its weight
        string combiner_input_node = get_node_by_tensor(node_mapping[sparse_segment_nodes[i]]->input(0));
        NodeDef* gather_node = skip_identity(node_mapping, combiner_input_node);
        if ( gather_node->op() != "GatherV2" && gather_node->op() != "ResourceGather" ) {
            printf(" [Over] - Cannot find gather_node for %s.\n", sparse_segment_nodes[i].c_str());
            return Status::OK();
        }

        string weight_name = gather_node->input(0);

        // Get the real input
        // Definition of SparseFillEmptyRows:
        //SparseFillEmptyRows(
        //  const ::tensorflow::Scope & scope,
        //  ::tensorflow::Input indices,
        //  ::tensorflow::Input values,
        //  ::tensorflow::Input dense_shape,
        //  ::tensorflow::Input default_value
        //)
        NodeDef* unique_node = node_mapping[get_node_by_tensor(gather_node->input(1))];
        if ( unique_node->op() != "Unique" ) {
            printf(" [Over] - Unique node is not detected for %s.\n", gather_node->name().c_str());
            return Status::OK();
        }

        NodeDef* sparse_fill_node = node_mapping[get_node_by_tensor(unique_node->input(0))];
        if ( sparse_fill_node->op() != "SparseFillEmptyRows" ) {
            printf(" [Over] - Cannot find SparseFillEmptyRows for %s.\n", gather_node->name().c_str());
            return Status::OK();
        }

        NodeDef* value_node = node_mapping[get_node_by_tensor(sparse_fill_node->input(1))];
        if ( value_node->op() != "GatherV2" ) {
            printf(" [Over] - Value not found for SparseFillEmptyRows: %s.\n", sparse_fill_node->name().c_str());
            return Status::OK();
        }

        string input_name = value_node->input(0);

        // Get the dense shape and indice which have value
        string dense_shape_name = get_node_by_tensor(sparse_fill_node->input(2));
        NodeDef* reshape_node = skip_identity(node_mapping, dense_shape_name); // dense_shape comes from SparseReshape
        if ( reshape_node->op() != "SparseReshape" ) {
            printf(" [Over] - Cannot find reshape node for %s.\n", gather_node->name().c_str());
            return Status::OK();
        }

        string indice_tensor = reshape_node->input(0);
        string dense_shape_tensor = reshape_node->input(1);

        // Find out where the embedding ends
        std::vector<string> target_ops = {"ZerosLike"};
        NodeDef* zeros_like_node = find_output_node(node_outputs, node_mapping, sparse_segment_nodes[i], target_ops);
        if ( !zeros_like_node ) {
            printf(" [Over] - Cannot find ZerosLike node for %s.\n", sparse_segment_nodes[i].c_str());
            return Status::OK();
        }

        target_ops = {"Select"};
        NodeDef* select_node = find_output_node(node_outputs, node_mapping, zeros_like_node->name(), target_ops);
        if ( !select_node ) {
            printf(" [Over] - Cannot find Select node for %s.\n", zeros_like_node->name().c_str());
            return Status::OK();
        }

        target_ops = {"Reshape"};
        NodeDef* reshape_node1 = find_output_node(node_outputs, node_mapping, select_node->name(), target_ops);
        if ( !reshape_node1 ) {
            printf(" [Over] - Cannot find Reshape node for %s.\n", select_node->name().c_str());
            return Status::OK();
        }
        NodeDef* reshape_node2 = find_output_node(node_outputs, node_mapping, reshape_node1->name(), target_ops);
        if ( reshape_node2 ) // If follows another reshape
            reshape_node1 = reshape_node2;

        string embedding_end_at = reshape_node1->name();
        NodeDef* combiner_node = node_mapping[sparse_segment_nodes[i]];

        // Create the new node
        string fuse_node_name = reshape_node1->name() + "_fused";
        DataType Tweight_value = GetInputType(properties, gather_node, 0);
        DataType Tshape_value = GetInputType(properties, reshape_node, 1);
        DataType Tid_value = GetInputType(properties, value_node, 0);
        int Combiner_value = 0;
        if ( combiner_node->op() == "SparseSegmentMean" ) Combiner_value = 1;

        NodeDef* fuse_node = optimized_graph->add_node();
        fuse_node->set_op("SparseEmbeddingWithShape");
        fuse_node->set_name(fuse_node_name);
        fuse_node->set_device(node_device);
        fuse_node->add_input(weight_name);
        fuse_node->add_input(input_name);
        fuse_node->add_input(dense_shape_tensor);
        fuse_node->add_input(indice_tensor);
        (*fuse_node->mutable_attr())["Tweight"].set_type(Tweight_value);
        (*fuse_node->mutable_attr())["Tshape"].set_type(Tshape_value);
        (*fuse_node->mutable_attr())["Tid"].set_type(Tid_value);
        (*fuse_node->mutable_attr())["Combiner"].set_i(Combiner_value);

        if( node_outputs.count(embedding_end_at) == 0 ) {
            printf(" [Over] - %s has no output.", embedding_end_at.c_str());
            return Status::OK();
        }

        // Replace the input node to the new node
        auto nodes = node_outputs[embedding_end_at];
        for ( int i = 0; i < nodes.size(); i ++ ) {
            string stop_nodename = nodes.at(i);
            for (auto& node : *optimized_graph->mutable_node()) {
                if ( node.name() != stop_nodename ) continue;

                for ( int i = 0; i < node.input_size(); i ++ ) {
                    // Replace the old embedding node with the new one
                    if ( node.input(i) == embedding_end_at ) {
                        node.set_input(i, fuse_node_name);
                        break;
                    }
                }

                break;
            }
        }

        optimizer_idx ++;
    }

    printf(" [Done] - SparseEmbeddingOptimizer.\n");
    return Status::OK();
}

void SparseEmbeddingOptimizer::Feedback(Cluster* /*cluster*/,
                                const GrapplerItem& /*item*/,
                                const GraphDef& /*optimized_graph*/,
                                double /*result*/) {
    // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // namespace tensorflow
