import tensorflow as tf
import sys
import os
from tensorflow.python.platform import gfile
from tensorflow.python.data.ops import dataset_ops
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from google.protobuf import text_format
from tensorflow.python.tools import optimize_for_inference_lib


batch_size = 1
input_node_name = 'import/IteratorGetNext'
output_node_name = 'import/navi_arena_ppo/global/sample_action/softmax_cross_entropy_with_logits/Reshape_2'

# def generator():
#     # for i in range(1, 10):
#     for _ in range(1):
#         yield ([1 for _ in range(batch_size)], 
#                [1 for _ in range(batch_size)],
#                [1 for _ in range(batch_size)],
#                [1 for _ in range(batch_size)],
#                [[[1 for _ in range(5123)] for _ in range(8)] for _ in range(batch_size)],
#                [[[1 for _ in range(8)] for _ in range(8)] for _ in range(batch_size)],
#                [[[1 for _ in range(90)] for _ in range(8)] for _ in range(batch_size)])

# dataset = dataset_ops.Dataset.from_generator(
#           generator, output_types=(tf.dtypes.uint64, tf.dtypes.float64, tf.dtypes.int64,
#                                    tf.dtypes.double, tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int64),
#           output_shapes=([None], [None], [None], [None], [None, 8, 5123], [None, 8, 8], [None, 8, 90]))

# dataset = dataset.batch(batch_size)
# dataset = dataset.prefetch(1)
# iterator = tf.data.Iterator.from_structure(tf.data.get_output_types(dataset), tf.data.get_output_shapes(dataset))
# next_elem = iterator.get_next()
# dataset_init_op = iterator.make_initializer(dataset)

if len(sys.argv) != 3:
    print("Usage: %s pb_file/saved_model_dir log_dir" % sys.argv[0])
    sys.exit(-1)

with tf.compat.v1.Session() as sess:
    model_filename = sys.argv[1]
    if os.path.isdir(model_filename):
        export_dir = model_filename

        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    # Is text format

    elif model_filename.endswith('pbtxt'):
        with open(model_filename) as f:
            text_graph = f.read()
            graph_def = text_format.Parse(text_graph, tf.GraphDef())
            tf.import_graph_def(graph_def)
    else:
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            print(sm.meta_graphs)
            # graph = tf.import_graph_def(sm.meta_graphs[0].graph_def)
            # graph_def = tf.get_default_graph().as_graph_def()
            # frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [output_node_name])
            # opt_graph_def = optimize_for_inference_lib.optimize_for_inference( frozen_graph_def, [input_node_name],
            #                                                                 [output_node_name], tf.float32.as_datatype_enum)
            # filename = './graph.pb'
            # graph_def_str = opt_graph_def.SerializeToString()
            # with open(filename, 'wb') as f:
            #     f.write(graph_def_str)

            iterator_op = tf.compat.v1.get_default_graph().get_operation_by_name('import/IteratorV2')

            # sess.run(tf.compat.v1.local_variables_initializer())
            # sess.run(tf.compat.v1.global_variables_initializer())
            # sess.run(iterator_op, feed_dict={})

            # next_elem = iterator.get_next()
            # print("Start to run.")
            # # print(sess.run([next_elem])) # [array([ 3.], dtype=float32)]
            # print(sess.run([graph])) # [array([ 3.], dtype=float32)]

    LOGDIR = sys.argv[2]
    train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
