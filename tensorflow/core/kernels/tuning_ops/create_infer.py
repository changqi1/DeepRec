from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops

from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
import argparse
import random
import sys
import numpy as np
import time
import argparse


#sys.path.append('./tuning_matmul/host/lib/')
#if len(sys.argv) > 1:
  #output_file = sys.argv[1]
  #custom_op = True
#    exop = tf.load_op_library('./host_packed_matmul.so')
#else:
#    exop = tf.load_op_library('./packed_matmul.so')
#else:
#  output_file = 'test_dnn_proxy_nobias_norelu.pb'
parser = argparse.ArgumentParser()
parser.add_argument('--loop', type=int,
                    help='how many loops to run',
                    dest='loop',
                    default=100,
                    required=False)
parser.add_argument('--m', type=int,
                    help='m shape',
                    dest='m',
                    default=117,
                    required=False)
parser.add_argument('--n', type=int,
                    help='n shape',
                    dest='n',
                    default=115,
                    required=False)
parser.add_argument('--k', type=int,
                    help='k shape',
                    dest='k',
                    default=1147,
                    required=False)
parser.add_argument('--host', 
                    action="store_true",            
                    help='load host .so'
                    #dest='host',
                    #default=True,
                    )
parser.add_argument('--t', 
                    type = int,            
                    help='load threads',
                    dest='threads',
                    default=1,
                    required=False)


args = parser.parse_args()
loops = args.loop
M_ = args.m
N_ = args.n
K_ = args.k
NUM_THREADS = args.threads
if(args.host):
    exop = tf.load_op_library('./build/host_packed_matmul_op/libhost_packed_matmul.so')
    #exop = tf.load_op_library('./host_packed_matmul_src.so')
else:
    exop = tf.load_op_library('./packed_matmul.so')

ver = tf.__version__
print("TF version is: ",ver)

v2 = False
if (ver[0] == '2'):
  v2 = True

dt = tf.float32
my_seed = 0
input_data = np.random.random_sample((M_, K_)).astype(np.float32)

#weight_sizes = [[K_, N_],[N_, 1211],[1211, 1211], [1211, 1211]]
weight_sizes = [[K_, N_]]
#create two sessions to load 2 graph.
def dnn_layer(x, dt, weight_size, layer_name, with_relu, my_seed, custom_op):
    #matmul_w = tf.constant(1, shape=weight_size, dtype=dt, name=layer_name + "/matmul_w")
    init = tf.random_normal_initializer(seed=my_seed)
    matmul_w = tf.Variable(initial_value=init(shape=weight_size, dtype=tf.float32), name=layer_name + "/matmul_w")
    tf.transpose(matmul_w)
    if custom_op:
        if(args.host):
            out = exop.host_packed_matmul(x, matmul_w,[], "None", name=layer_name + "/matmul_cus")
        else:
            out = exop.packed_matmul(x, matmul_w,[], "None", name=layer_name + "/matmul_cus")
    else:
        out = tf.matmul(x, matmul_w, name=layer_name + "/matmul_tf")

    #bias_w = tf.constant(-0.1, shape=[weight_size[1]], dtype=dt, name=layer_name + "/bias_w")
    #out = tf.nn.bias_add(out, bias_w, name = layer_name + "/bias_add")
    
    #if with_relu:
    #    out = tf.nn.relu(out, name = layer_name + "/relu")
    my_seed = my_seed + 10
    return out


tf_graph = tf.Graph()
cus_graph = tf.Graph()

with tf_graph.as_default():
    if v2:
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(dt, shape=(None, weight_sizes[0][0]), name="data")
    else:
        x = tf.placeholder(dt, shape=(None, weight_sizes[0][0]), name="data")
    
    layers = len(weight_sizes)
    custom_op = False
    #import pdb;pdb.set_trace()
    for i in range(layers):
        x = dnn_layer(x, dt, weight_sizes[i], 'layer_%d' % (i + 1), False if i == layers - 1 else True,i, custom_op)
    
    output_node_name = 'softmax'
    x = tf.nn.softmax(x, name=output_node_name)    

with cus_graph.as_default():
    if v2:
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(dt, shape=(None, weight_sizes[0][0]), name="data")
    else:
        x = tf.placeholder(dt, shape=(None, weight_sizes[0][0]), name="data")
    
    layers = len(weight_sizes)
    custom_op = True
    for i in range(layers):
        x = dnn_layer(x, dt, weight_sizes[i], 'layer_%d' % (i + 1), False if i == layers - 1 else True,i, custom_op)
    
    output_node_name = 'softmax'
    x = tf.nn.softmax(x, name=output_node_name)    


with tf.compat.v1.Session(graph = tf_graph) as sess:
    input_name = 'data:0'
    output_name = 'layer_1/matmul_tf:0'
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op) 
    input_node = sess.graph.get_tensor_by_name(input_name)
    np.random.seed(1000)
    #input_data = np.random.random_sample((117, 1147)).astype(np.float32)
    #output_name = 'layer_1/matmul:0'
    for _ in range(10):
        result = sess.run(output_name, feed_dict={input_node: input_data})
    start = time.time()
    #import pdb;pdb.set_trace()
    for _ in range(loops):
        result = sess.run(output_name, feed_dict={input_node: input_data})
    end = time.time()
    print('tf_Time:',(end - start) * 1000)

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_THREADS, 
                        inter_op_parallelism_threads=1) 
#                        #allow_soft_placement=True,
#                        #device_count = {'CPU': 4})
#
#
with tf.compat.v1.Session(graph = cus_graph, config=config) as sess:
##with tf.compat.v1.Session(graph = cus_graph) as sess:
    profiler = tf.compat.v1.profiler.Profiler(sess.graph)
    input_name = 'data:0'
    output_name = 'layer_1/matmul_cus:0'
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op) 
    input_node = sess.graph.get_tensor_by_name(input_name)
    np.random.seed(1000)
    #input_data = np.random.random_sample((117, 1147)).astype(np.float32)
    #output_name = 'layer_1/matmul:0'
    for _ in range(10):
        #import pdb;pdb.set_trace()
        result = sess.run(output_name, feed_dict={input_node: input_data})
    feed_dict = {input_node: input_data};
    #run_meta = tf.compat.v1.RunMetadata()
    #opts = option_builder.ProfileOptionBuilder.time_and_memory()
    #profiler.profile_operations(options=opts)
    start = time.time()
    for i in range(loops):
        #result = sess.run(output_name, feed_dict)
        print('time: ',i)
        result = sess.run(output_name,
                          feed_dict)            
                          #options=tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                          #run_metadata=run_meta)
        #profiler.add_step(i, run_meta)

    end = time.time()
    print('cus_Time:',(end - start) * 1000)
    #tl = timeline.Timeline(run_meta.step_stats)
    #ctf = tl.generate_chrome_trace_format()
    #with open("timeline.json", 'w') as f:
    #        f.write(ctf)
    
    #'''save pb_file'''
    #if v2:
    #    tf.io.write_graph(tf.compat.v1.get_default_graph(), '.', 'cus_graph.pb', as_text=False)
    #else:
    #    #tf.train.write_graph(tf.compat.v1.get_default_graph(), '.', output_file, as_text=False)
    #    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['softmax'])
    #    with tf.gfile.FastGFile('cus_graph.pb', mode='wb') as f:
    #        f.write(output_graph_def.SerializeToString())
    #        print("Successfully saved model to %s." % 'cus_graph.pb')

