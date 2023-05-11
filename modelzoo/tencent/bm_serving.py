import os
import tempfile
import pandas as pd
import tensorflow as tf
import numpy as np
import json
import requests
from time import perf_counter
import sys

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


metadata = [('x-goog-request-params', '{"profile_options": {"output": {"file_format": "json"}, "num_steps": 1}}')]

port = sys.argv[1]
channel = grpc.insecure_channel('127.0.0.1:' + port)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'tencent'
request.model_spec.signature_name = 'serving_default'

inputs_op_name = ['PlaceholderWithDefault:0', 'PlaceholderWithDefault_1:0', 'PlaceholderWithDefault_2:0', 'PlaceholderWithDefault_3:0', 'PlaceholderWithDefault_4:0', 'PlaceholderWithDefault_5:0', 'PlaceholderWithDefault_6:0', 'PlaceholderWithDefault_7:0', 'PlaceholderWithDefault_64:0', 'PlaceholderWithDefault_65:0', 'PlaceholderWithDefault_8:0', 'PlaceholderWithDefault_9:0', 'PlaceholderWithDefault_10:0', 'PlaceholderWithDefault_66:0', 'PlaceholderWithDefault_11:0', 'PlaceholderWithDefault_12:0', 'PlaceholderWithDefault_67:0', 'PlaceholderWithDefault_13:0', 'PlaceholderWithDefault_14:0', 'PlaceholderWithDefault_15:0', 'PlaceholderWithDefault_16:0', 'PlaceholderWithDefault_68:0', 'PlaceholderWithDefault_17:0', 'PlaceholderWithDefault_69:0', 'PlaceholderWithDefault_18:0', 'PlaceholderWithDefault_19:0', 'PlaceholderWithDefault_20:0', 'PlaceholderWithDefault_21:0', 'PlaceholderWithDefault_70:0', 'PlaceholderWithDefault_22:0', 'PlaceholderWithDefault_23:0', 'PlaceholderWithDefault_71:0', 'PlaceholderWithDefault_24:0', 'PlaceholderWithDefault_25:0', 'PlaceholderWithDefault_26:0', 'PlaceholderWithDefault_27:0', 'PlaceholderWithDefault_28:0', 'PlaceholderWithDefault_29:0', 'PlaceholderWithDefault_30:0', 'PlaceholderWithDefault_72:0', 'PlaceholderWithDefault_32:0', 'PlaceholderWithDefault_33:0', 'PlaceholderWithDefault_34:0', 'PlaceholderWithDefault_35:0', 'PlaceholderWithDefault_36:0', 'navi_arena_ppo/global/sample_action/play_mode:0', 'PlaceholderWithDefault_46:0', 'PlaceholderWithDefault_47:0', 'PlaceholderWithDefault_73:0', 'PlaceholderWithDefault_74:0', 'PlaceholderWithDefault_48:0', 'PlaceholderWithDefault_49:0', 'PlaceholderWithDefault_50:0', 'PlaceholderWithDefault_51:0', 'PlaceholderWithDefault_52:0', 'PlaceholderWithDefault_53:0', 'PlaceholderWithDefault_54:0', 'PlaceholderWithDefault_75:0']

inputs_feature_name = ['act_enemy_select_mask', 'act_fight_mask', 'act_move_mask', 'act_nav_select_mask', 'act_pitch_mask', 'act_pose_mask', 'act_skill_mask', 'act_yaw_mask', 'activate_weapon_id', 'allies_career_id', 'allies_pose', 'allies_space', 'allies_visible', 'ally_monster_ids', 'ally_monsters', 'ally_monsters_visible', 'ally_sub_object_ids', 'ally_sub_objects', 'ally_sub_objects_visible', 'basic', 'buff', 'career_id', 'cell', 'enemies_career_id', 'enemies_damage', 'enemies_pose', 'enemies_space', 'enemies_visible', 'enemy_monster_ids', 'enemy_monsters', 'enemy_monsters_visible', 'enemy_sub_object_ids', 'enemy_sub_objects', 'enemy_sub_objects_visible', 'enemy_team', 'env', 'fight', 'heightmap', 'hidden', 'monster_ids', 'monsters', 'monsters_visible', 'nav', 'nav_goals', 'occupy', 'play_mode', 'pose', 'skill', 'skill_ids', 'sub_object_ids', 'sub_objects', 'sub_objects_visible', 'target_enemy', 'team', 'trace_line_distance', 'track_points', 'weapon', 'weapon_ids']

inputs_feature_dtype= ['DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_FLOAT', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_BOOL', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64', 'DT_INT64', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_FLOAT', 'DT_INT64']

tf_dtype = {
    'DT_FLOAT': tf.float32,
    'DT_INT64': tf.int64,
    'DT_BOOL': tf.bool
}

bs = 1
inputs_feature_shape = [(bs, bs, 17), (bs, bs, 4), (bs, bs, 9), (bs, bs, 32), (bs, bs, 9), (bs, bs, 6), (bs, bs, 4), (bs, bs, 15), (bs, bs, 1), (bs, bs, 3), (bs, bs, 3, 6), (bs, bs, 3, 9), (bs, bs, 3), (bs, bs, 9), (bs, bs, 9, 12), (bs, bs, 9), (bs, bs, 18), (bs, bs, 18, 10), (bs, bs, 18), (bs, bs, 16), (bs, bs, 11), (bs, bs, 1), (bs, bs, 512), (bs, bs, 5), (bs, bs, 5, 2), (bs, bs, 5, 6), (bs, bs, 5, 9), (bs, bs, 5), (bs, bs, 12), (bs, bs, 12, 12), (bs, bs, 12), (bs, bs, 24), (bs, bs, 24, 10), (bs, bs, 24), (bs, bs, 2), (bs, bs, 1), (bs, bs, 7), (bs, bs, 25, 25, 4), (bs, bs, 512), (bs, bs, 3), (bs, bs, 3, 12), (bs, bs, 3), (bs, bs, 5), (bs, bs, 32, 12), (bs, bs, 3, 12), (bs), (bs, bs, 6), (bs, bs, 4, 6), (bs, bs, 4), (bs, bs, 6), (bs, bs, 6, 10), (bs, bs, 6), (bs, bs, 9), (bs, bs, 2), (bs, bs, 1, 1, 1), (bs, bs, 1, 6), (bs, bs, 4, 1), (bs, bs, 4)]

for _index, _name in enumerate(inputs_feature_name):
    _dtype = inputs_feature_dtype[_index]
    if _dtype == 'DT_FLOAT':
        dummy_input = np.random.normal(size=inputs_feature_shape[_index])
    elif _dtype == 'DT_INT64':
        dummy_input = np.random.randint(0, 4, size=inputs_feature_shape[_index])
    elif _dtype == 'DT_BOOL':
        dummy_input = np.random.choice(a=[False, True], size=inputs_feature_shape[_index])

    request.inputs[_name].CopyFrom(tf.make_tensor_proto(dummy_input, dtype=tf_dtype[_dtype]))

# print(request)
# result = stub.Predict(request, 10.0, metadata=metadata)  #timeout
# print("result", result)

def measure_latency(stub, request, iter = 1000):
    # prepare date
    latencies = []
    # warm up
    for _ in range(30):
        _ = stub.Predict(request, 10.0, metadata=metadata)
    # Timed run
    for _ in range(iter):
        start_time = perf_counter()
        _ = stub.Predict(request, 10.0, metadata=metadata)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};"

iter = sys.argv[2]
print(measure_latency(stub, request, int(iter)))
