import tensorflow as tf
from tensorflow_core.python.client import timeline
from tensorflow.core.framework.step_stats_pb2 import StepStats

import os
import json


def load_timeline(path: str = "timeline.json"):
    context: str = ""
    with open(path, "r") as fp:
        context = fp.read()

    if context == "":
        return "error load"

    json_obj = json.loads(context)
    print(json_obj["traceEvents"])
    for _item in json_obj["traceEvents"]:
        if _item["name"] == "unknown" and \
                _item["args"]["op"] == "unknown":
            _tmp_str: str = _item["args"]["name"]
            if ":" in _tmp_str:
                _strs = _tmp_str.split(":")
                _item["name"] = _strs[0]
                _item["args"]["op"] = _strs[1]

    with open(path, "w") as fp:
        json.dump(json_obj, fp)


def read_step_state(step_states, output_dir):
    print(step_states)
    step_stats = StepStats()
    with open(step_states, "rb") as f:
        step_stats.ParseFromString(f.read())
    x = timeline.Timeline(step_stats).generate_chrome_trace_format()
    with open(output_dir, 'w') as outfile:
        outfile.write(x)
		
if __name__ == "__main__":
    file_name = "timeline50"
    json_file = "timeline.json"
    read_step_state(file_name, json_file)
    load_timeline(json_file)
