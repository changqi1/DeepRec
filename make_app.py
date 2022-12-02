# coding: utf-8
import json
import os
import traceback
from abc import ABC
import tornado.web
from ourmodel import AlimamaYolo

num_processes = 20
MAX_WORKERS = 1




def on_system_start():
    # download models
    model_local_paths = []
    return model_local_paths

def on_process_start():
    # 这个函数每个CPU进程会执行一次，所以要把下载模型的代码移出
    # 这里只处理改CPU进程需要用到的内容
    global model
    model = AlimamaYolo()

class MainHandler(tornado.web.RequestHandler, ABC):
    def get(self):
        # 业务逻辑，获取url中的信息进行计算，返回
        image_url = self.get_query_argument("image_url", "")
        #result = {
        #}
        #result_dict = json.loads(result[consts.RESULT_JSON]) 
        #if result_dict["status"] == -1:
        #    raise Exception(result_dict["errMsg"])
        #output = json.dumps(result)
        return output
    def post(self):
        global model
        input = self.get_body_argument("input","")
        result = model.predict(input_str=input)
        self.write("success")

class ReadinessHandler(tornado.web.RequestHandler, ABC):
    def get(self):
        return self.write("OK")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/readiness", ReadinessHandler)
    ])

