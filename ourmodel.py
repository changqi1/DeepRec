#!/usr/bin/python
#****************************************************************#
# ScriptName: ourmodel.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-12-02 17:24
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-12-02 17:24
# Function: 
#***************************************************************#

import argparse
import torch
import cv2
import urllib.request
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
import torch.backends.cudnn as cudnn
import base64
import ast

torch.set_num_threads(1)
cv2.setNumThreads(1)

class AlimamaYolo(object):
    def __init__(self):
        # settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5m.pt', help='model.pt path(s)')
        parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        opt = parser.parse_args()
        # device initialize
        device = select_device(opt.device)
        # Load model
        model = attempt_load(weights=opt.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.half = device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            model.half()  # to FP16

        data = torch.rand(1, 3, 640, 640)

        import intel_extension_for_pytorch as ipex
        from intel_extension_for_pytorch.quantization import prepare, convert

        qconfig = ipex.quantization.default_static_qconfig
        prepared_model  = prepare(model, qconfig, example_inputs=data, inplace=False)
        prepared_model(data)
        convert_model = convert(prepared_model)

        with torch.no_grad():
            traced_model = torch.jit.trace(convert_model, data)
            traced_model = torch.jit.freeze(traced_model)

        # traced_model.save("quantized_model.pt")

        self.model = traced_model
        self.opt = opt
        self.stride = stride

    def predict(self, input_str):
        #input_str is base64 string
        data = ast.literal_eval(input_str)
        base64_data = data['imageBase64']
        image = base64.b64decode(base64_data)
        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        h, w, c = image.shape

        # Padded resize Convert
        img = letterbox(image, self.opt.img_size, auto=False, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img)#.to(device)
        img = img.float()  # uint8 to fp16/32/bf16
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # run inference
        t1 = time_synchronized()
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres)
        t2 = time_synchronized()

        print('time(s): {}'.format(t2 - t1))
        #print(pred)
        return pred

if __name__ == "__main__":
    import json
    model = AlimamaYolo()
    fname = "china_flag.txt"
    with open(fname) as f:
        imageBytes = f.readline()
    data = {'imageBase64': imageBytes, 'wordSize': '1'}
    model.predict(json.dumps(data))
