#!/usr/bin/python
#****************************************************************#
# ScriptName: ourmodel.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-12-02 17:24
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-12-02 17:24
# Function: 
#***************************************************************#
import os
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


def profile(model, inp):
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=-1))
        prof.export_chrome_trace(f"profile_json/{prof.step_num}.json")
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=3),
            # on_trace_ready=trace_handler,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    ) as p:
        for i in range(20):
            pred = model(inp)[0]
            p.step()
        return pred

class AlimamaYolo(object):
    def __init__(self):
        # settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5m.pt', help='model.pt path(s)')
        parser.add_argument('--torch_th', type=int, default=1, help='Torch num threads.')
        parser.add_argument('--cv_th', type=int, default=1, help='CV2 num threads.')
        parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--dtype', default='fp32', help='data type')
        parser.add_argument('--ipex', action='store_true', help='enable ipex.')
        parser.add_argument('--inc', action='store_true', help='enable inc.')
        opt = parser.parse_args()
        print("marvinTest>>> weights={0}, torch_th={1}, cv_th={2}, img_size={3}, conf_thres={4}, iou_thres={5}, device={6}, dtype={7}, ipex={8}, http_process={9}, inc={10}".format(opt.weights, opt.torch_th, opt.cv_th, opt.img_size, opt.conf_thres, opt.iou_thres, opt.device, opt.dtype, opt.ipex, int(os.getenv("http_process",20)), opt.inc))

        torch.set_num_threads(opt.torch_th)
        cv2.setNumThreads(opt.cv_th)
        # device initialize
        device = select_device(opt.device)
        # Load model
        model = attempt_load(weights=opt.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.half = device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            model.half()  # to FP16

        if opt.ipex and opt.dtype == "fp32":
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model)

        if opt.ipex and opt.dtype == "int8":
            data = torch.rand(1, 3, 640, 640)
            import intel_extension_for_pytorch as ipex
            from intel_extension_for_pytorch.quantization import prepare, convert

            qconfig = ipex.quantization.default_static_qconfig
            prepared_model  = prepare(model, qconfig, example_inputs=data, inplace=False)
            prepared_model(data)
            convert_model = convert(prepared_model)

            with torch.no_grad():
                traced_model = torch.jit.trace(convert_model, data)
                model = torch.jit.freeze(traced_model)

        # traced_model.save("yolov5m_quantized_{}.pt".format(opt.torch_th))

        # if opt.ipex and opt.dtype == "int8":
        #     import intel_extension_for_pytorch as ipex
        #     print("loading yolov5m_quantized_{}.pt".format(opt.torch_th))
        #     model = torch.jit.load("yolov5m_quantized_{}.pt".format(opt.torch_th))
        #     model = ipex.optimize(model)
        #     model.eval()
        #     model = torch.jit.freeze(model)

        if opt.inc and opt.dtype == "int8":
            from neural_compressor.model import Model
            from neural_compressor import quantization
            from neural_compressor.config import PostTrainingQuantConfig
            from neural_compressor.data.dataloaders.dataloader import DataLoader
            from neural_compressor.data import Datasets
            from neural_compressor.quantization import fit
            dataset = Datasets('pytorch')['dummy'](shape=(1, 3, 640, 640))
            config = PostTrainingQuantConfig()
            model = fit(model=model, conf=config,
                calib_dataloader=DataLoader(framework='pytorch', dataset=dataset))

        self.model = model
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
        # pred = profile(self.model, img)

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
