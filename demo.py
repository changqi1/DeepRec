
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



if __name__ == '__main__':
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
    #cudnn.benchmark = True
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights=opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()  # to FP16

    # warm up
    if device.type != 'cpu':
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))  # run once

    # image preprocess
    url = 'https://img.alicdn.com/imgextra/O1CN01gYMjHS1DAeOTla2w9_!!0-saturn_solar.jpg'
    resp = urllib.request.urlopen(url, timeout=3)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    h, w, c = image.shape

    # Padded resize Convert
    img = letterbox(image, opt.img_size, auto=False, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # run inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
    t2 = time_synchronized()

    print('time(s): {}'.format(t2 - t1))

    ## post-process
    #result = []
    #for i, det in enumerate(pred):  # detections per image
    #    if len(det):
    #        # Rescale boxes from img_size to original size
    #        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

    #        for idx1 in range(len(det)):
    #            pred_dict = {}
    #            box = [str(int(dd)) for dd in det[idx1][:4]]
    #            conf, cls = float(det[idx1][4]), int(det[idx1][5])
    #            name = id_name_dict[cls]
    #            if conf < id_thresh_dict[cls]:
    #                continue
    #            pred_dict["location"] = ','.join(box)
    #            pred_dict["name"] = name
    #            pred_dict["score"] = str(conf)
    #            result.append(pred_dict)

    #print(result)

