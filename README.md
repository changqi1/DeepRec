## Quick start
``` shell
#下载模型文件
wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5m.pt

# 服务端启动
CI_DOCKER_EXTRA_PARAMS=-it ./docker_env bash
# binding 20vcpu
numactl -C 0-9,64-73 python run_xflow.py [--options]
# 客户端启动
python3 client.py --max=1000 --pool_size=20
```
