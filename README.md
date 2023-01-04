## Quick start
``` shell
#下载模型文件
wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5m.pt
# wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt

# 服务端启动
CI_DOCKER_EXTRA_PARAMS=-it ./docker_env bash

# 客户端启动
python3 client.py --max=1000 --pool_size=20
```

## Performance
|                                | QPS | Latency(ms) |
| ------------------------------ | --- | ----------- |
| baseline(Ali)                  | 18  | 700         |
| baseline(Icx08)                | 18  | 1000        |
| baseline(Icx08+ipex)           | 23  | 800         |
| baseline(Icx08+ipex+Bf16)      | 6   | 2880        |
| baseline(Icx08+ipex+Int8)      | 26  | 744         |
| baseline(Icx08+OpenVINO+Int8)  | 32  | 590         |
| baseline(Icx08+inc+Int8)       | 39  | 480         |
| baseline(Archerspr02+inc+Int8) | 56  | 340         |