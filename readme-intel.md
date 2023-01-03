# Intel 内部测试 read me

具体说明查看阿里同学的`readme.md`, 此文档仅包含benchmark相关运行命令.

## Quick start
``` shell
#下载模型文件
wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5m.pt
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt

# 服务端启动
CI_DOCKER_EXTRA_PARAMS=-it ./ci_build bash run.sh

# 客户端启动
python3 client.py --max=1000 --pool_size=20
```

## Performance
|                               | QPS | Latency(ms) |
| ----------------------------- | --- | ----------- |
| baseline(Ali)                 | 18  | 700         |
| baseline(Icx08)               | 18  | 1000        |
| baseline(Icx08+ipex)          | 23  | 800         |
| baseline(Icx08+ipex+Bf16)     | 6   | 2880        |
| baseline(Icx08+ipex+Int8)     | 26  | 744         |
| baseline(Icx08+inc+Int8)      | 39  | 480         |
| baseline(Icx08+OpenVINO+Int8) | 32  | 590         |

## Test case
测试需求: 在20vcpu条件下测试尽可能多的QPS.
1. 服务线程数量(1~40);
2. Torch线程数量(1~20);
3. 数据类型(fp32/int8);


