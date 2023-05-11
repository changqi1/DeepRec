#!/bin/bash

for i in {1..10}
do
    # 运行后台程序
    python bm_serving.py 8500 10000 &>> test_8500.log &
    python bm_serving.py 8510 10000 &>> test_8510.log &

    # 等待程序退出
    wait
done
