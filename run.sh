#!/bin/sh
#****************************************************************#
# ScriptName: run.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-12-02 16:21
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-12-02 16:21
# Function: 
#***************************************************************#

for http_nums in {1..20}
do
((MAX_NUMS=20/$http_nums))
if [ $MAX_NUMS -le 1 ]; then
MAX_NUMS=4
fi
for((torch_nums=1; torch_nums<=$MAX_NUMS; torch_nums++))
# for torch_nums in {1..$http_nums}
do
echo ">>>marvintest http_nums=$http_nums torch_nums=$torch_nums"
export http_process=$http_nums

# # origin
# numactl -l -C 0-9,64-73 python3 run_xflow.py --torch_th=$torch_nums &
# sleep 3
# numactl -l -C 32-63 python3 client.py --max=300
# pkill -9 "python3"

# # ipex
# numactl -l -C 0-9,64-73 python3 run_xflow.py --torch_th=$torch_nums --ipex &
# sleep 3
# numactl -l -C 32-63 python3 client.py --max=300
# pkill -9 "python3"

# ipex+int8
numactl -l -C 0-9,64-73 python3 run_xflow.py --torch_th=$torch_nums --ipex --dtype=int8 &
sleep 3
numactl -l -C 32-63 python3 client.py --max=300
pkill -9 "python3"

done
done