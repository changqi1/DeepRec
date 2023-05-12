# Tensorflow performance profile

## Tensorflow-Serving 
### Performance
|TF-Serving	|Cores|	CPU usages|Latency-P95 (ms)	|Latency-AVG (ms)|		
|---|---|---|---|---|
|1.15.0	|4	|40+%	|3.921963198808953	|3.75	|+/-	0.12|
|2.12.0	|4	|55+%	|3.6914241733029485	|3.23	|+/-	0.36|
|2.12.0+ITEX | 4 |100% |2.8440896014217287	|2.74	|+/-	0.08|

### Reproduce steps
- TF-1.15
```shell
# create python env
conda create -n tencent_1 python=3.6
conda activate tencent_1

# install tf wheel package
pip install --upgrade pip
pip install tensorflow==1.15
pip install tensorflow-serving-api

# docker run TF-serving
sudo docker run -d -it --cpuset-cpus 0-3,96-99 --rm -p 8500:8500 -p 8501:8501 -v /home/marvin/workspace/DeepRec/modelzoo/tencent:/models/tencent -e MODEL_NAME=tencent tensorflow/serving:1.15.0

# test Latency
python bm_serving.py 8500 10000
```

- TF-2.x
```shell
# create python env
conda create -n tencent_2 python=3.10
conda activate tencent_2

# install tf wheel package
pip install --upgrade pip
pip install tensorflow==2.12.0
pip install tensorflow-serving-api

# docker run TF-serving
sudo docker run -d -it --cpuset-cpus 0-3,96-99 --rm -p 8510:8500 -p 8511:8501 -v /home/marvin/workspace/DeepRec/modelzoo/tencent:/models/tencent -e MODEL_NAME=tencent tensorflow/serving

# test Latency
python bm_serving.py 8510 10000
```

- TF-2.x + ITEX
```shell
# create python env
conda create -n tencent_2 python=3.10
conda activate tencent_2

# install tf wheel package
pip install --upgrade pip
pip install tensorflow==2.12.0
pip install tensorflow-serving-api

# docker run TF-serving
sudo docker load < tf_serving_itex
sudo docker run -d -it --cpuset-cpus 0-3,96-99 --rm -p 8520:8500 -p 8521:8501 -v /home/marvin/workspace/DeepRec/modelzoo/tencent:/models/tencent -e MODEL_NAME=tencent amr-registry.caas.intel.com/aipg-tf/itex-cpu:serving-normal

# test Latency
python bm_serving.py 8520 10000
```

## Tensorflow 
### Performance
|TF	|Cores|	CPU usages|Latency-P95 (ms)	|Latency-AVG (ms)|		
|---|---|---|---|---|
|1.15.0	|8	|25+%	|8.35677271	|6.75	|+/-	1.07|
|2.12.0	|8	|35+%	|7.150103398	|5.33	|+/-	1.36|
|2.12.0+ITEX	|8	|100%	|4.211079549	|4.11	|+/-	0.85|


### Reproduce steps
- TF-1.15
```shell
# create python env
conda create -n tencent_1 python=3.6
conda activate tencent_1

# install tf wheel package
pip install --upgrade pip
pip install tensorflow==1.15

numactl -C 0-7 python bm_TF.py
```

- TF-2.x
```shell
# create python env
conda create -n tencent_2 python=3.10
conda activate tencent_2

# install tf wheel package
pip install --upgrade pip
pip install tensorflow==2.12.0

numactl -C 0-7 python bm_TF.py
```

- TF-2.x + ITEX
```shell
# create python env
conda create -n tencent_2 python=3.10
conda activate tencent_2

# install tf wheel package
pip install --upgrade pip
pip install tensorflow==2.12.0
pip install --upgrade "intel-extension-for-tensorflow[cpu]"

numactl -C 0-7 python bm_TF.py
```

