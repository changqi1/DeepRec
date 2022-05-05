# WDL-proxy workload 

This is the workload based on WDL, which aims to evaluate how memory module (e.g., DRAM, Persistent memory) affect the training performance (throughput). The dataset is fake and has no real meaning.
## Environment Setup

Please rebuild the DeepRec wheel package with "pmem" option enabled. 
```bash
## example 
docker pull registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-cpu-py36-ubuntu18.04
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --copt="-L/usr/local/lib" --copt="-lpmem" --copt="-lmemkind" --config=opt //tensorflow/tools/pip_package:build_pip_package
```

Other required pip whls
```bash
numpy                         1.18.5
pandas                        1.1.5
```

## How to run the benchmark

```bash
## generate the dataset, the bigger the num, the bigger the dataset 
python gen_data.py --num=10000000  --name train.csv --output_dir data 
python gen_data.py --num=10000000  --name eval.csv --output_dir data 

## Run the model
# memkind mode: --ev_mem=pmem_memkind
# libpmem mode: --ev_mem=pmem_libpmem
# default mode is dram
python train.py --no_eval --steps=45000 --ev_mem=dram 

```
