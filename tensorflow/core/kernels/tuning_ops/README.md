# Host Tune MatMul Config
## Env
- gcc/g++ >=8
## Compile host and package host into .deb
``` sh
cd host
mkdir build
cd build
cmake ..
make package
```
## Install host
``` sh
sudo dpkg -i HOST_RELEASE-0.1.1-Linux.deb
```
## Compile host_proxy and host_proxy_test
``` sh
cd ../..
mkdir build
cmake ..
make -j
```
## Host tune with one core/multi cores
``` sh
numactl -C 0 ./host_proxy_test/host_proxy_test m1 n1 k1 m2 n2 k2 iters pops algo
numactl -C xx-xx ./host_proxy_test/host_proxy_test m1 n1 k1 m2 n2 k2 iters pops algo
```



