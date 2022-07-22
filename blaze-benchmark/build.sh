#!/bin/bash
set -x

# download example file
if [ ! -d example/ ]; then
 wget --no-proxy http://crt-e302.sh.intel.com/files/example.tar.gz && tar -zxvf example.tar.gz 
fi

# build tensorflow
pushd ../

bazel_build cpu

popd

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ 

pushd thirdparty/boost_1_53_0
if [ ! -d stage/lib ]; then
./bootstrap.sh --with-libraries=chrono,filesystem,system,thread && ./b2 -j32
fi
popd

if [ ! -d build ]; then
    mkdir build
fi
pushd build

LWP=`pwd`

cmake .. \
    -DTF_SRC_DIR=$LWP/../../ \
    -DAIOS=0

make VERBOSE=1 -j
popd
