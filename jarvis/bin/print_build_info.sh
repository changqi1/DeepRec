#!/usr/bin/env bash
# Print build info, including info related to the machine, OS, build tools
# and TensorFlow source code. This can be used by build tools such as Jenkins.
# All info is printed on a single line, in JSON format, to workaround the
# limitation of Jenkins Description Setter Plugin that multi-line regex is
# not supported.
#
# Usage:
#   print_build_info.sh (CONTAINER_TYPE) (COMMAND)
#     e.g.,
#       print_build_info.sh GPU bazel test -c opt --config=cuda //tensorflow/...

# Information about the command
COMMAND=("$@")

# Information about machine and OS
OS=$(uname)
KERNEL=$(uname -r)

ARCH=$(uname -p)
PROCESSOR=$(grep "model name" /proc/cpuinfo | head -1 | awk '{print substr($0, index($0, $4))}')
PROCESSOR_COUNT=$(grep "model name" /proc/cpuinfo | wc -l)

MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2, $3}')
SWAP_TOTAL=$(grep SwapTotal /proc/meminfo | awk '{print $2, $3}')

# Information about build tools
if [[ ! -z $(which bazel) ]]; then
  BAZEL_VER=$(bazel version | head -1)
fi

if [[ ! -z $(which javac) ]]; then
  JAVA_VER=$(javac -version 2>&1 | awk '{print $2}')
fi

if [[ ! -z $(which python) ]]; then
  PYTHON_VER=$(python -V 2>&1 | awk '{print $2}')
fi

if [[ ! -z $(which g++) ]]; then
  GPP_VER=$(g++ --version | head -1)
fi

if [[ ! -z $(which swig) ]]; then
  SWIG_VER=$(swig -version > /dev/null | grep -m 1 . | awk '{print $3}')
fi

# Information about TensorFlow source
TF_FETCH_URL=$(git config --get remote.origin.url)
TF_HEAD=$(git rev-parse HEAD)
TF_HEAD_2=$(git rev-parse HEAD^)

# NVIDIA & CUDA info
NVIDIA_DRIVER_VER=""
if [[ -f /proc/driver/nvidia/version ]]; then
  NVIDIA_DRIVER_VER=$(head -1 /proc/driver/nvidia/version | awk '{print $(NF-6)}')
fi

CUDA_DEVICE_COUNT="0"
CUDA_DEVICE_NAMES=""
if [[ ! -z $(which nvidia-debugdump) ]]; then
  CUDA_DEVICE_COUNT=$(nvidia-debugdump -l | grep "^Found [0-9]*.*device.*" | awk '{print $2}')
  CUDA_DEVICE_NAMES=$(nvidia-debugdump -l | grep "Device name:.*" | awk '{print substr($0, index($0,\
 $3)) ","}')
fi

CUDA_TOOLKIT_VER=""
if [[ ! -z $(which nvcc) ]]; then
  CUDA_TOOLKIT_VER=$(nvcc -V | grep release | awk '{print $(NF)}')
fi

# Print info
echo
echo ">>>>>>>>>>>>>>>>>System configuration>>>>>>>>>>>>>>>>>>>>"
echo -e "TF_BUILD_INFO = {"\
"\n\tcontainer_type: \"${CONTAINER_TYPE}\", "\
"\n\tcommand: \"${COMMAND[*]}\", "\
"\n\tsource_HEAD: \"${TF_HEAD}\", "\
"\n\tsource_HEAD^: \"${TF_HEAD_2}\", "\
"\n\tsource_remote_origin: \"${TF_FETCH_URL}\", "\
"\n\tOS: \"${OS}\", "\
"\n\tkernel: \"${KERNEL}\", "\
"\n\tarchitecture: \"${ARCH}\", "\
"\n\tprocessor: \"${PROCESSOR}\", "\
"\n\tprocessor_count: \"${PROCESSOR_COUNT}\", "\
"\n\tmemory_total: \"${MEM_TOTAL}\", "\
"\n\tswap_total: \"${SWAP_TOTAL}\", "\
"\n\tBazel_version: \"${BAZEL_VER}\", "\
"\n\tJava_version: \"${JAVA_VER}\", "\
"\n\tPython_version: \"${PYTHON_VER}\", "\
"\n\tgpp_version: \"${GPP_VER}\", "\
"\n\tswig_version: \"${SWIG_VER}\", "\
"\n\tNVIDIA_driver_version: \"${NVIDIA_DRIVER_VER}\", "\
"\n\tCUDA_device_count: \"${CUDA_DEVICE_COUNT}\", "\
"\n\tCUDA_device_names: \"${CUDA_DEVICE_NAMES}\", "\
"\n\tCUDA_toolkit_version: \"${CUDA_TOOLKIT_VER}\""\
"\n}"
echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo
sleep 3