#!/bin/bash
# docker non-root user

# export http_proxy
http_proxy=`cat ~/.bashrc | grep "export http_proxy='http://child-prc.intel.com:913'"`
if [ -z "$http_proxy" ]
then # $http_proxy is empty
    echo "export http_proxy='http://child-prc.intel.com:913'" >> ~/.bashrc
fi

# export https_proxy
https_proxy=`cat ~/.bashrc | grep "export https_proxy='http://child-prc.intel.com:913'"`
if [ -z "$https_proxy" ]
then # $https_proxy is empty
    echo "export https_proxy='http://child-prc.intel.com:913'" >> ~/.bashrc
fi

# export PATH
path=`cat ~/.bashrc | grep "~/.local/bin"`
if [ -z "$path" ]
then # $path is empty
    echo "export PATH=\$PATH:~/.local/bin" >> ~/.bashrc
fi

export http_proxy='http://child-prc.intel.com:913'
export https_proxy='http://child-prc.intel.com:913'
export PATH=$PATH:~/.local/bin

apt update
apt-get install sudo cmake fio -y
git clone https://github.com/intel-sandbox/cesg.alibaba.EasyRecOpt.git EasyRecOpt

cur_path=`pwd`

env_clear(){
    rm -rf venv-py3.6-google_tf1.15.0_DeepRec
    rm -rf EasyRec
    rm -rf easyrec_data
    rm -rf easyrec_data.tar.gz
}

dependency(){
    cd $cur_path
    pip install  virtualenv==16.7.7
    virtualenv -p python3.6 venv-py3.6-google_tf1.15.0_DeepRec
    source ./venv-py3.6-google_tf1.15.0_DeepRec/bin/activate
    pip install numpy==1.16.5 opt-einsum==2.3.2 future xlwt xlrd xlutils scikit-learn psutil -i https://pypi.douban.com/simple
    pip install Keras_Preprocessing Keras_Applications astor 
    pip install tensorflow==1.15.0 -i https://pypi.douban.com/simple
    # git lfs install --skip-repo
}

install_EasyRec(){
    # sudo git lfs uninstall --skip-repo
    # sudo rm -rf /etc/gitconfig
    git clone https://github.com/changqi1/EasyRec.git

    wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/easyrec_data.tar.gz
    tar xf easyrec_data.tar.gz

    cd EasyRec
    sh scripts/gen_proto.sh
    git submodule init 
    git submodule update
    #git-lfs pull

    # replace data directory with the extracted data directory(maybe not needed)
    mv data data-bak
    mv ../data ./
}

env_clear
dependency
install_EasyRec
