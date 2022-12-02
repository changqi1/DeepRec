FROM reg.docker.alibaba-inc.com/alibase/alios7u2

ENV PATH $PATH:/opt/conda/bin:/usr/local/cuda/bin


# Python and packages
RUN wget https://304874.oss-cn-hangzhou-zmf.aliyuncs.com/Miniconda3-py38_4.12.0-Linux-x86_64.sh \
  && bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p /opt/conda\
  && rm -rf Miniconda3-py38_4.12.0-Linux-x86_64.sh
# python packages for python3.8, and numpy must be <1.19.0 to compile, see issue 40688 of TF
# packages needed for tf
RUN wget http://304874.oss-cn-hangzhou-zmf.aliyuncs.com/tf_115_cp38_pkgs.tgz\
  && tar -zxvf tf_115_cp38_pkgs.tgz && cd tf_115_cp38_pkgs && ls | xargs /opt/conda/bin/pip install --no-cache-dir\
  && cd .. && rm -rf tf_115_cp38_pkgs*

# for CXXABI 1.3.8
RUN cp /opt/conda/lib/libstdc++.so.6 /usr/lib64/libstdc++.so.6

# other basic
RUN sudo yum install -y mesa-libGL.x86_64 && yum clean all
RUN chown admin:admin -R /home/admin

# pip packages
#RUN  /opt/conda/bin/pip install addict
#RUN  /opt/conda/bin/pip install ftfy
RUN  /opt/conda/bin/pip install -i https://pypi.antfin-inc.com/simple/  --no-cache-dir hsfpy3 imgaug psutil tornado pyclipper
RUN  /opt/conda/bin/pip install  --no-cache-dir protobuf==3.11.3 pandas seaborn wget -i https://pypi.antfin-inc.com/simple/
RUN  /opt/conda/bin/pip install  --no-cache-dir -i https://pypi.antfin-inc.com/simple/ uvicorn fastapi
RUN  /opt/conda/bin/pip install --no-cache-dir mrfh -i https://pypi.antfin-inc.com/simple/
RUN  /opt/conda/bin/pip --default-timeout=100 --no-cache-dir install -i https://pypi.antfin-inc.com/simple/ git+http://gitlab-ci-token:cc3e8d873b62f721e555e7a5185b16@gitlab.alibaba-inc.com/alimamacv/istio_pyservice_utils.git@0.2.6
RUN  /opt/conda/bin/pip install  --no-cache-dir http://246950.oss-cn-hangzhou-zmf.aliyuncs.com/archive/opencv_python-4.5.4.58-cp38-cp38-manylinux2014_x86_64.whl
# for oss download
RUN  /opt/conda/bin/pip install -i https://pypi.antfin-inc.com/simple/  --no-cache-dir oss2
#Image
RUN  /opt/conda/bin/pip install --no-cache-dir python-multipart -i https://pypi.antfin-inc.com/simple/
RUN  /opt/conda/bin/pip install --no-cache-dir Image -i https://pypi.antfin-inc.com/simple/
RUN /opt/conda/bin/pip  install tf2onnx -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir
# used by face, accelerate image decode
RUN /opt/conda/bin/pip  install PyTurboJPEG -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir &&\
    wget http://246950.oss-cn-hangzhou-zmf.aliyuncs.com/libturbojpeg.so &&\
    mv libturbojpeg.so /usr/lib64/


ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/nvidia/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib/
ENV PATH $PATH:/usr/local/nvidia/bin
    
RUN ldconfig
# PyTorch
RUN /opt/conda/bin/pip install --no-cache-dir http://246950.oss-cn-hangzhou-zmf.aliyuncs.com/a_keep/20220621_cuda113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl &&\
    /opt/conda/bin/pip install --no-cache-dir http://246950.oss-cn-hangzhou-zmf.aliyuncs.com/a_keep/20220626_cuda113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl 

