FROM silverlogic/python3.8  # you can replace this with any base docker image


# other basic
#RUN sudo yum install -y mesa-libGL.x86_64 && yum clean all
#RUN chown admin:admin -R /home/admin

# pip packages
RUN /usr/local/bin/pip3 install -i https://pypi.antfin-inc.com/simple/  --no-cache-dir hsfpy3 imgaug psutil tornado
RUN /usr/local/bin/pip3 install  --no-cache-dir opencv-python==4.5.4.58 -i https://pypi.antfin-inc.com/simple/
RUN /usr/local/bin/pip3 install  --no-cache-dir protobuf==3.11.3 pandas tqdm PyYAML==5.3.1 seaborn Pillow -i https://pypi.antfin-inc.com/simple/


RUN ldconfig
# PyTorch
RUN /usr/local/bin/pip3 install --no-cache-dir torch==1.11.0 -i https://pypi.antfin-inc.com/simple/ &&\
    /usr/local/bin/pip3 install --no-cache-dir torchvision==0.12.0 -i https://pypi.antfin-inc.com/simple/

