FROM silverlogic/python3.8

ARG CI_BUILD_GID
ARG CI_BUILD_GROUP
ARG CI_BUILD_UID
ARG CI_BUILD_USER
ARG CI_BUILD_PASSWD=qwer1234
ARG CI_BUILD_HOME=/home/${CI_BUILD_USER}

ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV no_proxy ${no_proxy}

ENV HTTP_PROXY ${HTTP_PROXY}
ENV HTTPS_PROXY ${HTTPS_PROXY}
ENV NO_PROXY ${NO_PROXY}

RUN apt-get update
RUN apt-get install -y sudo

############################# Set same user in container #############################
RUN getent group "${CI_BUILD_GID}" || addgroup --force-badname --gid ${CI_BUILD_GID} ${CI_BUILD_GROUP}
RUN getent passwd "${CI_BUILD_UID}" || adduser --force-badname --gid ${CI_BUILD_GID} --uid ${CI_BUILD_UID} \
      --disabled-password --home ${CI_BUILD_HOME} --quiet ${CI_BUILD_USER}
RUN usermod -a -G sudo ${CI_BUILD_USER}
RUN echo "${CI_BUILD_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-nopasswd-sudo

USER ${CI_BUILD_UID}:${CI_BUILD_GID}

RUN echo ${CI_BUILD_USER}:${CI_BUILD_PASSWD} | sudo chpasswd
RUN whoami

WORKDIR ${CI_BUILD_HOME}
######################################################################################

ENV PATH ${CI_BUILD_HOME}/bin:$PATH

RUN sudo -E apt-get install -y \
    vim \
    numactl \
    less

# other basic
#RUN sudo yum install -y mesa-libGL.x86_64 && yum clean all
#RUN chown admin:admin -R /home/admin

# pip packages
RUN /usr/local/bin/pip3 install --no-cache-dir imgaug psutil tornado #hsfpy3
RUN /usr/local/bin/pip3 install  --no-cache-dir opencv-python==4.5.4.58
RUN /usr/local/bin/pip3 install  --no-cache-dir protobuf==3.11.3 pandas tqdm PyYAML==5.3.1 seaborn Pillow

RUN sudo ldconfig

# PyTorch
RUN /usr/local/bin/pip3 install --no-cache-dir torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN /usr/local/bin/pip3 install intel-extension-for-pytorch==1.13.0
RUN /usr/local/bin/pip3 install matplotlib tensorboard
RUN /usr/local/bin/pip3 install torch-tb-profiler
RUN /usr/local/bin/pip3 install neural-compressor==2.0