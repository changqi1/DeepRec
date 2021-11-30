FROM registry.cn-shanghai.aliyuncs.com/pai-dlc/tensorflow-developer:1.15deeprec-dev-cpu-cibuild-py36-ubuntu18.04

ARG BAZEL_VERSION=0.26.1

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

# Some TF tools expect a "python" binary
RUN sudo ln -s $(which python3) /usr/local/bin/python

RUN sudo -E apt-get install -y \
    vim \
    numactl \
    openssh-server \
    less

EXPOSE 22

RUN sudo mkdir /var/run/sshd

# execute in the container
RUN echo "sudo /usr/sbin/sshd" >> ${CI_BUILD_HOME}/.bashrc

# RUN sudo bash -c "echo 0 >> /proc/sys/kernel/kptr_restrict"
# RUN sudo bash -c "echo 0 >> /proc/sys/kernel/perf_event_paranoid"
