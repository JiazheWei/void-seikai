---
title: Determined-AI 集群管理工具的BUG集散地
publishDate: 2024-12-01
description: 'DAI使用过程的踩坑日记暨入坑指南'
tags:
  - Tech
  - Determined-AI
heroImage: { src: 'deco27.jpg', color: '#64574D' }
language: 'Chinese'
---

Determined AI是一个能够高效记录并优化机器学习模型训练工作流的开源平台，同时也被广泛配置在大型集群上以规范集群使用和管理虚拟环境。本文作为一个quick start，可以在最短的时间内提供在DAI系统下搭建起一个最基本的能运行的环境所需要的知识。至于更繁琐的细节，我也还在学习，用到再更新。

# 编写dockerfile

没有接触过docker可以先去[docker的官方文档](https://docs.docker.com/)看看。一句话概括，docker就是将你的代码和运行环境打包在一起，这样你就可以在任何地方运行你的代码了。像虚拟机，但是不像虚拟机一样还打包一个操作系统，所以非常轻量。显然地，本来使用conda管理环境的话，我们要用conda来下载各种包，现在我们就将对各种包的需求写在叫做**Dockerfile**的文件里,然后由docker按照这个文件去下载对应的包，为我们搭建好环境。这个环境叫做**container**,container没有运行的时候叫做**image**。

提供一个简单的手搓dockerfile模板：
```
# Determined AI's base image
FROM harbor.lins.lab/determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.31.1
# Another one of their base images, with newer CUDA and pytorch
# FROM determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.27.1
# You can check out their images here: https://hub.docker.com/r/determinedai/environments/

# Some important environment variables in Dockerfile
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1
# Custom Configuration
RUN sed -i  "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list && \
    sed -i  "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/* && \
    apt-get update && \
    apt-get -y install tzdata && \
    apt-get install -y unzip python-opencv graphviz && \
    apt-get clean
COPY environment.yml /tmp/environment.yml
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN conda env update --name base --file /tmp/environment.yml
RUN conda clean --all --force-pkgs-dirs --yes
RUN eval "$(conda shell.bash hook)" && \
    conda activate base && \
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install --requirement /tmp/pip_requirements.txt
```
和一个使用nvidia的mini template:

```
#! ----EDIT HERE TO CHANGE BASE IMAGE----
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# COMMENT ABOVE & UNCOMMENT BELOW TO USE BASE IMAGE WITH PYTORCH:
# FROM nvcr.io/nvidia/pytorch:22.12-py3

# Environment variables
ARG PYTHON_VERSION=3.8.12
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# Install apt packages
RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Determined requirements and common tools
        autoconf automake autotools-dev build-essential ca-certificates \
        make cmake ninja-build pkg-config g++ ccache yasm \
        ccache doxygen graphviz plantuml \
        daemontools krb5-user ibverbs-providers libibverbs1 \
        libkrb5-dev librdmacm1 libssl-dev libtool \
        libnuma1 libnuma-dev libpmi2-0-dev \
        openssh-server openssh-client pkg-config nfs-common \
        ## Tools
        git curl wget unzip nano net-tools sudo htop iotop \
        cloc rsync xz-utils software-properties-common \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Conda and Determined AI stuff
#! ---EDIT notebook-requirements.txt TO ADD PYPI PACKAGES----
WORKDIR /tmp
ENV PATH="/opt/conda/bin:${PATH}"
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN git clone https://github.com/LingzheZhao/determinedai-container-scripts &&\
    cd determinedai-container-scripts &&\
    git checkout v0.2.2 &&\
    ./install_python.sh ${PYTHON_VERSION} &&\
    cp .condarc /opt/conda/.condarc &&\
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r notebook-requirements.txt &&\
    pip install -r additional-requirements.txt &&\
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements" &&\
    ./add_det_nobody_user.sh &&\
    ./install_libnss_determined.sh &&\
    rm -rf /tmp/*

#! ----EDIT HERE TO INSTALL CONDA PACKAGES----
# Example: PyTorch w/ cuda 11.6
# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# Example: Jax
# RUN conda install -c conda-forge jax
# Example: OpenCV
# RUN conda install -c conda-forge opencv
```

如果我们下载的项目中有一个对应的requirements.txt/setup.py等诸如此类的文件，就将他们填到这份dockerfile的对应位置中。

接着运行：

```
DOCKER_BUILDKIT=0 docker build -t my_image:v1.0 .
```
就可以创建一个叫做**my_image**的image，版本为v1.0。前面的**DOCKER_BUILDKIT=0**是可选的，如果设置为1，则docker会使用buildkit来加速构建，但是会报错，所以还是设置为0。因为我现在使用的是组里的私有注册表。

使用
```
docker images
```
就可以看到我们刚刚创建的image。

# 建立yaml文件

yaml文件是Determined AI用来管理集群的配置文件，我们只需要在文件中指定我们刚刚创建的image，然后填上代码和数据的路径就可以创建对应的container运行代码了。

要方便地管理yaml文件可以在命令行中使用**cli**工具。先下载：

```
pip install determined
```

接着登录：

```
det user login <username>
```

初始是没有密码的，但是可以通过change-password命令设置密码。

还是提供一份yaml文件的模板：

```
description: <task_name>
resources:
    slots: 1
    resource_pool: 64c128t_512_3090
    shm_size: 4G
bind_mounts:
    - host_path: /home/<username>/
      container_path: /run/determined/workdir/home/
    - host_path: /labdata0/<project_name>/
      container_path: /run/determined/workdir/data/
environment:
    image: determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-0.19.4
```
可以看见在这里**environment**中我们填入了我们刚刚创建的image。在第一个host_path中我们填入**代码所在的绝对路径**，第二个host中填入**数据所在的绝对路径**。数据路径一般默认使用组里已经下载好的大路径。

接着运行：

```
    det shell start --config-file test_task.yaml
```

就可以创建一个container运行代码了。当然注意yaml文件之类的名称要根据自己的情况改变。

# 比较模板的样式

上面提供的方案是我比较喜欢的一种，但是如果你还是希望少折腾，用一张已定义程度比较高的板子的话，也可以参考这一套。

首先在项目文件夹，和例如setup.py/yaml/requirement.txt文件同一目录下创建一个image_builder.sh：

```
SOURCE_IMAGE_NAME="public_vision:v1.2"
# Name of the source Docker image that will be used as the base image.

TARGET_IMAGE_NAME="public_vision:v1.3"
# Name of the target Docker image that will be created after applying changes.

INSTALL_PACKAGES="seaborn numpy"
# List of Python packages to be installed using pip.

UPGRADE_PACKAGES="lightly pandas"
# List of Python packages to be upgraded to their latest versions.

# Create a temporary file in the system's temporary directory.
TEMP_FILE=$(mktemp /tmp/temp_Dockerfile.XXXXXX.txt)

# Populate the temporary Dockerfile with the base image and package installation commands.
cat > "$TEMP_FILE" << EOF
FROM harbor.lins.lab/library/$SOURCE_IMAGE_NAME

RUN pip install $INSTALL_PACKAGES
# Install the specified packages using pip.

RUN pip install --upgrade $UPGRADE_PACKAGES
# Upgrade the specified packages to their latest versions using pip.
EOF

# Build a new Docker image with the specified tag, disabling BuildKit and using specific proxy settings.
DOCKER_BUILDKIT=0 docker build -t $TARGET_IMAGE_NAME -f $TEMP_FILE --build-arg http_proxy=http://192.168.123.169:18889 --build-arg https_proxy=http://192.168.123.169:18889 .

# Tag the newly built Docker image with the target name in the specified Docker registry.
docker tag $TARGET_IMAGE_NAME harbor.lins.lab/library/$TARGET_IMAGE_NAME

# Push the tagged Docker image to the specified Docker registry.
docker push harbor.lins.lab/library/$TARGET_IMAGE_NAME

```

将内容填写其中。名字和需求都可以自己自定义。接着bash这个脚本，你就能在docker hubs中看见这张创建出来的image。

接着我们需要上传到实验室的hub。在det登录一次之后就不需要再登陆了，使用：

```
det tag <your_image> harbor.lins.lab/library/<your_image>
```

为image打上规范的实验室tag，然后

```
docker push harbor.lins.lab/library/your_image:vX.X
```

就可以将image送入hub中。之后直接在创建环境脚本中替换image名称即可。