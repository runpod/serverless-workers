ARG BASE_IMAGE=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} as dev-base

WORKDIR /src

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends\
    wget\
    bash\
    openssh-server &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN apt-get update && apt-get install -y --no-install-recommends
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3.8 -y
RUN apt-get install python3-pip -y
RUN apt-get install python3.8-distutils -y

RUN apt-get install python3.8-dev -y
RUN apt-get install python3.8-venv -y
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN python3.8 -m pip install --upgrade pip==20.3
RUN pytohn3.8 -m pip isntall runpod==0.9.1
RUN python3.8 -m pip install torch==2.0.0
RUN python3.8 -m pip install transformers==4.27.2

COPY . /src/

CMD python3.8 -u runpod_infer.py
