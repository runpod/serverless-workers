ARG BASE_IMAGE=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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

# Install system dependencies
RUN apt-get install python3-opencv -y

# Install Python dependencies
RUN python3.8 -m pip install --upgrade pip==20.3
RUN python3.8 -m pip install basicsr==1.4.2
RUN python3.8 -m pip install torch==1.13.0
RUN python3.8 -m pip install torchvision==0.14.0
RUN python3.8 -m pip install numpy==1.21.6
RUN python3.8 -m pip install gradio==3.18.0
RUN python3.8 -m pip install albumentations==1.2.1
RUN python3.8 -m pip install opencv-contrib-python==4.6.0.66
RUN python3.8 -m pip install imageio==2.9.0
RUN python3.8 -m pip install imageio-ffmpeg==0.4.8
RUN python3.8 -m pip install pytorch-lightning==1.9.1
RUN python3.8 -m pip install omegaconf==2.3.0
RUN python3.8 -m pip install test-tube==0.7.5
RUN python3.8 -m pip install streamlit==1.18.1
RUN python3.8 -m pip install einops==0.6.0
RUN python3.8 -m pip install transformers==4.26.1
RUN python3.8 -m pip install webdataset==0.2.33
RUN python3.8 -m pip install kornia==0.6.9
RUN python3.8 -m pip install open_clip_torch==2.11.1
RUN python3.8 -m pip install invisible-watermark==0.1.5
RUN python3.8 -m pip install streamlit-drawable-canvas==0.9.2
RUN python3.8 -m pip install torchmetrics==0.11.1
RUN python3.8 -m pip install timm==0.6.12
RUN python3.8 -m pip install addict==2.4.0
RUN python3.8 -m pip install yapf==0.32.0
RUN python3.8 -m pip install prettytable==3.6.0
RUN python3.8 -m pip install runpod==0.9.1

# Copy model files
COPY . /src/

ARG MODEL_TYPE="openpose"
ENV MODEL_TYPE=${MODEL_TYPE}

RUN python3.8 download_weights.py --model_type="${MODEL_TYPE}"

CMD python3.8 -u runpod_infer.py --model_type="${MODEL_TYPE}"
