FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as builder

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# install python3.9
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    python3.9-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /

# install python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels runpod==0.9.2 basicsr realesrgan

# download models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P weights
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P weights

FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# install python3.9
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    python3.9-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /app

# install python packages
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# copy models
COPY --from=builder /weights /app/weights

COPY . .

CMD [ "python3", "-u", "handler.py" ]
