ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

WORKDIR /src

# Install missing dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils zstd python3.10-venv git-lfs && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

# Install Python packages
RUN python -m pip install --upgrade pip && \
    pip install torchvision==0.15.1 \
    python-slugify==8.0.1 \
    huggingface-hub==0.13.3 \
    bitsandbytes==0.35 \
    transformers==4.25.1 \
    xformers==0.0.18 \
    runpod==0.9.2

# RUN pip install setuptools==67.4.0
# RUN pip install https://github.com/runpod/runpod-python/archive/main.zip

ADD mainrunpodA1111.py /src/
ADD mainrunpodDreamboothv2.py /src/
ADD install.py /src/
RUN python -u install.py

# Replace paths.py with the one that works with the new paths
RUN cd /workspace/sd/stable-diffusion-webui/modules && \
    wget -q -O paths.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/paths.py && \
    sed -i 's@/content/gdrive/MyDrive/sd/stablediffusion@/workspace/sd/stablediffusion@' /workspace/sd/stable-diffusion-webui/modules/paths.py

ADD model_fetcher.sh /src/
RUN sh /src/model_fetcher.sh

RUN git clone https://github.com/TheLastBen/fast-stable-diffusion.git
RUN git clone --depth 1 --branch main https://github.com/TheLastBen/diffusers

ENV DEBIAN_FRONTEND noninteractive

ADD rp_dreambooth.py /src/
ADD rp_custom_model.py /src/
ADD rp_handler.py /src/
ADD rp_schemas.py /src/
ADD test_input.json /src/

ENV PYTHONUNBUFFERED=1
CMD python -u rp_handler.py
