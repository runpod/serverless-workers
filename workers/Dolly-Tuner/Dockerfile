ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

WORKDIR /

# Get Updates and install essentials
RUN apt-get update && apt-get isntall build-essentials -y

# Clone Dolly Repo and install dependencies
RUN git clone https://github.com/databrickslabs/dolly.git
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb -O /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-3_11.5.1.109-1_amd64.deb -O /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb -O /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-3_10.2.4.109-1_amd64.deb -O /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb && \
    dpkg -i /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
    dpkg -i /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
    dpkg -i /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
    dpkg -i /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb

# Install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Pre-fetch models
# COPY fetcher.py /fetcher.py
# RUN python -u /fetcher.py && \
#     rm /fetcher.py

# Add remaining files
COPY rp_handler.py /rp_handler.py
COPY rp_schemas.py /rp_schemas.py
COPY test_input.json /test_input.json

CMD ["python", "-u", "/rp_handler.py"]
