ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]


# Install dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade runpod==0.9.3
RUN pip install tensorflow-io
RUN pip install datasets==2.11.0
RUN pip install transformers==4.28.1
RUN pip install peft==0.2.0
RUN pip install bitsandbytes==0.38.1
RUN pip install protobuf==3.20.*
RUN pip install loralib
RUN pip install sentencepiece
