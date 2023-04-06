ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN pip install bitsandbytes==0.35
RUN pip install diffusers==0.14.0
RUN pip install transformers==4.25.1
RUN pip install xformers==0.0.18
RUN pip install runpod==0.9.1
RUN pip install setuptools==67.4.0

COPY . /src

RUN python model_fetcher.py

CMD python -u /src/runpod_handler.py
