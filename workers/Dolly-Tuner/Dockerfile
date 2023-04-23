ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

WORKDIR /

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
