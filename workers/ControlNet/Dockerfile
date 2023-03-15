FROM runpod/pytorch:3.10-1.13.1-116

SHELL ["/bin/bash", "-c"]

WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3.8.5 -y
RUN apt-get install python3-pip -y
RUN apt-get install python3.8.5-distutils -y

RUN python3.8.5 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install system dependencies
RUN apt-get install python3-opencv -y

# Install Python dependencies
RUN python3.8.5 -m pip install --upgrade pip==20.3
RUN python3.8.5 -m pip install basicsr==1.4.2
RUN python3.8.5 -m pip install torch==1.13.0
RUN python3.8.5 -m pip install torchvision==0.14.0
RUN python3.8.5 -m pip install numpy==1.21.6
RUN python3.8.5 -m pip install gradio==3.18.0
RUN python3.8.5 -m pip install albumentations==1.2.1
RUN python3.8.5 -m pip install opencv-contrib-python==4.6.0.66
RUN python3.8.5 -m pip install imageio==2.9.0
RUN python3.8.5 -m pip install imageio-ffmpeg==0.4.8
RUN python3.8.5 -m pip install pytorch-lightning==1.9.1
RUN python3.8.5 -m pip install omegaconf==2.3.0
RUN python3.8.5 -m pip install test-tube==0.7.5
RUN python3.8.5 -m pip install streamlit==1.18.1
RUN python3.8.5 -m pip install einops==0.6.0
RUN python3.8.5 -m pip install transformers==4.26.1
RUN python3.8.5 -m pip install webdataset==0.2.33
RUN python3.8.5 -m pip install kornia==0.6.9
RUN python3.8.5 -m pip install open_clip_torch==2.11.1
RUN python3.8.5 -m pip install invisible-watermark==0.1.5
RUN python3.8.5 -m pip install streamlit-drawable-canvas==0.9.2
RUN python3.8.5 -m pip install torchmetrics==0.11.1
RUN python3.8.5 -m pip install timm==0.6.12
RUN python3.8.5 -m pip install addict==2.4.0
RUN python3.8.5 -m pip install yapf==0.32.0
RUN python3.8.5 -m pip install prettytable==3.6.0
RUN python3.8.5 -m pip install runpod==0.9.0

# Copy model files
COPY . /src/

CMD ["python3.8.5", "-m", "-u", "runpod_infer.py"]
