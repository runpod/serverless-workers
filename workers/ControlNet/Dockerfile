FROM runpod/pytorch:3.10-1.13.1-116

SHELL ["/bin/bash", "-c"]

WORKDIR /src

# Install system dependencies
RUN apt-get install python3-opencv -y

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install torch==1.13.0
RUN pip install torchvision==0.14.0
RUN pip install numpy==1.21.6
RUN pip install gradio==3.18.0
RUN pip install albumentations==1.2.1
RUN pip install opencv-contrib-python==4.6.0.66
RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.8
RUN pip install pytorch-lightning==1.9.1
RUN pip install omegaconf==2.3.0
RUN pip install test-tube==0.7.5
RUN pip install streamlit==1.18.1
RUN pip install einops==0.6.0
RUN pip install transformers==4.26.1
RUN pip install webdataset==0.2.33
RUN pip install kornia==0.6.9
RUN pip install open_clip_torch==2.11.1
RUN pip install invisible-watermark==0.1.5
RUN pip install streamlit-drawable-canvas==0.9.2
RUN pip install torchmetrics==0.11.1
RUN pip install timm==0.6.12
RUN pip install addict==2.4.0
RUN pip install yapf==0.32.0
RUN pip install prettytable==3.6.0
RUN pip install runpod==0.9.0

# Copy model files
COPY . /src/

CMD ["python", "-u", "runpod_infer.py"]
