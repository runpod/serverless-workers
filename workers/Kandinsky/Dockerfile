# Base image
FROM runpod/pytorch:3.10-2.0.0-117

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /

# Install Python packages
RUN pip install --no-cache-dir runpod==0.9.2 opencv-python "git+https://github.com/ai-forever/Kandinsky-2.git" "git+https://github.com/openai/CLIP.git"

# Copy the fetcher script into the container
COPY fetcher.py /fetcher.py

# Run the fetcher script to fetch the model files
RUN python /fetcher.py && \
    rm /fetcher.py

# Copy the serverless implementation file into the container
COPY rp_handler.py /rp_handler.py

# Set the entry point
CMD ["python", "/rp_handler.py"]
