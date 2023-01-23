# Include Python
from python:3.11.1-buster

# Define your working directory
WORKDIR /

# Install runpod
RUN pip install runpod

# Add your file
ADD whatever.py .

# Call your file when your container starts
CMD [ "python", "-u", "/whatever.py" ]
