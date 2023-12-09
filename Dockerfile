# Source image
FROM python:3.8-slim

ARG TORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2

# Set the working directory in the container
WORKDIR /app

# Install some basic dependencies
RUN apt-get update
RUN apt-get install libgomp1

# Instatll requirements
RUN pip install -U torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}
RUN pip install autogluon==0.8.2
RUN pip install torchmetrics==0.9.3

# Copy the current directory contents into the container at /app
COPY . /app
