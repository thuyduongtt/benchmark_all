# Use the official NVIDIA CUDA 12.1 base image with Ubuntu as a base
# FROM ubuntu:24.04
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set up environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Upgrade pip to the latest version
RUN pip install --upgrade pip

RUN pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
RUN pip install ijson

# Set up working directory and copy code files
WORKDIR /app
COPY . /app

RUN chmod +x scripts/run_ReasonVQA.sh

# Define entrypoint (replace with your main script if different)
ENTRYPOINT ["bash", "scripts/run_ReasonVQA.sh"]
