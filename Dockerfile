# Use the official NVIDIA CUDA 12.1 base image with Ubuntu as a base
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# Set up environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Set up a virtual environment
RUN conda create -n llava_next python=3.10
RUN conda activate llava_next

# Install PyTorch (with CUDA 12.1 support) and other required packages
RUN conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
RUN pip install flash-attn
RUN pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
RUN pip install ijson

# Set up working directory and copy code files
WORKDIR /app
COPY . /app

RUN chmod +x scripts/run_ReasonVQA.sh

# Define entrypoint (replace with your main script if different)
ENTRYPOINT ["bash", "scripts/run_ReasonVQA.sh"]
