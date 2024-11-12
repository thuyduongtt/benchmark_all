# Use the official NVIDIA CUDA 12.1 base image with Ubuntu as a base
FROM ubuntu
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set up environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update &&
RUN apt-get install -y --no-install-recommends python3 python3-pip python3-venv
RUN rm -rf /var/lib/apt/lists/*

# Set up a virtual environment (optional but recommended for Python isolation)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install PyTorch (with CUDA 12.1 support) and other required packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install flash-attn
RUN pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
RUN pip install ijson

# Set up working directory and copy code files
WORKDIR /app
COPY . /app

RUN chmod +x scripts/run_ReasonVQA.sh

# Define entrypoint (replace with your main script if different)
ENTRYPOINT ["bash", "scripts/run_ReasonVQA.sh"]
