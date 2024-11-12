# Use the official NVIDIA CUDA 12.1 base image with Ubuntu as a base
FROM ubuntu
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set up environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh

RUN conda create -n llava_next python=3.10
RUN conda activate llava_next

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
