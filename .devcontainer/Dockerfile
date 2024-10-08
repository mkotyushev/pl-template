ARG CUDA_VERSION=11.8.0

# Base CUDA devel image
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git unzip wget tmux curl

# Install Miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# Add conda path to PATH
ENV PATH="/root/miniconda3/bin:${PATH}"

# Create conda env
RUN conda init bash
RUN \
    conda create \
    --name project \
    python=3.10 \
    gcc_linux-64 gxx_linux-64 libgcc \
    -c conda-forge -y && \
    conda install nvidia/label/cuda-${CUDA_VERSION}::cuda-toolkit

# Update current shell to use env
SHELL ["conda", "run", "-n", "project", "/bin/bash", "-c"]

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install torch & other related deps: 
# - not in requirements.txt because order of install matters
# - torch first due to some dependencies importing torch during install
RUN export CUDA_VERSION_STR=cu$(echo $CUDA_VERSION | sed -e "s/\.//g" -e "s/.$//") && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_VERSION_STR

# Set working directory
WORKDIR /workspace/<<[repository_name]>>

# Install requirements
COPY requirements.txt .
RUN pip install \
    -r requirements.txt 

# Add repo folder to PYTHONPATH
ENV PYTHONPATH="/workspace/<<[repository_name]>>:${PYTHONPATH}"
ARG PYTHONPATH="/workspace/<<[repository_name]>>:${PYTHONPATH}"