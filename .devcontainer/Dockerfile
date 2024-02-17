# Base CUDA devel image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace/<<[repository_name]>>

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git unzip wget tmux curl

# Install Miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# Add paths to PATH and PYTHONPATH
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Create conda env
RUN conda init bash
RUN \
    conda create \
    --name <<[repository_name]>> \
    python=3.10 \
    gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=12.1 \
    -c conda-forge -y
RUN /root/miniconda3/envs/<<[repository_name]>>/bin/python -m pip install --upgrade pip && \
    /root/miniconda3/envs/<<[repository_name]>>/bin/pip install uv

# Install torch & other related deps: 
# - not in requirements.txt because order of install matters
# - torch first due to some dependencies importing torch during install
RUN \
    /root/miniconda3/envs/<<[repository_name]>>/bin/uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121

# Install requirements
COPY requirements.txt .
RUN /root/miniconda3/envs/<<[repository_name]>>/bin/uv pip install \
    -r requirements.txt 

# Add repo folder to PYTHONPATH
ENV PYTHONPATH="/workspace/<<[repository_name]>>:${PYTHONPATH}"
ARG PYTHONPATH="/workspace/<<[repository_name]>>:${PYTHONPATH}"