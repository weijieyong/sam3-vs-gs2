FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    # required for opencv
    libgl1 libglib2.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Nightly (CUDA 12.8)
# Note: We use the nightly index for cu128 support for NVIDIA RTX 50-series GPUs
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Set working directory
WORKDIR /workspace

