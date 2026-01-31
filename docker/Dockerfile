# TRELLIS.2 Unity Studio Docker Image
# Based on the working TRELLIS.2 Dockerfile with conda environment
#
# Build:
#   docker build -f docker/Dockerfile -t trellis2-unity-studio .

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge based, no TOS acceptance required)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda create -n trellis2 python=3.10 -y
SHELL ["conda", "run", "-n", "trellis2", "/bin/bash", "-c"]

# Install PyTorch
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Set CUDA arch list for compiling extensions
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install basic dependencies (same as TRELLIS.2)
# Note: transformers is installed later with a specific version for compatibility
RUN pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh tensorboard pandas lpips zstandard kornia timm pillow

# Install server dependencies
RUN pip install fastapi "uvicorn[standard]" python-multipart httpx psutil plyfile aiofiles pyyaml gradio

# Install utils3d
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install flash-attention
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Install nvdiffrast
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation

# Install nvdiffrec
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation

# Install CuMesh
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation

# Install FlexGEMM
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation

# Install diffusers for Flux (need git version for Flux2KleinPipeline)
# The latest diffusers requires latest transformers for Qwen3 support
RUN pip install accelerate && \
    pip install git+https://github.com/huggingface/diffusers.git@main && \
    pip install git+https://github.com/huggingface/transformers.git@main

# Set working directory
WORKDIR /app

# Copy TRELLIS.2 vendor code
COPY vendor/TRELLIS.2 /app/vendor/TRELLIS.2

# Initialize submodules if needed
RUN if [ ! -d /app/vendor/TRELLIS.2/o-voxel/third_party/eigen ]; then \
        git -C /app/vendor/TRELLIS.2 submodule update --init --recursive || true; \
    fi

# Install o-voxel
RUN cp -r /app/vendor/TRELLIS.2/o-voxel /tmp/extensions/o-voxel && \
    pip install /tmp/extensions/o-voxel --no-build-isolation

# Set Python path for trellis2 imports
ENV PYTHONPATH=/app/src:/app/vendor/TRELLIS.2

# Copy server code
COPY src/ /app/src/
COPY app.py /app/

# Create outputs directory
RUN mkdir -p /app/outputs

# Set environment variables for optimization
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV ATTN_BACKEND=flash_attn
ENV SPARSE_ATTN_BACKEND=flash_attn
ENV SPARSE_CONV_BACKEND=flex_gemm

# Expose ports for API server and Gradio
EXPOSE 8000 7860

# Default command - run the API server with conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "trellis2", "uvicorn", "src.trellis2_server:app", "--host", "0.0.0.0", "--port", "8000"]
