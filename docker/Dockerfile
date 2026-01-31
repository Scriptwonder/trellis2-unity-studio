FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV ATTN_BACKEND=flash_attn
ENV SPARSE_ATTN_BACKEND=flash_attn
ENV SPARSE_CONV_BACKEND=flex_gemm
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
# Set Python path for trellis2 imports
ENV PYTHONPATH=/app/src:/app/vendor/TRELLIS.2

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl sudo libjpeg-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch first
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install basic Python deps
RUN pip install --no-cache-dir imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard pyyaml fastapi "uvicorn[standard]" python-multipart httpx pillow psutil plyfile aiofiles
RUN pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
# Install diffusers from main branch (includes Flux2KleinPipeline) + accelerate
# ARG forces cache bust - update date to force rebuild
ARG DIFFUSERS_CACHE_BUST=2026-01-31
RUN pip install --no-cache-dir accelerate git+https://github.com/huggingface/diffusers.git@main
RUN pip install --no-cache-dir kornia timm safetensors huggingface-hub rembg onnxruntime

# Flash attention
RUN pip install flash-attn==2.7.3 --no-build-isolation

# nvdiffrast - build from source and manually copy Python package
# The pyproject.toml doesn't properly include the Python sources in the wheel
# Also patch __init__.py to handle missing package metadata
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install /tmp/nvdiffrast --no-build-isolation && \
    cp -r /tmp/nvdiffrast/nvdiffrast /usr/local/lib/python3.10/dist-packages/ && \
    sed -i 's/from importlib.metadata import version/__version__ = "0.3.3"  # Hardcoded due to manual install\ntry:\n    from importlib.metadata import version\nexcept: pass/' /usr/local/lib/python3.10/dist-packages/nvdiffrast/__init__.py && \
    sed -i 's/__version__ = version(__package__ or .nvdiffrast.)/# __version__ already set above/' /usr/local/lib/python3.10/dist-packages/nvdiffrast/__init__.py && \
    rm -rf /tmp/nvdiffrast

# nvdiffrec (renderutils for mesh processing)
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec && \
    pip install /tmp/nvdiffrec --no-build-isolation && rm -rf /tmp/nvdiffrec

# Copy TRELLIS.2 code (if present) and ensure it is available
ARG TRELLIS2_REPO=https://github.com/microsoft/TRELLIS.2.git
ARG TRELLIS2_REF=5565d240c4a494caaf9ece7a554542b76ffa36d3
COPY vendor/TRELLIS.2 /app/vendor/TRELLIS.2
RUN if [ ! -d /app/vendor/TRELLIS.2/trellis2 ] || \
            [ ! -d /app/vendor/TRELLIS.2/o-voxel/third_party/eigen ]; then \
            rm -rf /app/vendor/TRELLIS.2 && \
            git clone --recursive ${TRELLIS2_REPO} /app/vendor/TRELLIS.2 && \
            git -C /app/vendor/TRELLIS.2 checkout ${TRELLIS2_REF} && \
            git -C /app/vendor/TRELLIS.2 submodule update --init --recursive; \
        else \
            git -C /app/vendor/TRELLIS.2 submodule update --init --recursive || true; \
        fi
WORKDIR /app/vendor/TRELLIS.2

# Build CuMesh
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive && \
    pip install /tmp/CuMesh --no-build-isolation && rm -rf /tmp/CuMesh

# Build FlexGEMM
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive && \
    pip install /tmp/FlexGEMM --no-build-isolation && rm -rf /tmp/FlexGEMM

# Build o-voxel from local
RUN pip install ./o-voxel --no-build-isolation

# Copy server code
WORKDIR /app
COPY src/ /app/src/
COPY app.py /app/

EXPOSE 8000 7860

CMD ["uvicorn", "src.trellis2_server:app", "--host", "0.0.0.0", "--port", "8000"]
