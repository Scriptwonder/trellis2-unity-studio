FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install -r requirements.txt

COPY vendor/TRELLIS.2 /app/vendor/TRELLIS.2
WORKDIR /app/vendor/TRELLIS.2
RUN bash setup.sh --basic --flash-attn --nvdiffrast --cumesh --o-voxel

WORKDIR /app
COPY src/ /app/src/
COPY app.py /app/

EXPOSE 8000 7860

CMD ["uvicorn", "src.trellis2_server:app", "--host", "0.0.0.0", "--port", "8000"]
