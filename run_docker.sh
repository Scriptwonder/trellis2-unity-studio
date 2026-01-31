#!/bin/bash
# TRELLIS.2 Unity Studio Docker Run Script
# This script runs the container with proper HuggingFace authentication
# 
# Prerequisites:
#   1. Build the image first:
#      docker build -f docker/Dockerfile -t trellis2-unity-studio .
#
#   2. Either:
#      a) Create a .env file with HF_TOKEN=your_huggingface_token
#      b) Export HF_TOKEN environment variable
#      c) Have ~/.cache/huggingface with logged-in credentials
#
# Usage:
#   ./run_docker.sh                    # Run with default settings
#   ./run_docker.sh --memory swap      # Run in swap mode (32GB RAM)
#   ./run_docker.sh --memory keep      # Run in keep_loaded mode (64GB+ RAM)
#   ./run_docker.sh --detach           # Run in background

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
CONTAINER_NAME="trellis2-unity-studio"
IMAGE_NAME="trellis2-unity-studio:latest"
MEMORY_MODE="auto"
DETACH=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --memory)
            if [[ "$2" == "swap" ]]; then
                MEMORY_MODE="swap"
            elif [[ "$2" == "keep" || "$2" == "keep_loaded" ]]; then
                MEMORY_MODE="keep_loaded"
            else
                MEMORY_MODE="$2"
            fi
            shift 2
            ;;
        --detach|-d)
            DETACH="-d"
            shift
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Load .env file if present
ENV_ARGS=""
if [[ -f .env ]]; then
    echo "[INFO] Loading environment from .env file"
    ENV_ARGS="--env-file .env"
elif [[ -f docker/.env ]]; then
    echo "[INFO] Loading environment from docker/.env file"
    ENV_ARGS="--env-file docker/.env"
fi

# Check for HuggingFace token
if [[ -z "$HF_TOKEN" ]] && [[ ! -f .env ]] && [[ ! -f docker/.env ]]; then
    if [[ -f ~/.cache/huggingface/token ]]; then
        echo "[INFO] Using HuggingFace token from ~/.cache/huggingface/token"
    else
        echo "[WARNING] No HuggingFace token found!"
        echo "         The DINOv3 model is gated and requires authentication."
        echo "         Please either:"
        echo "         1. Create a .env file with: HF_TOKEN=your_token"
        echo "         2. Export HF_TOKEN=your_token before running"
        echo "         3. Run 'huggingface-cli login' to cache credentials"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Create outputs directory
mkdir -p outputs

# Stop any existing container with the same name
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "[INFO] Starting TRELLIS.2 Unity Studio Server"
echo "[INFO] Memory mode: $MEMORY_MODE"
echo "[INFO] Container name: $CONTAINER_NAME"
echo "[INFO] Image: $IMAGE_NAME"

# Run the container
docker run \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --rm \
    $DETACH \
    -p 8000:8000 \
    -p 7860:7860 \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e MEMORY_MODE="$MEMORY_MODE" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTHONPATH=/app/src:/app/vendor/TRELLIS.2 \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    $ENV_ARGS \
    --shm-size=8g \
    $EXTRA_ARGS \
    "$IMAGE_NAME"

if [[ -z "$DETACH" ]]; then
    echo "[INFO] Server stopped"
else
    echo "[INFO] Server started in background"
    echo "       View logs: docker logs -f $CONTAINER_NAME"
    echo "       Stop: docker stop $CONTAINER_NAME"
    echo "       Health: curl http://localhost:8000/health"
fi
