# Docker Setup for TRELLIS.2 Unity Studio

This folder contains Docker configuration for running TRELLIS.2 Unity Studio in a containerized environment.

## Files

- **Dockerfile** — Main Docker image configuration
- **docker-compose.yml** — Docker Compose configuration (deprecated - use manual Docker commands instead)
- **.dockerignore** — Files excluded from build context

## Quick Start

### 1. Prerequisites

- Docker 20.10+
- NVIDIA Container Toolkit
- HuggingFace account with token

### 2. Setup Environment

```bash
# From repository root
cd /path/to/trellis2-unity-studio

# Create .env file with your HuggingFace token
cp .env.example .env
nano .env  # Add HF_TOKEN=hf_xxxxx
```

### 3. Build Image

```bash
docker build -f docker/Dockerfile -t trellis2-unity-studio .
```

Build time: ~10-20 minutes (downloads ~10GB dependencies)

### 4. Run Container

**Using helper script (recommended):**
```bash
../run_docker.sh --detach
```

**Manual run:**
```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env-file .env \
  -e MEMORY_MODE=auto \
  --shm-size=8g \
  --name trellis2-unity-studio \
  -d \
  trellis2-unity-studio
```

### 5. Verify

```bash
# Check health
curl http://localhost:8000/health

# View logs
docker logs -f trellis2-unity-studio

# Stop
docker stop trellis2-unity-studio
```

## Image Details

### Base Image
- **nvidia/cuda:12.4.1-devel-ubuntu22.04**
- Includes CUDA 12.4 development tools for building extensions

### Key Dependencies
- PyTorch 2.6.0 (CUDA 12.4)
- Flash Attention 2.7.3
- Diffusers (from main branch - includes Flux2KleinPipeline)
- TRELLIS.2 (from vendor submodule)
- Custom CUDA extensions: nvdiffrast, CuMesh, FlexGEMM, o-voxel

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace access token |
| `MEMORY_MODE` | `auto` | Memory management: `auto`, `swap`, `keep_loaded` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch CUDA allocator config |
| `ATTN_BACKEND` | `flash_attn` | Attention backend |
| `SPARSE_ATTN_BACKEND` | `flash_attn` | Sparse attention backend |
| `SPARSE_CONV_BACKEND` | `flex_gemm` | Sparse convolution backend |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|----------|
| `./outputs` | `/app/outputs` | Generated 3D models and images |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace model cache (persistent) |

### Ports

| Port | Service |
|------|---------|
| 8000 | FastAPI server (Unity API) |
| 7860 | Gradio web interface (optional) |

## Memory Modes

### Auto (Recommended)
```bash
-e MEMORY_MODE=auto
```
Detects system RAM and chooses optimal mode:
- **<48GB RAM** → swap mode
- **≥48GB RAM** → keep_loaded mode

### Swap Mode
```bash
-e MEMORY_MODE=swap
```
- **RAM Required:** 32GB+
- **Behavior:** Unload Flux before loading TRELLIS.2
- **Performance:** +50s overhead per text-to-3D job
- **Use Case:** Limited RAM systems

### Keep Loaded Mode
```bash
-e MEMORY_MODE=keep_loaded
```
- **RAM Required:** 64GB+
- **Behavior:** Both models stay in memory
- **Performance:** Fastest for repeated generations
- **Use Case:** High RAM systems, production servers

## Troubleshooting

### Build Issues

**Error: `CUDA error: no kernel image available`**
- Ensure NVIDIA drivers are updated
- Verify GPU compute capability matches TORCH_CUDA_ARCH_LIST

**Error: `failed to solve: process "/bin/sh -c pip install flash-attn..."`**
- Flash Attention requires compilation
- Ensure sufficient disk space (~20GB for build)
- Check CUDA toolkit is available in container

### Runtime Issues

**Error: `401 Unauthorized` / `GatedRepoError`**
- Check `.env` file exists and has valid HF_TOKEN
- Verify you have access to gated models
- Restart container: `docker restart trellis2-unity-studio`

**Container crashes with OOM**
- Use swap mode: `-e MEMORY_MODE=swap`
- Increase `--shm-size` to 16g or higher
- Use lower quality presets

**Models download on every restart**
- Ensure HuggingFace cache is mounted: `-v ~/.cache/huggingface:/root/.cache/huggingface`
- Check cache permissions: `ls -la ~/.cache/huggingface`

### Performance

**Slow first generation**
- Models download on first use (~40GB total)
- Subsequent generations use cached models
- Check download progress: `docker logs -f trellis2-unity-studio`

**High memory usage**
- Expected: ~30GB for swap mode, ~50GB for keep_loaded
- Monitor: `docker stats trellis2-unity-studio`

## Development

### Interactive Shell

```bash
docker exec -it trellis2-unity-studio bash
```

### Python REPL

```bash
docker exec -it trellis2-unity-studio python3
```

### Test Imports

```bash
docker exec trellis2-unity-studio python3 -c "import trellis2; import o_voxel; print('OK')"
```

### Rebuild Without Cache

```bash
docker build --no-cache -f docker/Dockerfile -t trellis2-unity-studio .
```

## Advanced Configuration

### Custom Model Paths

```bash
docker run ... \
  -e TRELLIS_MODEL=microsoft/TRELLIS.2-4B \
  -e FLUX_MODEL=black-forest-labs/FLUX.2-klein-4B \
  ...
```

### GPU Selection

```bash
# Use specific GPU
docker run --gpus '"device=0"' ...

# Use multiple GPUs
docker run --gpus '"device=0,1"' ...
```

### Custom Output Directory

```bash
docker run ... \
  -v /custom/output/path:/app/outputs \
  -e OUTPUT_DIR=/app/outputs \
  ...
```

## Security Notes

- **Never commit `.env` files** — Contains sensitive HF_TOKEN
- **Use read-only token** — Minimal "Read" permission recommended
- **Restrict network access** — Firewall port 8000 if not needed externally
- **Keep base image updated** — Regularly rebuild for security patches

## Size Optimization

### Image Size
- **Full image:** ~18GB
- **Compressed:** ~10GB (during download)
- **Build cache:** ~5GB additional

### Reduce Size
```dockerfile
# Add to Dockerfile to clean build artifacts
RUN pip cache purge && \
    rm -rf /tmp/* /var/tmp/* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```

## Support

- **Dockerfile issues:** Check build logs carefully
- **Runtime issues:** View `docker logs trellis2-unity-studio`
- **GitHub Issues:** https://github.com/Scriptwonder/trellis2-unity-studio/issues

---

**Docker configuration maintained for TRELLIS.2 Unity Studio**
