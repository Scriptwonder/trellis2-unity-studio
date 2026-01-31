# TRELLIS.2 Unity Studio - Detailed Setup Guide

This guide provides step-by-step instructions for setting up TRELLIS.2 Unity Studio with Docker.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **NVIDIA GPU** with 16GB+ VRAM (24GB+ recommended)
- [ ] **Docker** with NVIDIA Container Toolkit installed
- [ ] **32GB+ system RAM** (64GB+ for optimal performance)
- [ ] **HuggingFace account** with access to gated models
- [ ] **Ubuntu 22.04** or compatible Linux distribution

---

## Step 1: HuggingFace Setup

### 1.1 Create HuggingFace Account

1. Go to https://huggingface.co and create an account
2. Verify your email address

### 1.2 Create Access Token

1. Navigate to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `trellis2-unity-studio`
4. Type: **Read** (minimum required)
5. Click **"Generate token"**
6. **Copy the token** (starts with `hf_...`)
   - ⚠️ Save this token securely - you won't be able to see it again

### 1.3 Request Model Access

Visit each model page and click **"Request access"**:

1. **DINOv3** (required for image feature extraction)
   - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
   - Click "Agree and access repository"
   
2. **FLUX.2-klein-4B** (required for text-to-image)
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
   - Read and accept the license agreement
   
3. **RMBG-2.0** (optional, for background removal)
   - https://huggingface.co/briaai/RMBG-2.0
   - Accept the license

⏱️ **Access approval may take a few minutes to several hours**

---

## Step 2: Clone Repository

```bash
# Clone with submodules (required for TRELLIS.2 core)
git clone --recursive https://github.com/Scriptwonder/trellis2-unity-studio.git
cd trellis2-unity-studio

# If you already cloned without --recursive:
git submodule update --init --recursive
```

Verify the submodule is loaded:
```bash
ls vendor/TRELLIS.2/trellis2/
# Should show: __init__.py, models/, modules/, pipelines/, etc.
```

---

## Step 3: Configure Environment

### 3.1 Create `.env` File

```bash
# Copy template
cp .env.example .env
```

### 3.2 Add Your HuggingFace Token

Edit `.env` and replace the placeholder:

```bash
# Using nano
nano .env

# Or vim
vim .env

# Or any text editor
code .env
```

Replace:
```
HF_TOKEN=your_huggingface_token_here
```

With your actual token:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**Security Note:**
- ⚠️ Never commit `.env` to git (already in `.gitignore`)
- ⚠️ Don't share your token publicly
- ✅ The `.env` file is automatically loaded by `run_docker.sh`

---

## Step 4: Build Docker Image

```bash
# Build the image (this will take 10-20 minutes)
docker build -f docker/Dockerfile -t trellis2-unity-studio .
```

**What happens during build:**
- Installs CUDA 12.4 base image
- Installs PyTorch 2.6.0 with CUDA support
- Compiles Flash Attention, nvdiffrast, CuMesh, FlexGEMM
- Installs TRELLIS.2 and o-voxel packages
- Configures Python environment

**Expected output:**
```
[+] Building 600.0s (25/25) FINISHED
=> exporting to image
=> => writing image sha256:...
=> => naming to docker.io/library/trellis2-unity-studio
```

---

## Step 5: Run the Server

### Option A: Using Helper Script (Recommended)

```bash
# Run in background
./run_docker.sh --detach

# View logs
docker logs -f trellis2-unity-studio

# Stop server
docker stop trellis2-unity-studio
```

### Option B: Manual Docker Run

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

### Memory Mode Options

```bash
# Auto-detect (recommended)
./run_docker.sh --memory auto

# Force swap mode (32GB+ RAM)
./run_docker.sh --memory swap

# Keep models loaded (64GB+ RAM, fastest)
./run_docker.sh --memory keep
```

---

## Step 6: Verify Installation

### 6.1 Check Server Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy","service":"trellis2"}
```

### 6.2 Check Server Logs

```bash
docker logs trellis2-unity-studio
```

Expected output:
```
[SPARSE] Conv backend: flex_gemm; Attention backend: flash_attn
[INFO] Initializing pipeline...
[INFO] Memory mode: swap
[INFO] Pipeline ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6.3 Test Generation (Optional)

```bash
# Submit a simple test job
curl -X POST http://localhost:8000/submit/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A red cube", "quality": "superfast", "seed": 42}'
```

Expected response:
```json
{"job_id":"abc123-...", "status":"queued"}
```

Check status:
```bash
curl http://localhost:8000/status/abc123-...
```

---

## Step 7: Unity Integration

### 7.1 Copy Unity Package

```bash
# From your Unity project root:
cp -r /path/to/trellis2-unity-studio/unity/* Assets/Trellis2/
```

### 7.2 Install GLB Loader

Install one of these packages via Unity Package Manager:

- **GLTFUtility** (recommended): 
  ```
  https://github.com/Siccity/GLTFUtility.git
  ```
  
- **UnityGLTF** (official Khronos):
  ```
  https://github.com/KhronosGroup/UnityGLTF.git?path=/UnityGLTF/Assets/UnityGLTF
  ```

### 7.3 Open Generation Window

In Unity Editor:
1. **Tools → TRELLIS.2 → Generation Window**
2. Set **Server URL**: `http://localhost:8000`
3. Test connection (should show green checkmark)

---

## Troubleshooting

### Issue: `401 Unauthorized` or `GatedRepoError`

**Cause:** HuggingFace token not configured or access not granted

**Solution:**
1. Check `.env` file has correct token format: `HF_TOKEN=hf_...`
2. Verify you requested access to all required models
3. Restart container: `docker restart trellis2-unity-studio`
4. Check logs: `docker logs trellis2-unity-studio`

### Issue: `Cannot access gated repo`

**Cause:** Model access not approved yet

**Solution:**
1. Visit model pages and check approval status
2. Wait for approval (can take hours)
3. Verify token has "Read" permission at https://huggingface.co/settings/tokens

### Issue: Container crashes with OOM (Out of Memory)

**Cause:** Insufficient system RAM

**Solution:**
1. Use swap mode: `./run_docker.sh --memory swap`
2. Use lower quality preset: `superfast` or `fast`
3. Close other applications
4. Check available RAM: `free -h`

### Issue: Build fails during compilation

**Cause:** Missing dependencies or CUDA incompatibility

**Solutions:**
1. Ensure NVIDIA drivers are up to date
2. Verify CUDA 12.4 compatibility
3. Clean build: `docker build --no-cache -f docker/Dockerfile .`
4. Check Docker has GPU access: `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### Issue: Models download slowly

**Cause:** First-time model downloads are large (~40GB total)

**Solution:**
- Be patient - first run downloads all models
- Use cached models: `-v ~/.cache/huggingface:/root/.cache/huggingface`
- Check download progress in logs

---

## Performance Tuning

### Memory Modes

| Mode | RAM | Description | Best For |
|------|-----|-------------|----------|
| `auto` | Any | Auto-detects based on available RAM | Default |
| `swap` | 32GB+ | Unload Flux before loading TRELLIS.2 | Limited RAM |
| `keep_loaded` | 64GB+ | Keep both models in memory | Maximum speed |

### Quality vs Speed

| Preset | Time | Use Case |
|--------|------|----------|
| `superfast` | ~15s | Rapid prototyping |
| `fast` | ~60s | Iteration, mobile assets |
| `balanced` | ~90s | Production, PC games |
| `high` | ~180s | Hero assets, cinematics |

---

## Next Steps

✅ Server is running at `http://localhost:8000`  
✅ Unity integration is ready  

**Try generating your first asset:**
1. Open Unity: **Tools → TRELLIS.2 → Generation Window**
2. Enter prompt: "A cute robot toy"
3. Quality: **Balanced**
4. Click **Generate from Text**
5. Wait ~90 seconds
6. Model auto-imports to `Assets/Trellis2Results/`

**Explore the API:**
- View API docs: `http://localhost:8000/docs`
- See example scripts in `examples/`

---

## Support

- **Issues:** https://github.com/Scriptwonder/trellis2-unity-studio/issues
- **TRELLIS.2 Core:** https://github.com/microsoft/TRELLIS.2
- **HuggingFace Help:** https://huggingface.co/docs

---

**Happy generating! 🎨🤖**
