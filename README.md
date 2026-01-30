<p align="center">
  <img src="assets/examples/image.png" alt="TRELLIS.2 Unity Studio" width="600"/>
</p>

<h1 align="center">TRELLIS.2 Unity Studio</h1>

<p align="center">
  <b>AI-Powered 3D Generation for Unity</b><br>
  Generate high-quality 3D assets from text or images directly in Unity Editor via Flux2 and Trellis.2
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#unity-setup">Unity Setup</a> •
  <a href="#server-setup">Server Setup</a> •
  <a href="#api-reference">API Reference</a>
</p>

---

## Overview

**TRELLIS.2 Unity Studio** provides accessible SOTA 3D generation directly within Unity. Generating game assets with "easy" setups (Much easier than official instruction I hope).

**Key Features:**
- **Native Unity Integration** — Editor window for generation workflow, supporting both image-to-3d and text-to-3d via Flux 2.
- **Quality Presets** — Fast (60s), Balanced (90s), High (180s)
- **Auto-Import** — Generated GLBs import directly to project
- **Web Interface** — Optional Gradio UI for standalone use

---

## Quick Start

### 1. Clone Repository
```bash
git clone --recursive https://github.com/your-org/trellis2-unity-studio.git
cd trellis2-unity-studio
```

### 2. Start the Server
```bash
# Setup (first time only)
./scripts/setup.sh

# Start API server
python src/trellis2_server.py
```
Server runs at `http://localhost:8000`

### 3. Setup Unity Project
```
1. Copy unity/ folder contents to: Assets/Trellis2/
2. Open Tools > TRELLIS.2 > Generation Window
3. Enter prompt or assign image → Click Generate
4. Model auto-imports to Assets/Trellis2Results/
```

---

## Unity Setup

### Installation

1. **Copy Unity Package**
   ```
   trellis2-unity-studio/unity/  →  YourProject/Assets/Trellis2/
   ```
   
   Files to copy:
   - `Trellis2Client.cs` — Runtime client for API calls
   - `Trellis2Demo.cs` — Example usage component
   - `Editor/Trellis2Window.cs` — Editor window

2. **Install GLB Loader** (recommended)
   - [GLTFUtility](https://github.com/Siccity/GLTFUtility) — Simple, lightweight
   - [UnityGLTF](https://github.com/KhronosGroup/UnityGLTF) — Khronos official

### Editor Window

Open **Tools > TRELLIS.2 > Generation Window**

| Setting | Description |
|---------|-------------|
| Server URL | API endpoint (default: `http://localhost:8000`) |
| Quality | Fast / Balanced / High |
| Seed | Random seed (-1 for random) |
| Auto-Add to Scene | Spawn model on completion |

**Text-to-3D:** Enter prompt → Generate from Text  
**Image-to-3D:** Assign texture → Generate from Image

### Runtime Usage

```csharp
using UnityEngine;
using Trellis2;

public class Generator : MonoBehaviour
{
    Trellis2Client client;

    void Start()
    {
        client = gameObject.AddComponent<Trellis2Client>();
        client.serverUrl = "http://localhost:8000";
        client.quality = GenerationQuality.Balanced;
        client.OnGenerationComplete += OnComplete;
    }

    public void Generate()
    {
        // From text
        client.GenerateFromText("A cute robot toy");
        
        // From texture
        // client.GenerateFromTexture(myTexture);
        
        // From file
        // client.GenerateFromImageFile("/path/to/image.png");
    }

    void OnComplete(GenerationResult result)
    {
        Debug.Log($"GLB: {result.localGlbPath}");
        // Load with GLTFUtility:
        // var model = Importer.LoadFromFile(result.localGlbPath);
    }
}
```

### Coroutine Usage

```csharp
IEnumerator GenerateModel()
{
    GenerationResult result = null;
    
    yield return client.GenerateFromTextCoroutine(
        "A red sports car",
        r => result = r
    );
    
    if (string.IsNullOrEmpty(result.error))
        Debug.Log($"Success: {result.localGlbPath}");
}
```

---

## Server Setup

### Requirements

- **GPU:** NVIDIA with 24GB+ VRAM (A100/H100 recommended, but also tested on RTX4080/5080)
- **CUDA:** 12.4 or compatible
- **Python:** 3.8+
- **OS:** Linux
- **HuggingFace**: Trellis.2 include several HF dependencies like DinoV3, so be sure to login HF with your created token, and request access to: (1)[Flux.2](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), (2)[RMBG2.0](https://huggingface.co/briaai/RMBG-2.0), (3)[DinoV3](https://huggingface.co/collections/facebook/dinov3). 

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/trellis2-unity-studio.git
cd trellis2-unity-studio

# Run setup
./scripts/setup.sh

# Or manual setup:
conda create -n trellis2 python=3.10 -y
conda activate trellis2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd vendor/TRELLIS.2 && bash setup.sh --basic --flash-attn --nvdiffrast --cumesh --o-voxel
```

### Running the Server

**API Server (for Unity):**
```bash
cd src
uvicorn trellis2_server:app --host 0.0.0.0 --port 8000
```

**Web Interface (optional):**
```bash
python app.py --port 7860
```

### Docker

```bash
docker-compose up -d
```

Server available at `http://localhost:8000`

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/submit/text` | Submit text-to-3D job |
| `POST` | `/submit/image` | Submit image-to-3D job |
| `GET` | `/status/{job_id}` | Get job status |
| `GET` | `/result/{job_id}` | Get job result |
| `GET` | `/jobs` | List all jobs |
| `DELETE` | `/jobs/{job_id}` | Delete job |

### Text-to-3D

```bash
curl -X POST http://localhost:8000/submit/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cute robot toy", "quality": "balanced", "seed": 42}'
```

Response:
```json
{"job_id": "abc123", "status": "queued"}
```

### Image-to-3D

```bash
curl -X POST http://localhost:8000/submit/image \
  -F "file=@input.png" \
  -F "quality=balanced" \
  -F "seed=42"
```

### Check Status

```bash
curl http://localhost:8000/status/{job_id}
```

Response:
```json
{
  "job_id": "abc123",
  "status": "done",
  "result": {
    "glb": "download/abc123/model.glb",
    "image": "download/abc123/image.png"
  }
}
```

### Quality Presets

| Preset | Time | Resolution | Use Case |
|--------|------|------------|----------|
| `fast` | ~60s | 512³ | Quick iterations, mobile |
| `balanced` | ~90s | 1024³ | General use, games |
| `high` | ~180s | 1536³ | Hero assets, cinematics |

---

## Project Structure

```
trellis2-unity-studio/
├── app.py                 # Gradio web interface
├── requirements.txt       # Python dependencies
├── src/
│   ├── trellis2_server.py # FastAPI server
│   └── trellis2_wrapper.py# Generation wrapper
├── unity/
│   ├── Trellis2Client.cs  # Unity runtime client
│   ├── Trellis2Demo.cs    # Example component
│   └── Editor/
│       └── Trellis2Window.cs  # Editor window
├── scripts/
│   └── setup.sh           # Installation script
├── vendor/
│   └── TRELLIS.2/         # Core ML model (submodule)
└── outputs/               # Generated files
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Server not responding | Check `http://localhost:8000/health` |
| CUDA out of memory | Use `fast` quality or lower resolution |
| GLB not loading in Unity | Install GLTFUtility or UnityGLTF |
| Slow generation | Use `fast` preset for iterations |
| Connection refused | Ensure server is running, check firewall |

### Common Commands

```bash
# Check server health
curl http://localhost:8000/health

# View server logs
tail -f outputs/server.log

# Clear generated files
rm -rf outputs/*
```

---

## License

MIT License — see [LICENSE](LICENSE)

Built on [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT License)

---

<p align="center">
  <b>Generate 3D assets with AI, directly in Unity</b><br>
  <a href="https://github.com/your-org/trellis2-unity-studio">GitHub</a> •
  <a href="https://github.com/your-org/trellis2-unity-studio/issues">Issues</a>
</p>
