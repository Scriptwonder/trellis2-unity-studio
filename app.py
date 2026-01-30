#!/usr/bin/env python3
"""
TRELLIS.2 Unity Studio - Web Interface

Gradio-based web interface for 3D generation.
This is a complementary interface to the Unity Editor integration.
"""

import gradio as gr
import os
import sys
from pathlib import Path
import argparse
import requests
from datetime import datetime
import time

# Configuration
API_URL = os.environ.get("TRELLIS2_API_URL", "http://localhost:8000")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def check_server():
    """Check if the API server is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def submit_text_job(prompt: str, quality: str, seed: int):
    """Submit a text-to-3D job."""
    resp = requests.post(
        f"{API_URL}/submit/text",
        json={"prompt": prompt, "quality": quality.lower(), "seed": seed}
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def submit_image_job(image_path: str, quality: str, seed: int):
    """Submit an image-to-3D job."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{API_URL}/submit/image",
            files={"file": f},
            params={"quality": quality.lower(), "seed": seed}
        )
    resp.raise_for_status()
    return resp.json()["job_id"]


def wait_for_job(job_id: str, progress=gr.Progress()):
    """Wait for job completion with progress updates."""
    start = time.time()
    while True:
        resp = requests.get(f"{API_URL}/status/{job_id}")
        data = resp.json()
        
        status = data.get("status", "unknown")
        elapsed = time.time() - start
        stage = data.get("stage_description", status)
        
        progress(0.5, desc=f"{stage} ({elapsed:.0f}s)")
        
        if status == "done":
            return data
        elif status == "failed":
            raise Exception(data.get("error", "Job failed"))
        
        time.sleep(2)


def download_result(job_data: dict):
    """Download the GLB result."""
    result = job_data.get("result", {})
    glb_url = result.get("glb")
    
    if not glb_url:
        raise Exception("No GLB in result")
    
    job_id = job_data["job_id"]
    local_path = OUTPUT_DIR / f"{job_id}.glb"
    
    resp = requests.get(f"{API_URL}/{glb_url}")
    resp.raise_for_status()
    
    with open(local_path, "wb") as f:
        f.write(resp.content)
    
    return str(local_path)


def generate_from_text(prompt: str, quality: str, seed: int, progress=gr.Progress()):
    """Generate 3D model from text prompt."""
    if not check_server():
        raise gr.Error("Server not running. Start with: python src/trellis2_server.py")
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")
    
    progress(0.1, desc="Submitting job...")
    job_id = submit_text_job(prompt, quality, seed)
    
    progress(0.2, desc="Processing...")
    job_data = wait_for_job(job_id, progress)
    
    progress(0.9, desc="Downloading...")
    glb_path = download_result(job_data)
    
    progress(1.0, desc="Done!")
    return glb_path, f"✅ Generated: {Path(glb_path).name}"


def generate_from_image(image, quality: str, seed: int, progress=gr.Progress()):
    """Generate 3D model from image."""
    if not check_server():
        raise gr.Error("Server not running. Start with: python src/trellis2_server.py")
    
    if image is None:
        raise gr.Error("Please upload an image")
    
    # Save temp image
    temp_path = OUTPUT_DIR / "temp_input.png"
    image.save(temp_path)
    
    progress(0.1, desc="Submitting job...")
    job_id = submit_image_job(str(temp_path), quality, seed)
    
    progress(0.2, desc="Processing...")
    job_data = wait_for_job(job_id, progress)
    
    progress(0.9, desc="Downloading...")
    glb_path = download_result(job_data)
    
    progress(1.0, desc="Done!")
    return glb_path, f"✅ Generated: {Path(glb_path).name}"


# Build interface
with gr.Blocks(title="TRELLIS.2 Unity Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎮 TRELLIS.2 Unity Studio
    **Web Interface for 3D Generation**
    
    For Unity integration, use the Editor window: `Tools > TRELLIS.2 > Generation Window`
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Text to 3D"):
                    prompt = gr.Textbox(label="Prompt", placeholder="A cute robot toy")
                    text_btn = gr.Button("Generate", variant="primary")
                
                with gr.Tab("Image to 3D"):
                    image = gr.Image(type="pil", label="Input Image")
                    image_btn = gr.Button("Generate", variant="primary")
            
            with gr.Accordion("Settings", open=False):
                quality = gr.Radio(["Fast", "Balanced", "High"], value="Balanced", label="Quality")
                seed = gr.Number(value=42, label="Seed", precision=0)
        
        with gr.Column():
            output = gr.File(label="Generated GLB")
            status = gr.Textbox(label="Status", interactive=False)
    
    gr.Markdown("""
    ---
    **Server:** Make sure the API server is running: `uvicorn src.trellis2_server:app --port 8000`
    """)
    
    # Events
    text_btn.click(generate_from_text, [prompt, quality, seed], [output, status])
    image_btn.click(generate_from_image, [image, quality, seed], [output, status])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("TRELLIS.2 Unity Studio - Web Interface")
    print(f"{'='*50}")
    print(f"Web UI: http://localhost:{args.port}")
    print(f"API Server: {API_URL}")
    print(f"{'='*50}\n")
    
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
