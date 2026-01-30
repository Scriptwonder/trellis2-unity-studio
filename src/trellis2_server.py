"""
TRELLIS.2 + Flux FastAPI Server
Provides REST API for text-to-3D and image-to-3D generation.

Endpoints:
    POST /submit/text       - Submit text-to-3D job
    POST /submit/image      - Submit image-to-3D job
    GET  /status/{job_id}   - Get job status
    GET  /result/{job_id}   - Get job result (GLB download path)
    GET  /health            - Health check
"""
import io
import os
import threading
import uuid
from typing import Dict, Optional, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from trellis2_wrapper import run_text_to_3d, run_image_to_3d

app = FastAPI(
    title="TRELLIS.2 API",
    description="Text-to-3D and Image-to-3D generation using Flux + TRELLIS.2",
    version="1.0.0",
)

# Allow all origins for Unity/remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
JOBS: Dict[str, Dict] = {}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _as_download_path(path: str) -> str:
    """Convert absolute path to download URL using /download endpoint."""
    rel_path = os.path.relpath(path, OUTPUT_DIR)
    normalized = rel_path.replace(os.sep, "/")
    return f"download/{normalized}"


# =============================================================================
# Request/Response Models
# =============================================================================

class TextSubmitRequest(BaseModel):
    """Request body for text-to-3D submission."""
    prompt: str = Field(..., description="Text prompt describing the 3D object")
    quality: Literal["superfast", "fast", "balanced", "high"] = Field(
        default="balanced",
        description="Quality preset: superfast (~15s), fast (~60s), balanced (~90s), high (~180s)"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class ImageSubmitRequest(BaseModel):
    """Query parameters for image-to-3D submission."""
    quality: Literal["superfast", "fast", "balanced", "high"] = "balanced"
    seed: int = 42


class JobResponse(BaseModel):
    """Response for job submission."""
    job_id: str
    status: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    type: str
    status: str
    prompt: Optional[str] = None
    quality: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    timings: Optional[dict] = None


# =============================================================================
# Job Runner
# =============================================================================

def _run_job(job_id: str, image_data: Optional[bytes] = None) -> None:
    """Background job runner."""
    job = JOBS[job_id]
    job["status"] = "running"
    job["stage"] = "starting"
    job["stage_description"] = "Starting job"
    output_path = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_path, exist_ok=True)

    def _on_progress(stage: str, description: str):
        job["stage"] = stage
        job["stage_description"] = description

    try:
        if job["type"] == "text":
            result = run_text_to_3d(
                prompt=job["prompt"],
                output_dir=output_path,
                output_name="model",
                quality=job.get("quality", "balanced"),
                seed=job.get("seed", 42),
                on_progress=_on_progress,
            )
            job["result"] = {
                "glb": _as_download_path(result.glb_path),
                "image": _as_download_path(result.image_path) if result.image_path else None,
            }
            job["timings"] = result.timings

        elif job["type"] == "image":
            if image_data is None:
                raise RuntimeError("Image data missing for job")

            image = Image.open(io.BytesIO(image_data))
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")

            result = run_image_to_3d(
                image=image,
                output_dir=output_path,
                output_name="model",
                quality=job.get("quality", "balanced"),
                seed=job.get("seed", 42),
                on_progress=_on_progress,
            )
            job["result"] = {
                "glb": _as_download_path(result.glb_path),
            }
            job["timings"] = result.timings

        else:
            raise RuntimeError(f"Unsupported job type: {job['type']}")

        job["status"] = "done"
        job["stage"] = "complete"
        job["stage_description"] = "Complete"

    except Exception as exc:
        job["status"] = "failed"
        job["stage"] = "error"
        job["error"] = str(exc)
        import traceback
        traceback.print_exc()


def _enqueue_job(job_payload: Dict, worker_args: tuple = ()) -> str:
    """Enqueue a job for background processing."""
    job_id = str(uuid.uuid4())
    job_payload["job_id"] = job_id
    JOBS[job_id] = job_payload
    threading.Thread(target=_run_job, args=(job_id, *worker_args), daemon=True).start()
    return job_id


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "trellis2"}


@app.post("/submit/text", response_model=JobResponse)
def submit_text(request: TextSubmitRequest):
    """
    Submit a text-to-3D generation job.

    The job runs asynchronously. Use /status/{job_id} to check progress
    and /result/{job_id} to get the download URL when complete.
    """
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    job_id = _enqueue_job({
        "type": "text",
        "prompt": prompt,
        "quality": request.quality,
        "seed": request.seed,
        "status": "queued",
    })
    return {"job_id": job_id, "status": "queued"}


@app.post("/submit/image", response_model=JobResponse)
async def submit_image(
    file: UploadFile = File(..., description="Input image file"),
    quality: Literal["fast", "balanced", "high"] = Query(default="balanced"),
    seed: int = Query(default=42),
):
    """
    Submit an image-to-3D generation job.

    Upload an image and the API will generate a 3D model from it.
    Use /status/{job_id} to check progress.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file provided")

    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="empty file")

    # Validate it's a valid image
    try:
        img = Image.open(io.BytesIO(image_data))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    job_id = _enqueue_job(
        {
            "type": "image",
            "filename": file.filename,
            "quality": quality,
            "seed": seed,
            "status": "queued",
        },
        worker_args=(image_data,),
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """
    Get the status of a job.

    Returns job details including status (queued, running, done, failed),
    and result/error information when available.
    """
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return job


@app.get("/result/{job_id}")
def get_result(job_id: str):
    """
    Get the result of a completed job.

    Returns download URLs for generated files (GLB, image).
    Only available when job status is 'done'.
    """
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    if job.get("status") != "done":
        return JSONResponse(
            status_code=202,
            content={"error": "job not ready", "status": job.get("status")}
        )
    return {
        "result": job.get("result"),
        "timings": job.get("timings"),
    }


@app.get("/jobs")
def list_jobs(
    status: Optional[str] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, le=100),
):
    """List recent jobs, optionally filtered by status."""
    jobs = list(JOBS.values())
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    # Return most recent first
    jobs = sorted(jobs, key=lambda x: x.get("job_id", ""), reverse=True)[:limit]
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str):
    """
    Download a generated file (GLB or image).
    This endpoint goes through CORS middleware unlike StaticFiles.
    """
    from fastapi.responses import FileResponse

    # Sanitize filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, job_id, safe_filename)

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "file not found"})

    return FileResponse(
        file_path,
        filename=safe_filename,
        media_type="application/octet-stream",
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job and its outputs."""
    if job_id not in JOBS:
        return JSONResponse(status_code=404, content={"error": "job not found"})

    job = JOBS.pop(job_id)

    # Clean up output directory
    output_path = os.path.join(OUTPUT_DIR, job_id)
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path, ignore_errors=True)

    return {"deleted": job_id}


# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    print("=" * 60)
    print("TRELLIS.2 API Server")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Endpoints:")
    print("  POST /submit/text   - Text-to-3D")
    print("  POST /submit/image  - Image-to-3D")
    print("  GET  /status/{id}   - Job status")
    print("  GET  /result/{id}   - Job result")
    print("  GET  /health        - Health check")
    print("=" * 60)
