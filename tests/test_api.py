"""Basic API tests for TRELLIS.2 Unity Studio server."""
import requests
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test server health endpoint."""
    r = requests.get(f"{API_URL}/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"

def test_text_generation():
    """Test text-to-3D job submission."""
    # Submit job
    r = requests.post(f"{API_URL}/submit/text", json={
        "prompt": "A cube",
        "quality": "fast",
        "seed": 42
    })
    assert r.status_code == 200
    job_id = r.json().get("job_id")
    assert job_id
    
    # Poll status
    for _ in range(120):  # 2 min timeout
        r = requests.get(f"{API_URL}/status/{job_id}")
        status = r.json().get("status")
        if status == "done":
            break
        elif status == "error":
            assert False, f"Job failed: {r.json()}"
        time.sleep(1)
    
    assert status == "done"

if __name__ == "__main__":
    test_health()
    print("✓ Health check passed")
    test_text_generation()
    print("✓ Text generation passed")
