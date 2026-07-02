"""FastAPI backend for sentiment analysis agents."""

import sys
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from backend.database import db
from backend.tasks import process_scraping_job

# Pydantic models for request/response
class ScrapeRequest(BaseModel):
    topic: str
    subreddits: Optional[List[str]] = None
    mode: str = "single"   # "single" or "compare"

class RCARequest(BaseModel):
    date: str  # YYYY-MM-DD of the anomaly to investigate

class JobStatusLite(BaseModel):
    """Lightweight job status without large results payload"""
    id: str
    status: str
    progress: int
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class JobStatus(BaseModel):
    """Full job status with results"""
    id: str
    status: str
    progress: int
    results: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Backend API for multi-agent sentiment analysis system",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_checks():
    """Validate all required files exist before accepting any requests."""
    import os
    ok = True

    print("\n" + "="*60)
    print(" STARTUP CHECKS")
    print("="*60)

    # Check model files
    model_dir = "tools/models/bertweet_finetuned"
    required_files = {
        "model.safetensors": "Model weights",
        "config.json":       "Model config",
        "tokenizer_config.json": "Tokenizer config",
        "vocab.txt":         "Vocabulary",
    }
    for fname, label in required_files.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            print(f" [OK]   {label}: {fname}")
        else:
            print(f" [MISS] {label} MISSING: {path}")
            ok = False

    # Check .env API keys
    required_keys = ["GROQ_API_KEY"]
    for key in required_keys:
        if os.environ.get(key):
            print(f" [OK]   Env: {key}")
        else:
            print(f" [WARN] Env: {key} not set — Advisor Agent will fail")

    # Check data directories
    for d in ["data/filtered_data", "data/analysis", "data/raw_data"]:
        os.makedirs(d, exist_ok=True)

    print("="*60)
    if not ok:
        print(" [WARN] Some model files are missing — analysis jobs will fail.")
        print("        Upload missing files to:", os.path.abspath(model_dir))
    else:
        print(" [OK]  All checks passed — backend ready.")
    print("="*60 + "\n")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware to reduce response size and speed up transmission
#app.add_middleware(GZipMiddleware, minimum_size=1000)

# ========== ROUTES ==========

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/scrape/start")
async def start_scrape(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Start a new scraping job using FastAPI BackgroundTasks.

    This endpoint returns IMMEDIATELY with a "running" status while the
    scraping job executes in the background. This prevents HTTP timeouts
    and blocks the main server thread.

    Args:
        request: ScrapeRequest with topic and optional subreddits
        background_tasks: FastAPI BackgroundTasks for async execution

    Returns:
        Job ID and immediate response (job runs in background)
    """

    try:
        # Validate input
        if not request.topic or request.topic.strip() == "":
            raise HTTPException(status_code=400, detail="Topic cannot be empty")

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Use provided subreddits or default
        subreddits = request.subreddits or ["iphone", "apple", "technology"]

        # Create job in database (synchronous, fast operation)
        db.create_job(job_id, request.topic, subreddits)

        # Queue heavy processing in background task (NOT blocking the endpoint)
        # This returns immediately while the task runs separately
        background_tasks.add_task(process_scraping_job, job_id, request.topic, subreddits, request.mode)

        print(f"[QUEUED] Job {job_id} queued for background processing")

        return {
            "job_id": job_id,
            "status": "running",
            "message": f"Job {job_id} started for topic: {request.topic}",
            "next": f"Poll /scrape/status/{job_id} to check progress"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scrape/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatusLite:
    """
    Get the lightweight status of a scraping job (non-blocking).
    This endpoint returns status information WITHOUT the large results payload.
    Use /scrape/results/{job_id} to fetch the full results.

    Args:
        job_id: Job identifier

    Returns:
        Lightweight job status (without results)
    """

    job = db.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Return only lightweight status fields (exclude large results)
    return JobStatusLite(
        id=job['id'],
        status=job['status'],
        progress=job['progress'],
        error=job.get('error'),
        created_at=job['created_at'],
        started_at=job.get('started_at'),
        completed_at=job.get('completed_at')
    )

@app.get("/scrape/results/{job_id}")
async def get_job_results(job_id: str) -> JobStatus:
    """
    Get the FULL results of a completed scraping job (includes large results payload).
    This is a separate endpoint because results can be very large.

    Args:
        job_id: Job identifier

    Returns:
        Full job status with results
    """

    job = db.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(**job)

@app.get("/scrape/jobs")
async def list_jobs():
    """List all jobs with their statuses."""
    return {
        "jobs": db.list_jobs(),
        "total": len(db.list_jobs())
    }

@app.get("/scrape/jobs/{status}")
async def list_jobs_by_status(status: str):
    """
    List jobs by status.
    
    Args:
        status: Job status (pending, running, completed, error)
    
    Returns:
        List of jobs with matching status
    """
    
    all_jobs = db.list_jobs()
    filtered_jobs = [job for job in all_jobs if job["status"] == status]
    
    return {
        "status": status,
        "jobs": filtered_jobs,
        "total": len(filtered_jobs)
    }

@app.post("/rca/{job_id}")
async def run_rca_analysis(job_id: str, request: RCARequest):
    """
    Trigger Root Cause Analysis for a specific anomaly date in a completed job.
    Results are cached in the job record so repeated calls for the same date
    are instant.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before running RCA")

    spike_date = request.date

    # Return cached result if available
    rca_cache = job.get("rca_cache", {})
    if spike_date in rca_cache:
        return rca_cache[spike_date]

    try:
        results  = job.get("results", {})
        analysis = results.get("analyst", {}).get("analysis", {})
        metadata = results.get("scraper", {}).get("summary", {})

        topic              = metadata.get("topic") or job.get("topic", "unknown")
        detailed_sentiments = analysis.get("detailed_sentiments", [])
        aspect_analysis    = results.get("analyst", {}).get("aspect_analysis", {})

        if not detailed_sentiments:
            # Try loading from shared state
            import json, os
            state_path = "./data/shared_state.json"
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                detailed_sentiments = state.get("sentiment_results", {}).get("detailed_sentiments", [])
                aspect_analysis     = state.get("aspect_analysis", aspect_analysis)
                if not topic or topic == "unknown":
                    topic = state.get("metadata", {}).get("topic", "unknown")

        from langchain_groq import ChatGroq
        import os as _os
        groq_key = _os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=groq_key)

        from tools.rca_tools import run_rca
        result = run_rca(topic, spike_date, aspect_analysis, detailed_sentiments, llm)

        # Cache result
        rca_cache[spike_date] = result
        db.update_job(job_id, rca_cache=rca_cache)

        # Also write to shared_state for AdvisorAgent context
        try:
            import json, os
            state_path = "./data/shared_state.json"
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                state.setdefault("rca_cache", {})[spike_date] = result
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/scrape/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job."""
    job = db.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Remove from database
    db.jobs.pop(job_id, None)
    db.save_to_file()
    
    return {"message": f"Job {job_id} deleted"}

# ========== MAIN ==========

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    print(f"\n{'='*60}")
    print(f"[START] Sentiment Analysis Backend API")
    print(f"{'='*60}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/health")
    print(f"{'='*60}\n")

    print("[OK] Backend runs independently from Streamlit")
    print("[OK] All scraping happens in isolated background threads")
    print("[OK] No asyncio conflicts with Streamlit\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,  # NO reload in production
        log_level="info"
    )

if __name__ == "__main__":
    run_server()