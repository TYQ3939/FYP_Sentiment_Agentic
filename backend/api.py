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
        background_tasks.add_task(process_scraping_job, job_id, request.topic, subreddits)

        print(f"📌 Job {job_id} queued for background processing")

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
    print(f"🚀 Starting Sentiment Analysis Backend API")
    print(f"{'='*60}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/health")
    print(f"{'='*60}\n")
    
    print("✅ Backend runs independently from Streamlit")
    print("✅ All scraping happens in isolated background threads")
    print("✅ No asyncio conflicts with Streamlit\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,  # NO reload in production
        log_level="info"
    )

if __name__ == "__main__":
    run_server()