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


# ========== RCA ENDPOINT ==========

class RCARequest(BaseModel):
    date: str  # "YYYY-MM-DD"

@app.post("/rca/{job_id}")
async def run_rca_analysis(job_id: str, request: RCARequest):
    """
    Run root cause analysis for a specific anomaly date in a completed job.

    Pulls topic, aspect_analysis, and detailed_sentiments from the job's
    saved state, runs the full RCA pipeline (web search + LLM evaluation),
    caches the result back into the job record, and returns it.
    """
    import os
    from langchain_groq import ChatGroq
    from tools.rca_tools import run_rca

    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    spike_date = request.date

    # Check cache first
    rca_cache = job.get("rca_cache", {})
    if spike_date in rca_cache:
        return rca_cache[spike_date]

    # Extract required data from stored state
    state = job.get("results", {}).get("state", {})
    topic = state.get("metadata", {}).get("topic", "")
    aspect_analysis = state.get("aspect_analysis", {})
    detailed_sentiments = (
        state.get("sentiment_results", {}).get("detailed_sentiments", [])
    )

    if not topic:
        raise HTTPException(status_code=400, detail="Topic not found in job state")

    # Build LLM (same Groq model used by all agents)
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_key,
    )

    try:
        result = run_rca(topic, spike_date, aspect_analysis, detailed_sentiments, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RCA pipeline error: {str(e)[:200]}")

    # Cache in job record so repeated clicks are instant
    rca_cache[spike_date] = result
    db.update_job(job_id, rca_cache=rca_cache)

    # Also write to shared_state.json so AdvisorAgent.answer_question() can include
    # root cause context when the user asks follow-up questions in the chat
    try:
        import json
        state_file = "shared_state.json"
        if os.path.exists(state_file):
            with open(state_file, "r") as _f:
                _state = json.load(_f)
        else:
            _state = {}
        _state.setdefault("rca_cache", {})[spike_date] = result
        with open(state_file, "w") as _f:
            json.dump(_state, _f, indent=2)
    except Exception:
        pass  # Non-critical; advisor will still work without it

    return result

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