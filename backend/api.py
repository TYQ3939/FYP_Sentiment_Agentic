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

class AdvisorQuestionRequest(BaseModel):
    question: str

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
    """Check all pipeline dependencies at startup so problems surface immediately.

    Uses importlib.util.find_spec() for package presence checks (no actual
    import cost) and reserves real imports only for checks that need runtime
    state (CUDA, spaCy model path, StaticEmbedding, NLTK corpus).
    """
    import os
    import importlib.util

    W = 62
    issues = []
    warnings_ = []

    def ok(label):   print(f"  [OK]   {label}")
    def miss(label): print(f"  [MISS] {label}"); issues.append(label)
    def warn(label): print(f"  [WARN] {label}"); warnings_.append(label)

    def pkg(name):
        """Return True if package is installed (no import cost)."""
        return importlib.util.find_spec(name) is not None

    print("\n" + "=" * W)
    print("  STARTUP CHECKS")
    print("=" * W)

    # ── 1. Model files ────────────────────────────────────────
    print("\n  [1] BERTweet Model Files")
    model_dir = "tools/models/bertweet_finetuned"
    for fname in ["model.safetensors", "config.json", "tokenizer_config.json",
                  "vocab.txt", "special_tokens_map.json", "bpe.codes"]:
        if os.path.exists(os.path.join(model_dir, fname)):
            ok(fname)
        else:
            miss(f"{fname} missing in {model_dir}")

    # ── 2. GPU / PyTorch ─────────────────────────────────────
    # Real import needed to query CUDA state
    print("\n  [2] GPU / PyTorch")
    if pkg("torch"):
        try:
            import torch
            ok(f"torch {torch.__version__}")
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                ok(f"CUDA GPU: {gpu} ({mem:.1f} GB)")
            else:
                warn("CUDA not available — BERTweet will run on CPU (slower)")
        except Exception as e:
            warn(f"CUDA check failed: {e}")
    else:
        miss("torch not installed")

    # ── 3. Transformers ───────────────────────────────────────
    print("\n  [3] Transformers")
    if pkg("transformers"):
        ok("transformers")
    else:
        miss("transformers not installed")

    # ── 4. spaCy + English model ──────────────────────────────
    # find_spec checks spaCy; spacy.util.get_package_path checks the model
    print("\n  [4] spaCy")
    if pkg("spacy"):
        import spacy.util
        ok("spacy")
        try:
            spacy.util.get_package_path("en_core_web_sm")
            ok("en_core_web_sm model installed")
        except Exception:
            warn("en_core_web_sm not found — run: python -m spacy download en_core_web_sm")
    else:
        miss("spacy not installed")

    # ── 5. NLTK data ──────────────────────────────────────────
    # find_spec for the package; data.find() for the corpus
    print("\n  [5] NLTK")
    if pkg("nltk"):
        import nltk
        try:
            nltk.data.find("corpora/stopwords")
            ok("NLTK stopwords corpus")
        except LookupError:
            warn("NLTK stopwords missing — will auto-download on first use")
    else:
        miss("nltk not installed")

    # ── 6. ABSA stack ─────────────────────────────────────────
    print("\n  [6] ABSA Stack (BERTopic + UMAP + HDBSCAN + SentenceTransformers)")
    for name, label in [("bertopic", "bertopic"), ("umap", "umap-learn"),
                        ("hdbscan", "hdbscan")]:
        if pkg(name):
            ok(label)
        else:
            miss(f"{label} not installed")

    if pkg("sentence_transformers"):
        ok("sentence_transformers")
        # StaticEmbedding is a lightweight attribute check — import is already
        # cached if sentence_transformers was found above
        try:
            from sentence_transformers.models import StaticEmbedding  # noqa: F401
            ok("sentence_transformers.models.StaticEmbedding")
        except ImportError:
            warn("StaticEmbedding not found — upgrade: pip install 'sentence-transformers>=3.3.0,<4.0.0'")
    else:
        miss("sentence_transformers not installed")

    # ── 7. Visualization ──────────────────────────────────────
    print("\n  [7] Visualization")
    for name, label in [("wordcloud", "wordcloud"), ("plotly", "plotly"),
                        ("matplotlib", "matplotlib")]:
        if pkg(name):
            ok(label)
        else:
            miss(f"{label} not installed")

    # ── 8. LLM stack ──────────────────────────────────────────
    print("\n  [8] LLM Stack (Groq / LangChain)")
    for name, label in [("langchain_groq", "langchain_groq"),
                        ("groq", "groq")]:
        if pkg(name):
            ok(label)
        else:
            miss(f"{label} not installed")

    # ── 9. Environment variables ──────────────────────────────
    print("\n  [9] Environment Variables")
    for key, critical in [("GROQ_API_KEY", True), ("TAVILY_API_KEY", False),
                           ("SERPER_API_KEY", False), ("GOOGLE_API_KEY", False)]:
        if os.environ.get(key):
            ok(key)
        elif critical:
            miss(f"{key} not set — Advisor Agent will FAIL")
        else:
            warn(f"{key} not set (optional)")

    # ── 10. Data directories ──────────────────────────────────
    print("\n  [10] Data Directories")
    for d in ["data/filtered_data", "data/analysis", "data/raw_data",
              "data/processed_data"]:
        os.makedirs(d, exist_ok=True)
        ok(d)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * W)
    if issues:
        print(f"  [FAIL] {len(issues)} critical issue(s) — fix before running jobs:")
        for i in issues:
            print(f"         - {i}")
    else:
        print(f"  [OK]  All critical checks passed — backend ready.")
    if warnings_:
        print(f"  [WARN] {len(warnings_)} warning(s) — non-fatal but may affect features.")
    print("=" * W + "\n")

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
            state_path = "shared_state.json"
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
            state_path = "shared_state.json"
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


@app.post("/advisor/question/{job_id}")
async def advisor_question(job_id: str, request: AdvisorQuestionRequest):
    """Answer a follow-up question using the AdvisorAgent running on the backend."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before asking questions")
    try:
        from agents.advisor_agent import AdvisorAgent
        advisor = AdvisorAgent(job_id=job_id)
        answer = advisor.answer_question(request.question)
        return {"answer": answer}
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