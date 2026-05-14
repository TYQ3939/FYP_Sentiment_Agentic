"""
Helper script to run the FastAPI backend server.

Usage:
    python run_backend.py
    
Or use uvicorn directly:
    uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
"""

import sys
import asyncio
import uvicorn

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 Starting Sentiment Analysis Backend API")
    print(f"{'='*60}")
    print(f"Host: 127.0.0.1")
    print(f"Port: 8000")
    print(f"Docs: http://127.0.0.1:8000/docs")
    print(f"Health Check: http://127.0.0.1:8000/health")
    print(f"{'='*60}\n")
    
    # Use import string format to enable reload
    uvicorn.run(
        "backend.api:app",  # ← Import string, not app object
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )