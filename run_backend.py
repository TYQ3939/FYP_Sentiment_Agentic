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
    host = "0.0.0.0"
    port = 8000
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print(f"{'='*60}\n")

    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )