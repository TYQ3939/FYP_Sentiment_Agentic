"""In-memory job database. Can be replaced with PostgreSQL/MongoDB."""

import json
import os
from datetime import datetime
from typing import Dict, Optional
import threading

# Thread-safe job storage
class JobDatabase:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.db_file = "jobs_db.json"
        self.load_from_file()
    
    def create_job(self, job_id: str, topic: str, subreddits: list) -> dict:
        """Create a new job."""
        with self.lock:
            job = {
                "id": job_id,
                "topic": topic,
                "subreddits": subreddits,
                "status": "pending",
                "progress": 0,
                "results": None,
                "error": None,
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None
            }
            self.jobs[job_id] = job
            self.save_to_file()
            return job
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update job fields."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            self.jobs[job_id].update(kwargs)
            self.save_to_file()
            return True
    
    def list_jobs(self) -> list:
        """List all jobs."""
        with self.lock:
            return list(self.jobs.values())
    
    def save_to_file(self):
        """Persist jobs to file."""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.jobs, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save jobs to file: {e}")
    
    def load_from_file(self):
        """Load jobs from file."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.jobs = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load jobs from file: {e}")

# Global database instance
db = JobDatabase()