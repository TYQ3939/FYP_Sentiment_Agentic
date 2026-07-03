"""Job database — uses MongoDB Atlas if MONGODB_URI is set, falls back to local JSON file."""

import json
import os
from datetime import datetime
from typing import Dict, Optional
import threading

MONGODB_URI = os.getenv("MONGODB_URI", "")

# ── MongoDB backend ────────────────────────────────────────────────────────────
class MongoJobDatabase:
    def __init__(self, uri: str):
        from pymongo import MongoClient
        self.client = MongoClient(uri)
        self.col = self.client["fyp_sentiment"]["jobs"]
        print("[DB] Connected to MongoDB Atlas")

    def create_job(self, job_id: str, topic: str, subreddits: list) -> dict:
        job = {
            "_id": job_id,
            "id": job_id,
            "topic": topic,
            "subreddits": subreddits,
            "status": "pending",
            "progress": 0,
            "results": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
        }
        self.col.insert_one(job)
        return job

    def get_job(self, job_id: str) -> Optional[dict]:
        doc = self.col.find_one({"_id": job_id})
        if doc:
            doc.pop("_id", None)
        return doc

    def update_job(self, job_id: str, **kwargs) -> bool:
        result = self.col.update_one({"_id": job_id}, {"$set": kwargs})
        return result.matched_count > 0

    def list_jobs(self) -> list:
        return [{k: v for k, v in doc.items() if k != "_id"}
                for doc in self.col.find()]

    def save_to_file(self):
        pass  # MongoDB handles persistence

    def jobs(self):
        return {}


# ── Local JSON fallback ────────────────────────────────────────────────────────
class LocalJobDatabase:
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_file = os.path.join(_root, "jobs_db.json")
        self._load()
        print("[DB] Using local JSON file (set MONGODB_URI to use MongoDB Atlas)")

    def create_job(self, job_id: str, topic: str, subreddits: list) -> dict:
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
                "completed_at": None,
            }
            self._jobs[job_id] = job
            self._save()
            return job

    def get_job(self, job_id: str) -> Optional[dict]:
        with self.lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> bool:
        with self.lock:
            if job_id not in self._jobs:
                return False
            self._jobs[job_id].update(kwargs)
            self._save()
            return True

    def list_jobs(self) -> list:
        with self.lock:
            return list(self._jobs.values())

    def _save(self):
        try:
            with open(self.db_file, "w") as f:
                json.dump(self._jobs, f, indent=2, default=str)
        except Exception as e:
            print(f"[WARN] Failed to save jobs: {e}")

    def _load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    self._jobs = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load jobs: {e}")

    # Keep attribute name compatible with old code that accesses db.jobs directly
    @property
    def jobs(self):
        return self._jobs

    def save_to_file(self):
        self._save()


# ── Auto-select backend ────────────────────────────────────────────────────────
def _create_db():
    if MONGODB_URI:
        try:
            return MongoJobDatabase(MONGODB_URI)
        except Exception as e:
            print(f"[WARN] MongoDB connection failed ({e}), falling back to local JSON")
    return LocalJobDatabase()

db = _create_db()
