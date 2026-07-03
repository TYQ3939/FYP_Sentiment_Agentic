import json
import os
import threading
from datetime import datetime
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Module-level lock so all agents in all threads share a single write gate.
# This prevents two parallel compare-mode jobs from corrupting shared_state.json
# when they both try to read-modify-write at the same moment.
_state_lock = threading.Lock()

# Absolute path so agents work regardless of the process cwd
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_FILE = os.path.join(_PROJECT_ROOT, "shared_state.json")


# Base Agent class that will be inherited by the Worker Agents
class BaseAgent:
    def __init__(self, name, persona, job_id=None):
        self.name = name
        self.persona = persona
        self.job_id = job_id          # None → legacy single-job mode
        self.state_file = _STATE_FILE
        self.llm = self._initialize_llm()

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _initialize_llm(self):
        """Initialize LLM with Groq (most reliable option)."""

        try:
            self.log("Attempting to initialize Groq AI (Llama-3.3-70b)...")

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY not found in environment variables.\n"
                    "Please add it to your .env file:\n"
                    "GROQ_API_KEY=your_key_here\n"
                    "Get one free at: https://console.groq.com/keys"
                )

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=groq_api_key
            )

            self.log("✓ Groq AI (Llama-3.3-70b-versatile) initialized successfully")
            return llm

        except Exception as e:
            self.log(f"[ERROR] Failed to initialize Groq: {str(e)[:150]}...")
            raise Exception(
                f"LLM initialization failed.\n\n"
                f"Error: {str(e)}\n\n"
                f"Solution: Add GROQ_API_KEY to your .env file\n"
                f"Get a free key at: https://console.groq.com/keys\n\n"
                f"Available models:\n"
                f"  - llama-3.3-70b-versatile (Recommended)\n"
                f"  - llama-3.1-70b-versatile\n"
                f"  - gemma-7b-it"
            )

    def log(self, message):
        """Prints a timestamped log to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.name}]: {message}")

    # ── State helpers ─────────────────────────────────────────────────────────

    def _load_raw_state(self) -> dict:
        """Read the full shared_state.json without any lock (caller must hold it if needed)."""
        if not os.path.exists(self.state_file):
            return {"mode": "single", "jobs": {}}
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"mode": "single", "jobs": {}}

    def _save_raw_state(self, raw: dict):
        """Write the full dict to shared_state.json (caller must hold the lock)."""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(raw, f, indent=4)

    # ── Public state API ──────────────────────────────────────────────────────

    def save_state(self, key, value):
        """
        Write one key into this job's namespace inside shared_state.json.

        Layout after write:
            {
              "mode": "single" | "compare",
              "jobs": {
                "<job_id>": {
                  "metadata": {...},
                  "filtered_data_path": "...",
                  "processed_data": [...],
                  ...
                }
              }
            }
        """
        with _state_lock:
            raw = self._load_raw_state()

            if self.job_id:
                # Ensure this job's slot exists
                raw.setdefault("jobs", {}).setdefault(
                    self.job_id, self._get_default_state()
                )
                raw["jobs"][self.job_id][key] = value
                raw["jobs"][self.job_id]["metadata"]["last_updated"] = (
                    datetime.now().isoformat()
                )
                raw["jobs"][self.job_id]["metadata"]["current_agent"] = self.name
            else:
                # Legacy / manual runs: write at top level
                raw[key] = value
                raw.setdefault("metadata", {})["last_updated"] = (
                    datetime.now().isoformat()
                )
                raw.setdefault("metadata", {})["current_agent"] = self.name

            self._save_raw_state(raw)

    def load_state(self) -> dict:
        """
        Return this job's state slice.
        In compare mode each job only sees its own data, preventing cross-contamination.
        """
        raw = self._load_raw_state()
        if self.job_id:
            return raw.get("jobs", {}).get(self.job_id, self._get_default_state())
        # Legacy: return entire file
        return raw

    def get_mode(self) -> str:
        """Return 'single' or 'compare' — agents can branch on this if needed."""
        return self._load_raw_state().get("mode", "single")

    # ── Job registration (called by tasks.py before agents run) ──────────────

    @classmethod
    def register_job(cls, job_id: str, topic: str, mode: str = "single"):
        """
        Initialize a job's namespace in shared_state.json before the pipeline starts.
        Sets the top-level mode field so all agents know the run context.
        Must be called once per job, before ScraperAgent runs.
        """
        with _state_lock:
            try:
                with open(_STATE_FILE, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
            except Exception:
                raw = {"mode": mode, "jobs": {}}

            raw["mode"] = mode
            raw.setdefault("jobs", {})[job_id] = {
                "metadata": {
                    "topic": topic,
                    "status": "initialized",
                    "last_updated": datetime.now().isoformat(),
                    "current_agent": ""
                },
                "raw_data": [],
                "processed_data": [],
                "sentiment_results": {},
                "recommendations": [],
                "visuals_ready": False
            }

            with open(_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(raw, f, indent=4)

    # ── Default structure ─────────────────────────────────────────────────────

    def _get_default_state(self) -> dict:
        """Empty per-job state template."""
        return {
            "metadata": {
                "topic": "",
                "status": "initialized",
                "last_updated": "",
                "current_agent": ""
            },
            "raw_data": [],
            "processed_data": [],
            "sentiment_results": {},
            "recommendations": [],
            "visuals_ready": False
        }
