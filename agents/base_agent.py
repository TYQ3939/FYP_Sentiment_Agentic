import json
import os
from datetime import datetime
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Agent class that will be inherited by the Worker Agents
class BaseAgent:
    def __init__(self, name, persona):
        self.name = name
        self.persona = persona
        self.state_file = "shared_state.json"
        self.llm = self._initialize_llm()

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
            
            # Initialize Groq LLM with current available model
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=groq_api_key
            )
            
            self.log("✓ Groq AI (Llama-3.3-70b-versatile) initialized successfully")
            return llm
        
        except Exception as e:
            self.log(f"❌ Failed to initialize Groq: {str(e)[:150]}...")
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

    def save_state(self, key, value):
        """Writes a specific piece of data to the shared JSON file."""
        state = self.load_state()
        state[key] = value
        state["metadata"]["last_updated"] = datetime.now().isoformat()
        state["metadata"]["current_agent"] = self.name
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)

    def load_state(self):
        """Reads the entire whiteboard. If file is missing, returns a default template."""
        if not os.path.exists(self.state_file):
            return self._get_default_state()
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return self._get_default_state()

    def _get_default_state(self):
        """The initial structure of your Shared State."""
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