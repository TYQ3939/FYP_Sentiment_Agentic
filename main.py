# Windows asyncio fix MUST be first
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Now import everything else
import json
import os
from dotenv import load_dotenv
from agents.scraper_agent import ScraperAgent
from agents.processor_agent import ProcessorAgent
from agents.analyst_agent import AnalystAgent
from agents.advisor_agent import AdvisorAgent
from agents.visualization_agent import VisualizationAgent

# Load environment variables from .env file
load_dotenv()

# Verify API keys are loaded
def verify_api_keys():
    """Verify that all required API keys are available."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("⚠️ Warning: GROQ_API_KEY not found in .env file")
        return False
    return True

# Function to initialize the shared state 
def initialize_shared_state(topic):
    """Resets the JSON whiteboard for a new run."""
    initial_state = {
        "metadata": {
            "topic": topic, 
            "status": "started",
            "subreddits_scraped": [],
            "total_posts": 0,
            "total_comments": 0
        },
        "raw_data": [], 
        "processed_data": [], 
        "sentiment_results": {}, 
        "recommendations": []
    }

    # Write the initial state to the JSON file
    with open("shared_state.json", "w") as f:
        json.dump(initial_state, f, indent=4)

def run_scraper(topic):
    """Run the scraper agent."""
    if not verify_api_keys():
        raise Exception("API keys not configured. Check your .env file.")
    
    agent = ScraperAgent()
    return agent.run(f"Collect data about {topic} from all available sources.")

def run_processor():
    """Run the processor agent."""
    agent = ProcessorAgent()
    return agent.run()

def run_analyst():
    """Run the analyst agent."""
    agent = AnalystAgent()
    return agent.run()

def run_advisor():
    """Run the advisor agent."""
    agent = AdvisorAgent()
    return agent.run()

def run_visualizer():
    """Run the visualizer agent."""
    agent = VisualizationAgent()
    return agent.run()
