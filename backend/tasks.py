"""Background tasks for scraping - runs in ISOLATED threads."""

import os
import threading
from datetime import datetime
from typing import List
import json

from agents.base_agent import BaseAgent
from agents.scraper_agent import ScraperAgent
from agents.processor_agent import ProcessorAgent
from agents.analyst_agent import AnalystAgent
from agents.advisor_agent import AdvisorAgent
from agents.visualization_agent import VisualizationAgent
from backend.database import db


def process_scraping_job(job_id: str, topic: str, subreddits: List[str], mode: str = "single"):
    """
    Process a scraping job in a background thread.
    
    Flow:
    1. ScraperAgent: Scrapes Reddit data and saves to data/filtered_data
    2. ProcessorAgent: Reads from data/filtered_data and processes
    3. AnalystAgent: Analyzes sentiment
    4. AdvisorAgent: Generates recommendations
    5. VisualizationAgent: Prepares visualization data
    
    Args:
        job_id: Unique job identifier
        topic: Topic to analyze
        subreddits: List of subreddits to scrape
    """
    
    try:
        # Update job status
        db.update_job(job_id, status="running", started_at=datetime.now().isoformat())

        # Register this job's namespace in shared_state.json before any agent runs.
        # Sets the mode field so agents know whether this is a single or compare run.
        BaseAgent.register_job(job_id, topic, mode)

        print(f"\n{'='*60}")
        print(f"[START] Starting job {job_id} (mode={mode})")
        print(f"{'='*60}")

        # Step 1: Scrape data
        print("\n[Step 1] Scraping subreddits...")
        db.update_job(job_id, progress=20)
        
        try:
            scraper = ScraperAgent(job_id=job_id)
            scraper_result = scraper.run(f"Collect data about {topic}")
            
            if scraper_result.get("status") != "success":
                raise Exception(f"Scraping failed: {scraper_result.get('error')}")

            summary = scraper_result.get("summary", {})
            if summary.get("comments", 0) == 0:
                timed_out_flag = scraper_result.get("data", [{}])[0].get("timed_out", False) if scraper_result.get("data") else False
                if timed_out_flag:
                    raise Exception(
                        "Scraping timed out after 25 minutes with no data collected. "
                        "The topic may be too niche for these subreddits, or the API is slow. "
                        "Please try again or use a broader topic."
                    )
                raise Exception(
                    "No comments were collected. "
                    "The topic may not have enough discussion in the selected subreddits. "
                    "Please try a different or broader topic."
                )
            
            print("[OK] Scraping complete")
            print(f"   - Posts: {scraper_result.get('summary', {}).get('posts', 0)}")
            print(f"   - Comments: {scraper_result.get('summary', {}).get('comments', 0)}")
            db.update_job(job_id, progress=30)
        
        except Exception as e:
            print(f"[ERROR] Scraping error: {str(e)[:100]}")
            raise

        # Step 2: Process data
        print("\n[Step 2] Processing data...")
        db.update_job(job_id, progress=40)
        
        try:
            processor = ProcessorAgent(job_id=job_id)
            processor_result = processor.run()
            
            if processor_result.get("status") not in ["success", "warning"]:
                raise Exception(f"Processing failed: {processor_result.get('error')}")
            
            print("[OK] Processing complete")
            summary = processor_result.get('summary', {})
            print(f"   - Posts processed: {summary.get('posts', 0)}")
            print(f"   - Comments processed: {summary.get('comments', 0)}")
            print(f"   - Words for wordcloud: {summary.get('wordcloud_words', 0)}")
            print(f"   - Texts for sentiment: {summary.get('sentiment_texts', 0)}")
            db.update_job(job_id, progress=50)
        
        except Exception as e:
            print(f"[ERROR] Processing error: {str(e)[:100]}")
            raise

        # Step 3: Analyze sentiment
        print("\n[Step 3] Analyzing sentiment...")
        db.update_job(job_id, progress=65)
        
        try:
            analyst = AnalystAgent(job_id=job_id)
            analyst_result = analyst.run()
            
            if analyst_result.get("status") != "success":
                raise Exception(f"Analysis failed: {analyst_result.get('error')}")
            
            print("[OK] Analysis complete")
            analysis = analyst_result.get('analysis', {})
            print(f"   - Overall sentiment: {analysis.get('overall_sentiment', 'N/A').upper()}")
            print(f"   - Confidence: {analysis.get('confidence', 0):.2%}")
            distribution = analysis.get('sentiment_distribution', {})
            if distribution:
                pos = distribution.get('positive', {}).get('percentage', 0)
                neu = distribution.get('neutral', {}).get('percentage', 0)
                neg = distribution.get('negative', {}).get('percentage', 0)
                print(f"   - Distribution: Positive {pos:.1f}% | Neutral {neu:.1f}% | Negative {neg:.1f}%")
            db.update_job(job_id, progress=75)
        
        except Exception as e:
            print(f"[ERROR] Analysis error: {str(e)[:100]}")
            raise

        # Step 4: Generate recommendations
        print("\n[Step 4] Generating recommendations...")
        db.update_job(job_id, progress=85)
        
        try:
            advisor = AdvisorAgent(job_id=job_id)
            advisor_result = advisor.run()
            
            if advisor_result.get("status") != "success":
                raise Exception(f"Advisor failed: {advisor_result.get('error')}")
            
            recommendations = advisor_result.get('recommendations', [])
            print("[OK] Recommendations generated")
            print(f"   - Generated {len(recommendations)} recommendation(s)")
            db.update_job(job_id, progress=90)
        
        except Exception as e:
            print(f"[ERROR] Advisor error: {str(e)[:100]}")
            raise

        # Step 5: Prepare visualizations
        print("\n[Step 5] Preparing visualizations...")
        db.update_job(job_id, progress=95)
        
        try:
            visualization = VisualizationAgent(job_id=job_id)
            visualization_result = visualization.run()
            
            if visualization_result.get("status") != "success":
                raise Exception(f"Visualization failed: {visualization_result.get('error')}")
            
            print("[OK] Visualizations prepared")
            db.update_job(job_id, progress=98)

        except Exception as e:
            print(f"[ERROR] Visualization error: {str(e)[:100]}")
            raise

        # Step 6: Load final state
        print("\n[Step 6] Loading final state...")
        db.update_job(job_id, progress=99)

        try:
            if os.path.exists("shared_state.json"):
                with open("shared_state.json", 'r') as f:
                    raw = json.load(f)
                final_state = raw.get("jobs", {}).get(job_id, {})
            else:
                final_state = {}
        except Exception as e:
            print(f"[WARN] Could not load final state: {str(e)[:100]}")
            final_state = {}

        # Mark job as complete
        db.update_job(
            job_id,
            status="completed",
            progress=100,
            results={
                "scraper": scraper_result,
                "processor": processor_result,
                "analyst": analyst_result,
                "advisor": advisor_result,
                "visualization": visualization_result,
                "state": final_state
            },
            completed_at=datetime.now().isoformat()
        )

        print(f"\n{'='*60}")
        print(f"[DONE] Job {job_id} completed successfully!")
        print(f"{'='*60}\n")

    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*60}")
        print(f"[FAIL] Job {job_id} failed: {error_msg}")
        print(f"{'='*60}\n")

        db.update_job(
            job_id,
            status="error",
            error=error_msg,
            completed_at=datetime.now().isoformat()
        )


def start_job_async(job_id: str, topic: str, subreddits: List[str], mode: str = "single"):
    """
    Start a scraping job in a completely isolated background thread.

    Args:
        job_id: Unique job identifier
        topic: Topic to analyze
        subreddits: List of subreddits to scrape
        mode: "single" or "compare" — stored in shared_state so agents know the context
    """

    # Create daemon thread
    thread = threading.Thread(
        target=process_scraping_job,
        args=(job_id, topic, subreddits, mode),
        daemon=True,
        name=f"scraper-{job_id}"
    )

    # Start the thread
    thread.start()

    print(f"[OK] Background job {job_id} started in isolated thread (mode={mode})")