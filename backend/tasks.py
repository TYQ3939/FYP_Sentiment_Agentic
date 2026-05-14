"""Background tasks for scraping - runs in ISOLATED threads."""

import os
import threading
from datetime import datetime
from typing import List
import json

from agents.scraper_agent import ScraperAgent
from agents.processor_agent import ProcessorAgent
from agents.analyst_agent import AnalystAgent
from agents.advisor_agent import AdvisorAgent
from agents.visualization_agent import VisualizationAgent
from backend.database import db


def process_scraping_job(job_id: str, topic: str, subreddits: List[str]):
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
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting job {job_id}")
        print(f"{'='*60}")
        
        # Step 1: Scrape data
        print("\n📊 Step 1: Scraping subreddits...")
        db.update_job(job_id, progress=20)
        
        try:
            scraper = ScraperAgent()
            subreddits_str = ", ".join(subreddits)
            scraper_result = scraper.run(f"Collect data about {topic} from subreddits: {subreddits_str}")
            
            if scraper_result.get("status") != "success":
                raise Exception(f"Scraping failed: {scraper_result.get('error')}")
            
            print("✅ Scraping complete")
            print(f"   - Posts: {scraper_result.get('summary', {}).get('posts', 0)}")
            print(f"   - Comments: {scraper_result.get('summary', {}).get('comments', 0)}")
            db.update_job(job_id, progress=30)
        
        except Exception as e:
            print(f"❌ Scraping error: {str(e)[:100]}")
            raise
        
        # Step 2: Process data
        print("\n🔄 Step 2: Processing data...")
        db.update_job(job_id, progress=40)
        
        try:
            processor = ProcessorAgent()
            processor_result = processor.run()
            
            if processor_result.get("status") not in ["success", "warning"]:
                raise Exception(f"Processing failed: {processor_result.get('error')}")
            
            print("✅ Processing complete")
            summary = processor_result.get('summary', {})
            print(f"   - Posts processed: {summary.get('posts', 0)}")
            print(f"   - Comments processed: {summary.get('comments', 0)}")
            print(f"   - Words for wordcloud: {summary.get('wordcloud_words', 0)}")
            print(f"   - Texts for sentiment: {summary.get('sentiment_texts', 0)}")
            db.update_job(job_id, progress=50)
        
        except Exception as e:
            print(f"❌ Processing error: {str(e)[:100]}")
            raise
        
        # Step 3: Analyze sentiment
        print("\n🧠 Step 3: Analyzing sentiment...")
        db.update_job(job_id, progress=65)
        
        try:
            analyst = AnalystAgent()
            analyst_result = analyst.run()
            
            if analyst_result.get("status") != "success":
                raise Exception(f"Analysis failed: {analyst_result.get('error')}")
            
            print("✅ Analysis complete")
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
            print(f"❌ Analysis error: {str(e)[:100]}")
            raise
        
        # Step 4: Generate recommendations
        print("\n💡 Step 4: Generating recommendations...")
        db.update_job(job_id, progress=85)
        
        try:
            advisor = AdvisorAgent()
            advisor_result = advisor.run()
            
            if advisor_result.get("status") != "success":
                raise Exception(f"Advisor failed: {advisor_result.get('error')}")
            
            recommendations = advisor_result.get('recommendations', [])
            print("✅ Recommendations generated")
            print(f"   - Generated {len(recommendations)} recommendation(s)")
            db.update_job(job_id, progress=90)
        
        except Exception as e:
            print(f"❌ Advisor error: {str(e)[:100]}")
            raise
        
        # Step 5: Prepare visualizations
        print("\n📈 Step 5: Preparing visualizations...")
        db.update_job(job_id, progress=95)
        
        try:
            visualization = VisualizationAgent()
            visualization_result = visualization.run()
            
            if visualization_result.get("status") != "success":
                raise Exception(f"Visualization failed: {visualization_result.get('error')}")
            
            print("✅ Visualizations prepared")
            db.update_job(job_id, progress=98)
        
        except Exception as e:
            print(f"❌ Visualization error: {str(e)[:100]}")
            raise
        
        # Step 6: Load final state
        print("\n📁 Step 6: Loading final state...")
        db.update_job(job_id, progress=99)
        
        try:
            if os.path.exists("shared_state.json"):
                with open("shared_state.json", 'r') as f:
                    final_state = json.load(f)
            else:
                final_state = {}
        except Exception as e:
            print(f"⚠️ Could not load final state: {str(e)[:100]}")
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
        print(f"✅ Job {job_id} completed successfully!")
        print(f"{'='*60}\n")
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*60}")
        print(f"❌ Job {job_id} failed: {error_msg}")
        print(f"{'='*60}\n")
        
        db.update_job(
            job_id,
            status="error",
            error=error_msg,
            completed_at=datetime.now().isoformat()
        )


def start_job_async(job_id: str, topic: str, subreddits: List[str]):
    """
    Start a scraping job in a completely isolated background thread.
    
    Args:
        job_id: Unique job identifier
        topic: Topic to analyze
        subreddits: List of subreddits to scrape
    """
    
    # Create daemon thread
    thread = threading.Thread(
        target=process_scraping_job,
        args=(job_id, topic, subreddits),
        daemon=True,
        name=f"scraper-{job_id}"
    )
    
    # Start the thread
    thread.start()
    
    print(f"✅ Background job {job_id} started in isolated thread")