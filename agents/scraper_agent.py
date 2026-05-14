# Windows asyncio fix MUST be first
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Now import everything else
import os
import json
from agents.base_agent import BaseAgent
from tools.scraper_tools import infer_subreddits_from_topic, scrape_with_retry

class ScraperAgent(BaseAgent):
    """Scraper Agent that collects Reddit data using modern langchain pattern."""
    
    def __init__(self):
        super().__init__(
            "ScraperAgent", 
            "You are a data scraper agent that collects social media data from Reddit."
        )

    def run(self, user_request: str) -> dict:
        """
        Run the scraper agent.
        
        Args:
            user_request: User's request (e.g., "Collect data about iPhone 17 Pro Max")
            
        Returns:
            Dictionary with scraped data
        """
        self.log(f"Processing request: {user_request}")
        
        try:
            # Extract topic from user request
            topic = self._extract_topic(user_request)
            self.log(f"Extracted topic: {topic}")
            
            # Step 1: Use LLM to infer relevant subreddits
            self.log("Step 1: Inferring relevant subreddits...")
            subreddits = infer_subreddits_from_topic(topic, self.llm)
            self.log(f"Identified subreddits: {subreddits}")
            
            # Step 2: Scrape data from each subreddit with retry logic
            all_raw_data = []
            for subreddit in subreddits:
                self.log(f"Step 2: Scraping r/{subreddit} with retry logic...")
                try:
                    # Call the function that will start the scraping process
                    result = scrape_with_retry(subreddit, topic, min_comments=500)
                    all_raw_data.append(result)
                    
                    self.log(f"✅ r/{subreddit} completed:")
                    self.log(f"   - Posts: {result['posts_count']}")
                    self.log(f"   - Comments: {result['comments_count']}")
                    self.log(f"   - Iterations: {result['iterations_used']}")
                
                except Exception as e:
                    self.log(f"⚠️ Failed to scrape r/{subreddit}: {str(e)[:100]}")
                    continue
            
            # Step 3: Save raw data to shared state
            self.log("Step 3: Saving raw data to shared state...")
            self.save_state("raw_data", all_raw_data)
            
            # Step 4: Save filtered data files to data/filtered_data for processor
            self.log("Step 4: Saving filtered data files for processor agent...")
            self._save_filtered_data(all_raw_data, topic)
            
            # Calculate totals
            total_posts = sum(r.get("posts_count", 0) for r in all_raw_data)
            total_comments = sum(r.get("comments_count", 0) for r in all_raw_data)
            
            self.save_state("metadata", {
                "topic": topic,
                "subreddits_scraped": subreddits,
                "total_posts": total_posts,
                "total_comments": total_comments
            })
            
            self.log(f"✅ Data scraping completed!")
            self.log(f"   Total Posts: {total_posts}")
            self.log(f"   Total Comments: {total_comments}")
            
            return {
                "status": "success",
                "data": all_raw_data,
                "summary": {
                    "topic": topic,
                    "posts": total_posts,
                    "comments": total_comments
                }
            }
        
        except Exception as e:
            self.log(f"❌ Error during scraping: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _extract_topic(self, user_request: str) -> str:
        """Extract the main topic from the user request."""
        if "about" in user_request.lower():
            return user_request.lower().split("about", 1)[1].strip().rstrip(".")
        return user_request

    def _save_filtered_data(self, all_raw_data: list, topic: str) -> None:
        """
        Save filtered data to data/filtered_data directory for processor agent.
        
        Args:
            all_raw_data: List of raw data from all subreddits
            topic: The topic being analyzed
        """
        
        # Create filtered_data directory if it doesn't exist
        filtered_data_dir = "./data/filtered_data"
        os.makedirs(filtered_data_dir, exist_ok=True)
        
        # Save each subreddit's data as a separate JSON file
        for data in all_raw_data:
            subreddit = data.get("subreddit", "unknown")
            
            # Prepare filtered data structure
            filtered_data = {
                "subreddit": subreddit,
                "topic": topic,
                "posts": data.get("posts", []),
                "comments": data.get("comments", []),
                "metadata": {
                    "posts_count": data.get("posts_count", 0),
                    "comments_count": data.get("comments_count", 0),
                    "scraped_at": data.get("scraped_at", ""),
                    "iterations_used": data.get("iterations_used", 0)
                }
            }
            
            # Save as JSON file
            filename = f"{subreddit}_{topic.replace(' ', '_')}_filtered.json"
            filepath = os.path.join(filtered_data_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                
                self.log(f"✅ Saved filtered data: {filename}")
            
            except Exception as e:
                self.log(f"⚠️ Failed to save filtered data for {subreddit}: {str(e)}")