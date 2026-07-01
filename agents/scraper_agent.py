import os
import json
from agents.base_agent import BaseAgent
from tools.scraper_tools import (
    infer_topic_structure,
    scrape_with_api,
    scrape_with_youtube,
    find_local_dataset,
    versioned_path,
    export_to_csv,
)


class ScraperAgent(BaseAgent):
    """
    Collects Reddit comments via the Arctic Shift live API.

    Fallback chain when Arctic Shift returns 0 comments:
      1. YouTube Data API v3  (requires YOUTUBE_API_KEY in .env)
      2. Local cached dataset (data/filtered_data/combined_<topic>_filtered*.json)
    """

    def __init__(self):
        super().__init__(
            "ScraperAgent",
            "You are a data scraper agent that collects social media data.",
        )

    def run(self, user_request: str) -> dict:
        self.log(f"Processing request: {user_request}")

        try:
            topic = self._extract_topic(user_request)
            self.log(f"Extracted topic: {topic}")

            # Step 1 — LLM router: subreddits + keywords
            self.log("Step 1: Structuring topic with LLM...")
            structure  = infer_topic_structure(topic, self.llm)
            subreddits = structure["subreddits"]
            keywords   = structure["keywords"]
            self.log(f"Subreddits : {subreddits}")
            self.log(f"Keywords   : {keywords}")

            # Steps 2-5 — Primary: Arctic Shift (Reddit)
            self.log("Step 2-5: Collecting data via Arctic Shift API...")
            result = scrape_with_api(subreddits, keywords)
            data_source = "reddit"

            # Fallback 1: YouTube Data API
            if result.get("comments_count", 0) == 0:
                self.log("Arctic Shift returned 0 comments — trying YouTube API fallback...")
                try:
                    result = scrape_with_youtube(topic, keywords)
                    data_source = "youtube"
                    self.log(f"YouTube: collected {result['comments_count']} comments")
                except EnvironmentError as env_err:
                    self.log(str(env_err))
                except Exception as yt_err:
                    self.log(f"YouTube fallback failed: {str(yt_err)[:120]}")

            # Fallback 2: Local cached dataset
            if result.get("comments_count", 0) == 0:
                self.log("All live sources returned 0 comments — checking local cache...")
                local_path = find_local_dataset(topic)
                if local_path:
                    self.log(f"Found local dataset: {local_path}")
                    try:
                        with open(local_path, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        comments = cached.get("comments", [])
                        self.log(f"Loaded {len(comments)} comments from local cache")

                        self.save_state("raw_data", [cached])
                        self.save_state("filtered_data_path", local_path)
                        self.save_state("metadata", {
                            "topic"             : topic,
                            "subreddits_scraped": subreddits,
                            "keywords_used"     : keywords,
                            "total_posts"       : 0,
                            "total_comments"    : len(comments),
                            "data_source"       : "local_cache",
                        })

                        return {
                            "status": "success",
                            "data"  : [cached],
                            "summary": {
                                "topic"   : topic,
                                "posts"   : 0,
                                "comments": len(comments),
                                "source"  : f"local cache ({os.path.basename(local_path)})",
                            },
                        }
                    except Exception as cache_err:
                        self.log(f"Local cache load failed: {str(cache_err)[:100]}")
                else:
                    self.log("No local dataset found for this topic")

            # Step 6 — Export (versioned to avoid overwriting prior runs)
            csv_dir = "./data/filtered_data"
            os.makedirs(csv_dir, exist_ok=True)
            safe     = topic.replace(" ", "_").replace("/", "_")[:60]
            csv_path = versioned_path(csv_dir, f"{safe}_comments", ".csv")
            export_to_csv(result, csv_path)
            self.log(f"CSV exported: {csv_path}")

            # Persist for downstream agents
            self.log("Saving raw data to shared state...")
            self.save_state("raw_data", [result])

            self.log("Saving filtered data file for ProcessorAgent...")
            self._save_filtered_data(result, topic)

            self.save_state("metadata", {
                "topic"             : topic,
                "subreddits_scraped": subreddits if data_source == "reddit" else ["YouTube"],
                "keywords_used"     : keywords,
                "total_posts"       : 0,
                "total_comments"    : result.get("comments_count", 0),
                "data_source"       : data_source,
            })

            self.log(f"Scraping complete — {result.get('comments_count', 0)} comments ({data_source})")

            return {
                "status": "success",
                "data"  : [result],
                "summary": {
                    "topic"   : topic,
                    "posts"   : 0,
                    "comments": result.get("comments_count", 0),
                    "source"  : data_source,
                },
            }

        except Exception as exc:
            self.log(f"Error during scraping: {str(exc)}")
            return {"status": "error", "error": str(exc)}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _extract_topic(self, user_request: str) -> str:
        if "about" in user_request.lower():
            return user_request.lower().split("about", 1)[1].strip().rstrip(".")
        return user_request

    def _save_filtered_data(self, result: dict, topic: str) -> None:
        """
        Write the scrape result to data/filtered_data/ so ProcessorAgent can find it.
        Uses versioned filenames so same-topic runs never overwrite each other.
        """
        filtered_data_dir = "./data/filtered_data"
        os.makedirs(filtered_data_dir, exist_ok=True)

        safe     = topic.replace(" ", "_").replace("/", "_")[:60]
        filepath = versioned_path(filtered_data_dir, f"combined_{safe}_filtered", ".json")

        payload = {
            "subreddit": result.get("subreddit", "combined"),
            "topic"    : topic,
            "posts"    : result.get("posts", []),
            "comments" : result.get("comments", []),
            "metadata" : {
                "posts_count"   : result.get("posts_count",    0),
                "comments_count": result.get("comments_count", 0),
                "scraped_at"    : result.get("scraped_at",     ""),
                "target_reached": result.get("target_reached", False),
                "source"        : result.get("source", "reddit"),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        self.save_state("filtered_data_path", filepath)
        self.log(f"Saved filtered data: {os.path.basename(filepath)}")
