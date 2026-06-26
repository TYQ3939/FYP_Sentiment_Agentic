import os
import json
from agents.base_agent import BaseAgent
from tools.scraper_tools import infer_topic_structure, scrape_with_api, export_to_csv


class ScraperAgent(BaseAgent):
    """Collects Reddit comments via the Arctic Shift live API."""

    def __init__(self):
        super().__init__(
            "ScraperAgent",
            "You are a data scraper agent that collects social media data from Reddit.",
        )

    def run(self, user_request: str) -> dict:
        """
        End-to-end scraping run.

        Args:
            user_request: e.g. "Collect data about iPhone 17 Pro Max"

        Returns:
            {"status": "success"|"error", "data": [...], "summary": {...}}
        """
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

            # Steps 2-5 — API collection loop
            self.log("Step 2-5: Collecting data via Arctic Shift API...")
            result = scrape_with_api(subreddits, keywords)

            # Step 6 — Export CSV
            csv_dir  = "./data/filtered_data"
            os.makedirs(csv_dir, exist_ok=True)
            safe     = topic.replace(" ", "_").replace("/", "_")[:60]
            csv_path = os.path.join(csv_dir, f"{safe}_comments.csv")
            export_to_csv(result, csv_path)
            self.log(f"CSV exported: {csv_path}")

            # Persist for downstream agents
            self.log("Saving raw data to shared state...")
            self.save_state("raw_data", [result])

            self.log("Saving filtered data file for ProcessorAgent...")
            self._save_filtered_data(result, topic)

            self.save_state("metadata", {
                "topic"             : topic,
                "subreddits_scraped": subreddits,
                "keywords_used"     : keywords,
                "total_posts"       : 0,
                "total_comments"    : result.get("comments_count", 0),
            })

            self.log(f"Scraping complete — {result.get('comments_count', 0)} comments")

            return {
                "status": "success",
                "data"  : [result],
                "summary": {
                    "topic"   : topic,
                    "posts"   : 0,
                    "comments": result.get("comments_count", 0),
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
        Write the scrape result to data/filtered_data/ so ProcessorAgent can
        find it via its glob scan for *_filtered.json.
        """
        filtered_data_dir = "./data/filtered_data"
        os.makedirs(filtered_data_dir, exist_ok=True)

        safe     = topic.replace(" ", "_").replace("/", "_")[:60]
        filename = f"combined_{safe}_filtered.json"
        filepath = os.path.join(filtered_data_dir, filename)

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
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # Tell ProcessorAgent exactly which file belongs to this job
        self.save_state("filtered_data_path", filepath)
        self.log(f"Saved filtered data: {filename}")
