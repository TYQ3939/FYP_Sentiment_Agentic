from agents.base_agent import BaseAgent
import json
import os
import glob

class ProcessorAgent(BaseAgent):
    """Processor Agent that cleans and processes raw data."""
    
    def __init__(self):
        super().__init__(
            "ProcessorAgent",
            "You are a data processor agent that cleans and prepares data for analysis."
        )

    def run(self) -> dict:
        """Process filtered data from data/filtered_data directory."""
        
        self.log("Starting data processing...")
        
        try:
            # Load metadata to get topic
            state = self.load_state()
            metadata = state.get("metadata", {})
            topic = metadata.get("topic", "")
            
            # Step 1: Find the filtered data file for this job
            self.log("Step 1: Loading filtered data file from data/filtered_data...")

            filtered_data_dir = "./data/filtered_data"

            if not os.path.exists(filtered_data_dir):
                self.log(f"⚠️ Filtered data directory not found: {filtered_data_dir}")
                return {"status": "warning", "message": "No filtered data directory found"}

            # Prefer the exact path the ScraperAgent recorded for this job
            saved_path = state.get("filtered_data_path", "")
            if saved_path and os.path.exists(saved_path):
                json_files = [saved_path]
                self.log(f"Using job-specific file: {os.path.basename(saved_path)}")
            else:
                # Fallback: pick up any filtered file (legacy / manual runs)
                json_files = glob.glob(os.path.join(filtered_data_dir, "*_filtered.json"))
                self.log(f"No job path in state — found {len(json_files)} file(s) via glob")

            if not json_files:
                self.log(f"⚠️ No filtered data files found in {filtered_data_dir}")
                return {"status": "warning", "message": "No filtered data files found"}
            
            self.log(f"Found {len(json_files)} filtered data file(s)")
            
            # Step 2: Load and process each file
            from tools.processor_tools import (
                remove_duplicates,
                preprocess_wordcloud_with_pos_tagging,
                preprocess_for_sentiment
            )
            
            processed_data = []
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_content = json.load(f)
                    
                    self.log(f"\n📁 Processing {os.path.basename(json_file)}...")
                    
                    # ===== DETECT DATA FORMAT =====
                    # Format 1: Dict with structure {"subreddit": str, "topic": str, "posts": list, "comments": list, "metadata": dict}
                    # Format 2: List of raw records [{...comment fields...}, {...comment fields...}]
                    
                    if isinstance(file_content, dict) and "subreddit" in file_content:
                        # ===== FORMAT 1: STANDARD FROM SCRAPER AGENT =====
                        self.log("   ✅ Format: Standard (from ScraperAgent)")
                        
                        subreddit = file_content.get("subreddit", "unknown")
                        comments = file_content.get("comments", [])
                        posts = file_content.get("posts", [])
                        
                        self.log(f"   Subreddit: r/{subreddit}")
                        self.log(f"   Raw: {len(posts)} posts, {len(comments)} comments")
                    
                    elif isinstance(file_content, list) and len(file_content) > 0:
                        # ===== FORMAT 2: RAW RECORDS FROM EXTERNAL SOURCE =====
                        self.log("   ✅ Format: Raw records (external source)")
                        
                        subreddit = "unknown"
                        comments = []
                        posts = []
                        
                        # CRITICAL: Each record has BOTH post metadata AND comment_body
                        # We ONLY extract comment_body for sentiment analysis
                        # post_title and post_selftext are just context, NOT for processing
                        
                        for record in file_content:
                            if not isinstance(record, dict):
                                continue
                            
                            # Extract COMMENT from comment_body ONLY
                            has_comment_body = "comment_body" in record and record["comment_body"]
                            
                            # Skip if no comment or comment is [removed]
                            if not has_comment_body or record["comment_body"] == "[removed]":
                                continue
                            
                            # Create comment record
                            comment_record = {
                                "text": record.get("comment_body", ""),
                                "score": record.get("score", 0),
                                "created_at": record.get("date_utc", ""),
                                "author": record.get("author", ""),
                                "link_id": record.get("link_id", ""),
                                "post_title": record.get("post_title", "")  # Keep for context
                            }
                            
                            if comment_record["text"]:
                                comments.append(comment_record)
                        
                        self.log(f"   Standardized: {len(posts)} posts, {len(comments)} comments")
                    
                    else:
                        # ===== UNKNOWN FORMAT =====
                        self.log(f"❌ Unknown data format: {type(file_content)}")
                        self.log(f"   Expected: dict with 'subreddit' key OR list of records")
                        continue
                    
                    # Check if we have any data
                    if not comments and not posts:
                        self.log(f"⚠️ No valid data found in file")
                        continue
                    
                    # Step 3: Remove duplicates
                    comments = remove_duplicates(comments)
                    posts = remove_duplicates(posts)
                    
                    self.log(f"   After dedup: {len(posts)} posts, {len(comments)} comments")
                    
                    # Step 4: Extract text for preprocessing
                    comment_texts = [c.get("text", "") for c in comments if c.get("text")]
                    post_texts = [p.get("text", "") for p in posts if p.get("text")]

                    self.log(f"   Text extracted: {len(comment_texts)} comment texts, {len(post_texts)} post texts")

                    if not comment_texts and not post_texts:
                        self.log(f"⚠️ No text content found in records")
                        continue

                    # Step 5: Preprocess for wordcloud and sentiment
                    self.log(f"   Preprocessing...")
                    from tools.processor_tools import preprocess_for_sentiment_bertweet

                    wordcloud_comments = preprocess_wordcloud_with_pos_tagging(comment_texts, topic=topic, llm=self.llm)
                    wordcloud_posts = preprocess_wordcloud_with_pos_tagging(post_texts, topic=topic, llm=self.llm)

                    # Preprocess one-at-a-time to keep created_at aligned with each text.
                    # preprocess_for_sentiment_bertweet can filter out short/empty texts,
                    # so batch processing breaks index alignment — per-item is the only safe way.
                    sentiment_comment_pairs = []
                    for c in comments:
                        raw = c.get("text", "")
                        if not raw:
                            continue
                        result_texts = preprocess_for_sentiment_bertweet([raw])
                        if result_texts:
                            sentiment_comment_pairs.append({
                                "text"      : result_texts[0],
                                "created_at": str(c.get("created_at", "")),
                            })

                    sentiment_post_pairs = []
                    for p in posts:
                        raw = p.get("text", "")
                        if not raw:
                            continue
                        result_texts = preprocess_for_sentiment_bertweet([raw])
                        if result_texts:
                            sentiment_post_pairs.append({
                                "text"      : result_texts[0],
                                "created_at": str(p.get("created_at", "")),
                            })

                    sentiment_comments     = [x["text"] for x in sentiment_comment_pairs]
                    comment_timestamps     = [x["created_at"] for x in sentiment_comment_pairs]
                    sentiment_posts        = [x["text"] for x in sentiment_post_pairs]
                    post_timestamps        = [x["created_at"] for x in sentiment_post_pairs]

                    self.log(f"   ✅ Preprocessing done: {len(wordcloud_comments)} wordcloud, {len(sentiment_comments)} sentiment")

                    # Step 6: Create processed item
                    processed_item = {
                        "subreddit": subreddit,
                        "topic": file_content.get("topic", topic) if isinstance(file_content, dict) else topic,
                        "total_posts": len(posts),
                        "total_comments": len(comments),
                        "posts": posts,
                        "comments": comments,
                        "preprocessing": {
                            "wordcloud": {
                                "comments": wordcloud_comments,
                                "posts": wordcloud_posts,
                                "total_words": len(wordcloud_comments) + len(wordcloud_posts)
                            },
                            "sentiment": {
                                "comments"          : sentiment_comments,
                                "comment_timestamps": comment_timestamps,
                                "posts"             : sentiment_posts,
                                "post_timestamps"   : post_timestamps,
                                "total_texts": len(sentiment_comments) + len(sentiment_posts)
                            }
                        },
                        "metadata": file_content.get("metadata", {}) if isinstance(file_content, dict) else {}
                    }
                    
                    processed_data.append(processed_item)
                    self.log(f"   ✅ Processed successfully")
                
                except json.JSONDecodeError as e:
                    self.log(f"❌ JSON decode error: {str(e)[:100]}")
                    continue
                
                except Exception as e:
                    self.log(f"❌ Error: {str(e)[:100]}")
                    continue
            
            # Step 7: Validate results
            if not processed_data:
                self.log("\n❌ No data could be processed from any file")
                return {"status": "error", "error": "No data could be processed"}
            
            # Step 8: Save to shared state
            self.log("Step 2: Saving processed data to shared state...")
            self.save_state("processed_data", processed_data)
            
            # Calculate statistics
            total_posts = sum(item["total_posts"] for item in processed_data)
            total_comments = sum(item["total_comments"] for item in processed_data)
            total_words_for_wordcloud = sum(
                item["preprocessing"]["wordcloud"]["total_words"] 
                for item in processed_data
            )
            total_texts_for_sentiment = sum(
                item["preprocessing"]["sentiment"]["total_texts"] 
                for item in processed_data
            )
            
            self.log(f"\n✅ Processing complete:")
            self.log(f"   - Files processed: {len(processed_data)}")
            self.log(f"   - Total posts: {total_posts}")
            self.log(f"   - Total comments: {total_comments}")
            self.log(f"   - Wordcloud words: {total_words_for_wordcloud}")
            self.log(f"   - Sentiment texts: {total_texts_for_sentiment}")
            
            return {
                "status": "success",
                "items_processed": len(processed_data),
                "summary": {
                    "posts": total_posts,
                    "comments": total_comments,
                    "wordcloud_words": total_words_for_wordcloud,
                    "sentiment_texts": total_texts_for_sentiment
                }
            }
        
        except Exception as e:
            self.log(f"❌ Critical error: {str(e)}")
            return {"status": "error", "error": str(e)}