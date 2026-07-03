from agents.base_agent import BaseAgent
import json

class AnalystAgent(BaseAgent):
    """Analyst Agent that performs sentiment analysis using fine-tuned BERTweet model on GPU."""
    
    def __init__(self, job_id=None):
        super().__init__(
            "AnalystAgent",
            "You are a sentiment analysis agent that analyzes data using fine-tuned BERTweet model on GPU.",
            job_id=job_id,
        )

    def run(self) -> dict:
        """
        Analyze processed data for sentiment using fine-tuned BERTweet model on GPU.
        
        Returns:
            Dictionary with sentiment analysis results
        """
        self.log("Starting sentiment analysis with GPU acceleration...")
        
        try:
            from tools.analyst_tools import (
                analyze_sentiment_batch,
                get_device_info,
                discover_aspects_bertopic,
                verify_aspects_with_llm,
            )
            
            # Log device info
            device_info = get_device_info()
            
            self.log(f"Device Information:")
            for key, value in device_info.items():
                self.log(f"  - {key}: {value}")
            
            # Load processed data from shared state
            state           = self.load_state()
            processed_data  = state.get("processed_data", [])
            meta            = state.get("metadata", {})
            topic           = meta.get("topic",           "")
            category        = meta.get("category",        "")
            category_detail = meta.get("category_detail", "")
            
            if not processed_data:
                self.log("⚠️ No processed data to analyze")
                return {"status": "warning", "message": "No processed data found"}
            
            self.log(f"Analyzing {len(processed_data)} data sources...")
            
            # Aggregate texts AND timestamps together so the timeline chart
            # can group by real date instead of falling back to synthetic dates.
            all_sentiment_texts = []
            all_created_ats     = []
            for source in processed_data:
                preprocessing = source.get("preprocessing", {})
                sentiment_data = preprocessing.get("sentiment", {})

                texts      = sentiment_data.get("comments", [])
                timestamps = sentiment_data.get("comment_timestamps", [])
                for i, text in enumerate(texts):
                    all_sentiment_texts.append(text)
                    all_created_ats.append(timestamps[i] if i < len(timestamps) else "")

                texts      = sentiment_data.get("posts", [])
                timestamps = sentiment_data.get("post_timestamps", [])
                for i, text in enumerate(texts):
                    all_sentiment_texts.append(text)
                    all_created_ats.append(timestamps[i] if i < len(timestamps) else "")
            
            if not all_sentiment_texts:
                self.log("⚠️ No texts available for sentiment analysis")
                return {"status": "warning", "message": "No texts found for sentiment analysis"}
            
            self.log(f"Analyzing {len(all_sentiment_texts)} texts for sentiment...")
            
            # Perform sentiment analysis with fine-tuned BERTweet on GPU
            sentiment_results = analyze_sentiment_batch(all_sentiment_texts)
            
            # Attach real timestamps so the timeline chart can group by date
            sentiments = sentiment_results.get("sentiments", [])
            for i, s in enumerate(sentiments):
                s["created_at"] = all_created_ats[i] if i < len(all_created_ats) else ""
            
            positive_count = sum(1 for s in sentiments if s.get("label") == "positive")
            neutral_count = sum(1 for s in sentiments if s.get("label") == "neutral")
            negative_count = sum(1 for s in sentiments if s.get("label") == "negative")
            
            total = len(sentiments)
            
            analysis_data = {
                "overall_sentiment": sentiment_results.get("overall_sentiment", "neutral"),
                "confidence": sentiment_results.get("average_confidence", 0.0),
                "data_sources": len(processed_data),
                "total_posts": sum(d.get('total_posts', 0) for d in processed_data),
                "total_comments": sum(d.get('total_comments', 0) for d in processed_data),
                "total_texts_analyzed": total,
                "sentiment_distribution": {
                    "positive": {
                        "count": positive_count,
                        "percentage": (positive_count / total * 100) if total > 0 else 0
                    },
                    "neutral": {
                        "count": neutral_count,
                        "percentage": (neutral_count / total * 100) if total > 0 else 0
                    },
                    "negative": {
                        "count": negative_count,
                        "percentage": (negative_count / total * 100) if total > 0 else 0
                    }
                },
                "detailed_sentiments": sentiments,
                "device_used": device_info.get("device", "UNKNOWN")
            }
            
            # Save sentiment analysis results
            self.save_state("sentiment_results", analysis_data)

            # ── ABSA: K-Means clustering + c-TF-IDF aspect discovery ──────────
            self.log("Running unsupervised ABSA (K-Means + c-TF-IDF)...")
            try:
                raw_aspects = discover_aspects_bertopic(
                    all_sentiment_texts,
                    sentiments,
                )
                self.log(f"Raw ABSA aspects: {list(raw_aspects.keys())}")

                # LLM verification: filter noise + polish labels for the topic
                self.log("Running LLM verification layer on aspect labels...")
                aspect_analysis = verify_aspects_with_llm(
                    raw_aspects, topic, self.llm,
                    category=category,
                    category_detail=category_detail,
                )

                self.save_state("aspect_analysis", aspect_analysis)
                self.log(f"✅ ABSA complete — {len(aspect_analysis)} final aspects: "
                         f"{list(aspect_analysis.keys())}")
            except Exception as absa_exc:
                self.log(f"⚠️ ABSA failed (non-fatal): {str(absa_exc)[:120]}")
                aspect_analysis = {}
                self.save_state("aspect_analysis", {})

            # Save analysis outputs to files for inspection
            try:
                import os
                from datetime import datetime as _dt
                topic_slug = topic.replace(" ", "_").replace("/", "_")[:60] if topic else "unknown"
                from agents.base_agent import _PROJECT_ROOT
                analysis_dir = os.path.join(_PROJECT_ROOT, "data", "analysis")
                os.makedirs(analysis_dir, exist_ok=True)

                # Sentiment results file (all comments with full text + label)
                sentiment_export = {
                    "topic": topic,
                    "saved_at": _dt.now().isoformat(),
                    "overall_sentiment": analysis_data["overall_sentiment"],
                    "confidence": analysis_data["confidence"],
                    "sentiment_distribution": analysis_data["sentiment_distribution"],
                    "total_texts_analyzed": total,
                    "detailed_sentiments": sentiments,
                }
                sentiment_path = f"{analysis_dir}/sentiment_{topic_slug}.json"
                with open(sentiment_path, "w", encoding="utf-8") as _f:
                    json.dump(sentiment_export, _f, indent=2, ensure_ascii=False)
                self.log(f"✅ Saved sentiment results → {sentiment_path}")

                # ABSA results file
                absa_export = {
                    "topic": topic,
                    "saved_at": _dt.now().isoformat(),
                    "aspects": aspect_analysis,
                }
                absa_path = f"{analysis_dir}/absa_{topic_slug}.json"
                with open(absa_path, "w", encoding="utf-8") as _f:
                    json.dump(absa_export, _f, indent=2, ensure_ascii=False)
                self.log(f"✅ Saved ABSA results → {absa_path}")

            except Exception as save_exc:
                self.log(f"⚠️ Could not save analysis files: {str(save_exc)[:100]}")

            self.log(f"✅ Sentiment analysis complete")
            self.log(f"   Device: {device_info.get('device', 'UNKNOWN')}")
            self.log(f"   Total texts analyzed: {total}")
            self.log(f"   Positive: {positive_count} ({(positive_count/total*100):.1f}%)")
            self.log(f"   Neutral: {neutral_count} ({(neutral_count/total*100):.1f}%)")
            self.log(f"   Negative: {negative_count} ({(negative_count/total*100):.1f}%)")
            self.log(f"   Average Confidence: {sentiment_results.get('average_confidence', 0):.2%}")
            
            return {
                "status": "success",
                "analysis": analysis_data
            }
        
        except Exception as e:
            self.log(f"❌ Error during sentiment analysis: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}