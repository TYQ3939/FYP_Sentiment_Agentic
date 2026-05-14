from agents.base_agent import BaseAgent
import json

class AnalystAgent(BaseAgent):
    """Analyst Agent that performs sentiment analysis using fine-tuned BERTweet model on GPU."""
    
    def __init__(self):
        super().__init__(
            "AnalystAgent",
            "You are a sentiment analysis agent that analyzes data using fine-tuned BERTweet model on GPU."
        )

    def run(self) -> dict:
        """
        Analyze processed data for sentiment using fine-tuned BERTweet model on GPU.
        
        Returns:
            Dictionary with sentiment analysis results
        """
        self.log("Starting sentiment analysis with GPU acceleration...")
        
        try:
            # Import from tools.analyst_tools (correct location)
            from tools.analyst_tools import analyze_sentiment_batch, get_device_info
            
            # Log device info
            device_info = get_device_info()
            
            self.log(f"Device Information:")
            for key, value in device_info.items():
                self.log(f"  - {key}: {value}")
            
            # Load processed data from shared state
            state = self.load_state()
            processed_data = state.get("processed_data", [])
            
            if not processed_data:
                self.log("⚠️ No processed data to analyze")
                return {"status": "warning", "message": "No processed data found"}
            
            self.log(f"Analyzing {len(processed_data)} data sources...")
            
            # Aggregate all sentiment texts from processed data
            all_sentiment_texts = []
            for source in processed_data:
                preprocessing = source.get("preprocessing", {})
                sentiment_data = preprocessing.get("sentiment", {})
                all_sentiment_texts.extend(sentiment_data.get("comments", []))
                all_sentiment_texts.extend(sentiment_data.get("posts", []))
            
            if not all_sentiment_texts:
                self.log("⚠️ No texts available for sentiment analysis")
                return {"status": "warning", "message": "No texts found for sentiment analysis"}
            
            self.log(f"Analyzing {len(all_sentiment_texts)} texts for sentiment...")
            
            # Perform sentiment analysis with fine-tuned BERTweet on GPU
            sentiment_results = analyze_sentiment_batch(all_sentiment_texts)
            
            # Aggregate statistics
            sentiments = sentiment_results.get("sentiments", [])
            
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
                "detailed_sentiments": sentiments[:100],  # Store first 100 for reference
                "device_used": device_info.get("device", "UNKNOWN")
            }
            
            # Save sentiment analysis results
            self.save_state("sentiment_results", analysis_data)
            
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