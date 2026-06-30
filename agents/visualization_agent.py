from agents.base_agent import BaseAgent
import json

class VisualizationAgent(BaseAgent):
    """Visualization Agent that prepares all visualization data."""
    
    def __init__(self):
        super().__init__(
            "VisualizationAgent",
            "You are a visualization agent that prepares data insights for visual presentation."
        )

    def run(self) -> dict:
        """Prepare all visualization data."""
        
        self.log("Preparing visualizations...")
        
        try:
            state = self.load_state()
            
            processed_data = state.get("processed_data", [])
            sentiment_results = state.get("sentiment_results", {})
            metadata = state.get("metadata", {})
            
            topic = metadata.get("topic", "")
            
            # ABSA results are pre-computed by AnalystAgent — read from state
            aspect_analysis = state.get("aspect_analysis", {})
            self.log(f"✅ Loaded {len(aspect_analysis)} ABSA aspects from shared state: "
                     f"{list(aspect_analysis.keys())}")
            
            # Prepare wordclouds with topic-aware filtering
            self.log("Preparing wordcloud data...")
            
            visualization_data = {
                "topic": metadata.get("topic", "Unknown"),
                "category": metadata.get("category", ""),
                "category_detail": metadata.get("category_detail", ""),
                "total_posts": metadata.get("total_posts", 0),
                "total_comments": metadata.get("total_comments", 0),
                "subreddits": metadata.get("subreddits_scraped", []),
                "sentiment_distribution": sentiment_results.get("sentiment_distribution", {}),
                "overall_sentiment": sentiment_results.get("overall_sentiment", "neutral"),
                "confidence": sentiment_results.get("confidence", 0.0),
                "aspect_analysis": aspect_analysis,
                "wordclouds_generated": len(processed_data) > 0
            }
            
            self.log(f"✅ Visualization data prepared")
            
            return {
                "status": "success",
                "visualization_data": visualization_data
            }
        
        except Exception as e:
            self.log(f"❌ Error during visualization: {str(e)}")
            return {"status": "error", "error": str(e)}