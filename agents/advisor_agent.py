from agents.base_agent import BaseAgent
import json

class AdvisorAgent(BaseAgent):
    """Advisor Agent that generates aspect-level recommendations and handles follow-up questions."""
    
    def __init__(self):
        super().__init__(
            "AdvisorAgent",
            "You are an advisor agent that provides strategic recommendations based on aspect-level sentiment analysis."
        )

    def run(self) -> dict:
        """Generate aspect-level recommendations based on sentiment analysis."""
    
        self.log("Starting recommendation generation...")
    
        try:
            state = self.load_state()
            sentiment_results = state.get("sentiment_results", {})
            metadata = state.get("metadata", {})
            aspect_analysis = state.get("aspect_analysis", {})
        
            if not sentiment_results:
                self.log("⚠️ No sentiment data to base recommendations on")
                return {"status": "warning", "message": "No sentiment data found"}
        
            topic = metadata.get("topic", "the analyzed topic")
            overall_sentiment = sentiment_results.get("overall_sentiment", "neutral")
        
            self.log("Generating aspect-level recommendations...")
        
            from tools.advisor_tools import generate_aspect_recommendations, generate_suggested_questions
        
            # Generate recommendations based on valid aspects only
            recommendations = generate_aspect_recommendations(
                aspect_analysis,
                topic,
                overall_sentiment,
                min_mentions=3  # Only show aspects mentioned 3+ times
            )
        
            # Generate suggested questions for follow-up
            suggested_questions = generate_suggested_questions(aspect_analysis, topic)
        
            self.save_state("recommendations", recommendations)
            self.save_state("suggested_questions", suggested_questions)
        
            self.log(f"✅ Generated {len(recommendations)} recommendations")
            self.log(f"✅ Generated {len(suggested_questions)} suggested questions")
        
            return {
                "status": "success",
                "recommendations": recommendations,
                "suggested_questions": suggested_questions,
                "aspect_analysis": aspect_analysis
            }
    
        except Exception as e:
            self.log(f"❌ Error during recommendation generation: {str(e)}")
            return {"status": "error", "error": str(e)}

    def answer_question(self, question: str) -> str:
        """
        Answer follow-up questions using the Groq LLM with full analysis context
        including any root cause findings from the timeline anomaly detection.
        """
        self.log(f"Answering question: {question}")

        try:
            state = self.load_state()
            sentiment_results = state.get("sentiment_results", {})
            aspect_analysis   = state.get("aspect_analysis", {})
            metadata          = state.get("metadata", {})
            rca_cache         = state.get("rca_cache", {})

            topic        = metadata.get("topic", "unknown")
            overall      = sentiment_results.get("overall_sentiment", "neutral")
            distribution = sentiment_results.get("sentiment_distribution", {})
            confidence   = sentiment_results.get("confidence", 0)

            # Sentiment distribution summary
            pos_pct = distribution.get("positive", {}).get("percentage", 0)
            neu_pct = distribution.get("neutral",  {}).get("percentage", 0)
            neg_pct = distribution.get("negative", {}).get("percentage", 0)
            dist_str = f"{pos_pct:.1f}% positive / {neu_pct:.1f}% neutral / {neg_pct:.1f}% negative"

            # Top aspects
            top_aspects = list(aspect_analysis.items())[:8]
            aspects_str = "\n".join(
                f"  - {a}: {d.get('positive',{}).get('percentage',0):.0f}% pos, "
                f"{d.get('negative',{}).get('percentage',0):.0f}% neg "
                f"({d.get('total_mentions', 0)} mentions)"
                for a, d in top_aspects
            ) or "  (no aspect data available)"

            # Root cause findings (from timeline anomaly analysis)
            rca_str = ""
            if rca_cache:
                rca_lines = []
                for date, r in rca_cache.items():
                    summary = r.get("root_cause_summary", "")
                    status  = r.get("status", "?")
                    aspects_hit = ", ".join(r.get("aspects_analyzed", []))
                    rca_lines.append(
                        f"  - {date}: [{status}] {summary}"
                        + (f" (aspects: {aspects_hit})" if aspects_hit else "")
                    )
                rca_str = "\nRoot Cause Analysis Findings:\n" + "\n".join(rca_lines)

            prompt = (
                f"You are a helpful assistant explaining Reddit sentiment analysis results "
                f"about '{topic}' to a curious general-public reader.\n\n"
                f"Overall sentiment: {overall.upper()} ({confidence:.0%} confidence)\n"
                f"Sentiment distribution: {dist_str}\n\n"
                f"Top discussed aspects:\n{aspects_str}\n"
                f"{rca_str}\n\n"
                f"User question: {question}\n\n"
                f"Answer conversationally in 2-4 sentences. Be specific with numbers and "
                f"aspect names where relevant. Frame the answer for a consumer deciding "
                f"whether to buy/try {topic}, not for a business analyst."
            )

            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            self.log("Question answered via LLM")
            return answer

        except Exception as e:
            self.log(f"LLM answer failed, using fallback: {str(e)[:80]}")
            # Rule-based fallback so the chat never hard-fails
            try:
                state = self.load_state()
                topic = state.get("metadata", {}).get("topic", "this topic")
                overall = state.get("sentiment_results", {}).get("overall_sentiment", "neutral")
                dist    = state.get("sentiment_results", {}).get("sentiment_distribution", {})
                pos = dist.get("positive", {}).get("percentage", 0)
                neg = dist.get("negative", {}).get("percentage", 0)
                return (
                    f"Based on the Reddit analysis of '{topic}', the overall sentiment is "
                    f"{overall.upper()} with {pos:.1f}% positive and {neg:.1f}% negative comments. "
                    f"For more detail, try asking about specific aspects like camera, battery, or price."
                )
            except Exception:
                return "I couldn't retrieve the analysis data. Please make sure an analysis has been completed."