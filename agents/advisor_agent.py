from agents.base_agent import BaseAgent
import json

class AdvisorAgent(BaseAgent):
    """Advisor Agent that generates LLM-powered per-tab insights and handles follow-up questions."""

    def __init__(self):
        super().__init__(
            "AdvisorAgent",
            "You are an advisor agent that provides insights and recommendations based on sentiment analysis.",
        )

    def run(self) -> dict:
        """Generate LLM-powered per-tab insights and consumer recommendations."""

        self.log("Starting advisor insight generation...")

        try:
            state             = self.load_state()
            sentiment_results = state.get("sentiment_results", {})
            aspect_analysis   = state.get("aspect_analysis", {})
            metadata          = state.get("metadata", {})
            processed_data    = state.get("processed_data", [])
            detailed          = sentiment_results.get("detailed_sentiments", [])

            if not sentiment_results:
                self.log("No sentiment data available")
                return {"status": "warning", "message": "No sentiment data found"}

            topic           = metadata.get("topic", "the analyzed topic")
            category_detail = metadata.get("category_detail", "")
            overall         = sentiment_results.get("overall_sentiment", "neutral")

            from tools.advisor_tools import (
                generate_summary_insight,
                generate_timeline_insight,
                generate_wordcloud_insight,
                generate_absa_insight,
                generate_absa_recommendations,
                generate_suggested_questions,
            )
            from tools.visualization_tools import get_top_words_by_sentiment

            # Per-tab LLM insights
            self.log("Generating summary insight...")
            summary_insight = generate_summary_insight(sentiment_results, topic, self.llm)

            self.log("Generating timeline insight...")
            timeline_insight = generate_timeline_insight(detailed, topic, self.llm)

            self.log("Generating wordcloud insight...")
            word_freqs = get_top_words_by_sentiment(
                sentiment_results, processed_data, topic, category_detail
            )
            wordcloud_insight = generate_wordcloud_insight(word_freqs, topic, self.llm)

            self.log("Generating ABSA insight...")
            absa_insight = generate_absa_insight(aspect_analysis, topic, self.llm)

            self.log("Generating consumer recommendations...")
            recommendations = generate_absa_recommendations(aspect_analysis, overall, topic, self.llm)

            self.log("Generating suggested questions...")
            suggested_questions = generate_suggested_questions(aspect_analysis, topic)

            advisor_insights = {
                "summary"   : summary_insight,
                "timeline"  : timeline_insight,
                "wordcloud" : wordcloud_insight,
                "absa"      : absa_insight,
            }

            self.save_state("advisor_insights",    advisor_insights)
            self.save_state("recommendations",     recommendations)
            self.save_state("suggested_questions", suggested_questions)

            self.log(f"Advisor insights saved ({len(advisor_insights)} tabs)")

            return {
                "status"            : "success",
                "advisor_insights"  : advisor_insights,
                "recommendations"   : recommendations,
                "suggested_questions": suggested_questions,
                "aspect_analysis"   : aspect_analysis,
            }

        except Exception as e:
            self.log(f"Error during advisor generation: {str(e)}")
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