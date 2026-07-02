from agents.base_agent import BaseAgent
import json


class AdvisorAgent(BaseAgent):
    """Generates LLM-powered per-section insights and handles follow-up Q&A."""

    def __init__(self, job_id=None):
        super().__init__(
            "AdvisorAgent",
            "You are an advisor agent that provides strategic insights based on Reddit sentiment analysis.",
            job_id=job_id,
        )

    # ── main pipeline step ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Generate insights for each analysis section (summary, timeline,
        wordcloud, ABSA) and save them to shared state.

        Returns:
            {
                "status": "success" | "warning" | "error",
                "advisor_insights": {
                    "summary"  : str,
                    "timeline" : str,
                    "wordcloud": str,
                    "absa"     : list[str]   # numbered recommendations
                },
                "recommendations"   : list[str],   # same as advisor_insights.absa
                "suggested_questions": list[str],
                "aspect_analysis"   : dict
            }
        """
        self.log("Starting advisor analysis...")

        try:
            from tools.advisor_tools import (
                generate_summary_insight,
                generate_timeline_insight,
                generate_wordcloud_insight,
                generate_absa_insight,
                generate_absa_recommendations,
                generate_suggested_questions,
            )

            # ── load analysis data produced by previous agents ────────────────
            state            = self.load_state()
            sentiment_results = state.get("sentiment_results", {})
            aspect_analysis  = state.get("aspect_analysis",   {})
            metadata         = state.get("metadata",           {})
            processed_data   = state.get("processed_data",     [])

            if not sentiment_results:
                self.log("No sentiment data found in shared state")
                return {"status": "warning", "message": "No sentiment data found"}

            topic             = metadata.get("topic", "the analyzed topic")
            category_detail   = metadata.get("category_detail", "")
            overall_sentiment = sentiment_results.get("overall_sentiment", "neutral")
            detailed_sentiments = sentiment_results.get("detailed_sentiments", [])

            self.log(f"Topic: {topic}")
            self.log(f"Overall sentiment: {overall_sentiment}")
            self.log(f"Aspect count: {len(aspect_analysis)}")
            self.log(f"Detailed sentiments: {len(detailed_sentiments)}")

            # ── generate per-section LLM insights ────────────────────────────

            self.log("Generating summary insight...")
            summary_insight = _safe_generate(
                generate_summary_insight, sentiment_results, topic, self.llm,
                fallback=f"Overall sentiment for {topic} is {overall_sentiment.upper()}."
            )

            self.log("Generating timeline insight...")
            timeline_insight = _safe_generate(
                generate_timeline_insight, detailed_sentiments, topic, self.llm,
                fallback=f"Sentiment timeline analyzed for {topic}."
            )

            self.log("Generating wordcloud / theme insight...")
            from tools.visualization_tools import get_top_words_by_sentiment
            word_frequencies = get_top_words_by_sentiment(
                sentiment_results, processed_data,
                topic=topic, category_detail=category_detail,
            )
            wordcloud_insight = _safe_generate(
                generate_wordcloud_insight, word_frequencies, topic, self.llm,
                fallback=f"Key discussion themes identified for {topic}."
            )

            self.log("Generating ABSA insight...")
            absa_insight = _safe_generate(
                generate_absa_insight, aspect_analysis, topic, self.llm,
                fallback=f"Aspect analysis complete for {topic}."
            )

            self.log("Generating ABSA recommendations...")
            absa_recs = _safe_generate(
                generate_absa_recommendations, aspect_analysis, topic,
                overall_sentiment, self.llm,
                fallback=[f"Overall sentiment for {topic} is {overall_sentiment.upper()}."]
            )

            # ── generate suggested Q&A questions ─────────────────────────────
            suggested_questions = generate_suggested_questions(aspect_analysis, topic)

            # ── bundle and persist ────────────────────────────────────────────
            advisor_insights = {
                "summary"      : summary_insight,
                "timeline"     : timeline_insight,
                "wordcloud"    : wordcloud_insight,
                "absa_insight" : absa_insight,
                "absa"         : absa_recs,
            }

            self.save_state("advisor_insights",   advisor_insights)
            self.save_state("recommendations",    absa_recs)
            self.save_state("suggested_questions", suggested_questions)

            self.log(f"Generated 4 section insights + {len(absa_recs)} recommendations")
            self.log(f"Generated {len(suggested_questions)} suggested questions")

            return {
                "status"              : "success",
                "advisor_insights"    : advisor_insights,
                "recommendations"     : absa_recs,
                "suggested_questions" : suggested_questions,
                "aspect_analysis"     : aspect_analysis,
            }

        except Exception as exc:
            self.log(f"Error during advisor analysis: {str(exc)}")
            import traceback
            self.log(traceback.format_exc())
            return {"status": "error", "error": str(exc)}

    # ── interactive Q&A ───────────────────────────────────────────────────────

    def answer_question(self, question: str) -> str:
        """
        Answer a follow-up question using the LLM with full analysis context.
        Reads from shared_state.json so it works from both the backend thread
        and from Streamlit directly.
        """
        self.log(f"Answering: {question}")

        try:
            state             = self.load_state()
            sentiment_results = state.get("sentiment_results", {})
            aspect_analysis   = state.get("aspect_analysis",   {})
            metadata          = state.get("metadata",           {})
            advisor_insights  = state.get("advisor_insights",   {})

            topic = metadata.get("topic", "unknown topic")
            dist  = sentiment_results.get("sentiment_distribution", {})
            pos   = dist.get("positive", {})
            neu   = dist.get("neutral",  {})
            neg   = dist.get("negative", {})

            # Format top aspects
            sorted_aspects = sorted(
                aspect_analysis.items(),
                key=lambda x: x[1].get("total_mentions", 0),
                reverse=True
            )[:8]
            aspect_lines = "\n".join(
                f"  - {a}: {d.get('positive',{}).get('percentage',0):.0f}% pos, "
                f"{d.get('negative',{}).get('percentage',0):.0f}% neg "
                f"({d.get('total_mentions',0)} mentions)"
                for a, d in sorted_aspects
            ) or "  (no aspects identified)"

            context = (
                f'Sentiment analysis for "{topic}":\n'
                f'Overall: {sentiment_results.get("overall_sentiment","N/A").upper()} '
                f'({sentiment_results.get("confidence",0):.0%} confidence)\n'
                f'Positive : {pos.get("percentage",0):.1f}% ({pos.get("count",0)} comments)\n'
                f'Neutral  : {neu.get("percentage",0):.1f}% ({neu.get("count",0)} comments)\n'
                f'Negative : {neg.get("percentage",0):.1f}% ({neg.get("count",0)} comments)\n'
                f'Total analyzed: {sentiment_results.get("total_texts_analyzed",0)} texts\n\n'
                f'Top aspects:\n{aspect_lines}\n\n'
                f'Key insights:\n'
                f'- Summary  : {advisor_insights.get("summary","N/A")}\n'
                f'- Trends   : {advisor_insights.get("timeline","N/A")}\n'
                f'- Themes   : {advisor_insights.get("wordcloud","N/A")}\n'
            )

            prompt = (
                f"You are a sentiment analysis expert and strategic advisor.\n\n"
                f"Context:\n{context}\n"
                f"User question: {question}\n\n"
                f"Answer the question in 3-5 concise sentences. "
                f"Reference specific data from the context. "
                f"Be helpful and direct."
            )

            response = self.llm.invoke(prompt)
            answer   = response.content if hasattr(response, "content") else str(response)
            self.log("Question answered successfully")
            return answer.strip()

        except Exception as exc:
            self.log(f"Error answering question: {str(exc)}")
            return (
                f"I was unable to answer your question at this time. "
                f"Error: {str(exc)[:150]}"
            )


# ── module-level helper (avoids repetitive try/except in run()) ───────────────

def _safe_generate(fn, *args, fallback=None):
    """Call fn(*args); return fallback on any exception."""
    try:
        return fn(*args)
    except Exception as exc:
        print(f"  [advisor] {fn.__name__} failed: {str(exc)[:120]}")
        return fallback if fallback is not None else ""
