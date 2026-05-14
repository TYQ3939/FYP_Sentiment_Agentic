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
        Answer follow-up questions from the user based on analysis.
        No LLM used here - just data-driven answers.
        """
        
        self.log(f"Answering question: {question}")
        
        try:
            # Load analysis data
            state = self.load_state()
            sentiment_results = state.get("sentiment_results", {})
            aspect_analysis = state.get("aspect_analysis", {})
            metadata = state.get("metadata", {})
            
            topic = metadata.get("topic", "unknown")
            distribution = sentiment_results.get('sentiment_distribution', {})
            
            # Build data-driven answer without requiring langchain_groq
            question_lower = question.lower()
            
            # Answer based on keywords
            if 'aspect' in question_lower or 'topic' in question_lower or 'what' in question_lower:
                top_aspects = list(aspect_analysis.items())[:3]
                answer = f"The top 3 aspects discussed about '{topic}' are:\n\n"
                for i, (aspect, data) in enumerate(top_aspects, 1):
                    neg_pct = data['negative']['percentage']
                    pos_pct = data['positive']['percentage']
                    answer += f"{i}. **{aspect.title()}**: "
                    if neg_pct > 60:
                        answer += f"Highly negative ({neg_pct:.1f}% negative)\n"
                    elif pos_pct > 60:
                        answer += f"Highly positive ({pos_pct:.1f}% positive)\n"
                    else:
                        answer += f"Mixed sentiment\n"
            
            elif 'negative' in question_lower or 'problem' in question_lower or 'issue' in question_lower:
                negative_aspects = sorted(
                    [(a, d) for a, d in aspect_analysis.items()],
                    key=lambda x: x[1]['negative']['percentage'],
                    reverse=True
                )[:3]
                
                answer = "The top 3 aspects with negative sentiment are:\n\n"
                for i, (aspect, data) in enumerate(negative_aspects, 1):
                    answer += f"{i}. **{aspect.title()}**: {data['negative']['percentage']:.1f}% negative ({data['negative']['count']} mentions)\n"
            
            elif 'positive' in question_lower or 'strength' in question_lower or 'good' in question_lower:
                positive_aspects = sorted(
                    [(a, d) for a, d in aspect_analysis.items()],
                    key=lambda x: x[1]['positive']['percentage'],
                    reverse=True
                )[:3]
                
                answer = "The top 3 aspects with positive sentiment are:\n\n"
                for i, (aspect, data) in enumerate(positive_aspects, 1):
                    answer += f"{i}. **{aspect.title()}**: {data['positive']['percentage']:.1f}% positive ({data['positive']['count']} mentions)\n"
            
            elif 'overall' in question_lower or 'sentiment' in question_lower:
                pos_pct = distribution.get('positive', {}).get('percentage', 0)
                neu_pct = distribution.get('neutral', {}).get('percentage', 0)
                neg_pct = distribution.get('negative', {}).get('percentage', 0)
                
                answer = f"**Overall Sentiment Analysis for '{topic}':**\n\n"
                answer += f"- Positive: {pos_pct:.1f}%\n"
                answer += f"- Neutral: {neu_pct:.1f}%\n"
                answer += f"- Negative: {neg_pct:.1f}%\n\n"
                answer += f"The dominant sentiment is **{sentiment_results.get('overall_sentiment', 'neutral').upper()}**"
            
            else:
                # Generic answer for other questions
                answer = (
                    f"Based on the analysis of '{topic}':\n\n"
                    f"The overall sentiment is {sentiment_results.get('overall_sentiment', 'neutral').upper()} "
                    f"({sentiment_results.get('confidence', 0):.0%} confidence).\n\n"
                    f"Key aspects discussed: {', '.join([a for a in list(aspect_analysis.keys())[:5]])}\n\n"
                    f"For more specific information, ask about positive/negative aspects, "
                    f"or specific topics you're interested in."
                )
            
            self.log(f"✅ Question answered")
            
            return answer
        
        except Exception as e:
            self.log(f"❌ Error answering question: {str(e)}")
            return f"I encountered an error while answering: {str(e)[:100]}"