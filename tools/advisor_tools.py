"""Tools for advisor agent with improved recommendations and suggested questions."""

from typing import Dict, List


def generate_aspect_recommendations(aspect_analysis: Dict, topic: str, 
                                   overall_sentiment: str, min_mentions: int = 3) -> List[str]:
    """
    Generate strategic recommendations based on aspect-level sentiment.
    
    FILTERS:
    - Only considers aspects with at least min_mentions
    - Filters out meaningless aspects (pronouns, generic words)
    - Shows only tech-related aspects
    
    Args:
        aspect_analysis: Dictionary of aspects with sentiment data
        topic: Main topic being analyzed
        overall_sentiment: Overall sentiment (positive/neutral/negative)
        min_mentions: Minimum mentions for an aspect to be considered (default 3)
    
    Returns:
        List of strategic recommendations
    """
    
    recommendations = []
    
    # Filter out meaningless aspects
    MEANINGLESS_ASPECTS = {
        'a', 'lot', 'both', 'one', 'two', 'three', 'you', 'they', 'them', 'what', 'which',
        'this', 'that', 'it', 'is', 'are', 'have', 'has', 'was', 'were', 'be', 'been',
        'like', 'just', 'really', 'very', 'pretty', 'thing', 'stuff', 'bit', 'way', 'time'
    }
    
    try:
        # Filter aspects by:
        # 1. Minimum mentions (at least 3)
        # 2. Not in meaningless list
        # 3. Length > 2 characters
        valid_aspects = {
            aspect: data for aspect, data in aspect_analysis.items()
            if data.get("total_mentions", 0) >= min_mentions
            and aspect.lower() not in MEANINGLESS_ASPECTS
            and len(aspect) > 2
        }
        
        if not valid_aspects:
            # Fallback if no valid aspects
            return [
                f"📊 Overall sentiment is {overall_sentiment.upper()}.",
                "Continue monitoring community feedback and engage with users to address concerns.",
                "Use specific aspect insights to guide product improvements."
            ]
        
        # Identify CRITICAL aspects (high negative)
        critical_aspects = []
        for aspect, sentiment_data in valid_aspects.items():
            negative_pct = sentiment_data.get("negative", {}).get("percentage", 0)
            if negative_pct >= 50:  # High threshold for "critical"
                critical_aspects.append({
                    "aspect": aspect,
                    "negative_pct": negative_pct,
                    "mentions": sentiment_data.get("total_mentions", 0)
                })
        
        critical_aspects.sort(key=lambda x: x["negative_pct"], reverse=True)
        
        if critical_aspects:
            top_issue = critical_aspects[0]
            recommendations.append(
                f"🔴 CRITICAL: **{top_issue['aspect'].title()}** has {top_issue['negative_pct']:.1f}% "
                f"negative sentiment ({top_issue['mentions']} mentions). "
                f"This is a major concern requiring immediate attention."
            )
        
        # Identify STRENGTHS (high positive)
        strong_aspects = []
        for aspect, sentiment_data in valid_aspects.items():
            positive_pct = sentiment_data.get("positive", {}).get("percentage", 0)
            if positive_pct >= 60:  # High threshold for "strength"
                strong_aspects.append({
                    "aspect": aspect,
                    "positive_pct": positive_pct,
                    "mentions": sentiment_data.get("total_mentions", 0)
                })
        
        strong_aspects.sort(key=lambda x: x["positive_pct"], reverse=True)
        
        if strong_aspects:
            top_strength = strong_aspects[0]
            recommendations.append(
                f"✅ STRENGTH: **{top_strength['aspect'].title()}** receives {top_strength['positive_pct']:.1f}% "
                f"positive sentiment ({top_strength['mentions']} mentions). "
                f"This is a key competitive advantage."
            )
        
        # Add mixed/neutral aspects
        mixed_aspects = [
            (aspect, data) for aspect, data in valid_aspects.items()
            if 40 <= data.get("positive", {}).get("percentage", 0) < 60
        ]
        
        if mixed_aspects:
            top_mixed = sorted(mixed_aspects, key=lambda x: x[1].get("total_mentions", 0), reverse=True)[0]
            recommendations.append(
                f"⚠️ MIXED: **{top_mixed[0].title()}** has divided sentiment. "
                f"Consider as an opportunity for improvement."
            )
        
        # Overall summary
        recommendations.append(
            f"📊 Overall sentiment is **{overall_sentiment.upper()}**. "
            f"Focus resources on critical issues while building on strengths."
        )
        
        return recommendations
    
    except Exception as e:
        print(f"Error in generate_aspect_recommendations: {str(e)}")
        return []


def generate_suggested_questions(aspect_analysis: Dict, topic: str) -> List[str]:
    """
    Generate suggested follow-up questions for the user.
    
    Args:
        aspect_analysis: Dictionary of aspects with sentiment data
        topic: Main topic being analyzed
    
    Returns:
        List of suggested questions user can click to ask
    """
    
    questions = []
    
    try:
        # Get top aspects
        sorted_aspects = sorted(
            aspect_analysis.items(),
            key=lambda x: x[1].get("total_mentions", 0),
            reverse=True
        )[:5]
        
        for aspect, data in sorted_aspects:
            aspect_name = aspect.title()
            
            # Suggest questions based on sentiment
            neg_pct = data.get("negative", {}).get("percentage", 0)
            pos_pct = data.get("positive", {}).get("percentage", 0)
            
            if neg_pct > 60:
                questions.append(f"What are the main issues with {aspect_name}?")
                questions.append(f"How can we improve {aspect_name}?")
            elif pos_pct > 60:
                questions.append(f"What makes {aspect_name} so well-received?")
                questions.append(f"How should we market {aspect_name}?")
            else:
                questions.append(f"What are the mixed opinions on {aspect_name}?")
        
        # Add general questions
        questions.extend([
            f"What are the top concerns about {topic}?",
            f"What do users like most about {topic}?",
            f"How does {topic} compare to competitors?",
            "What improvements would users want?",
            "What's driving the overall sentiment?"
        ])
        
        # Remove duplicates and return top 6
        return list(dict.fromkeys(questions))[:6]
    
    except Exception as e:
        print(f"Error generating suggested questions: {str(e)}")
        return [
            "What are the main issues?",
            "What do users like most?",
            "How can we improve?",
            "What's driving sentiment?"
        ]