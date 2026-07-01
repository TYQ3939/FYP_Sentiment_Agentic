"""LLM-powered advisor tools for per-tab insights, consumer recommendations, and Q&A."""

from typing import Dict, List


# ── Per-Tab LLM Insight Functions ─────────────────────────────────────────────

def generate_summary_insight(sentiment_results: Dict, topic: str, llm) -> str:
    dist = sentiment_results.get("sentiment_distribution", {})
    pos  = dist.get("positive", {}).get("percentage", 0)
    neu  = dist.get("neutral",  {}).get("percentage", 0)
    neg  = dist.get("negative", {}).get("percentage", 0)
    overall = sentiment_results.get("overall_sentiment", "neutral")
    total = sum(dist.get(k, {}).get("count", 0) for k in ("positive", "neutral", "negative"))

    prompt = (
        f"You are summarizing Reddit sentiment about '{topic}' for a general reader.\n\n"
        f"Data: {total} comments — {pos:.1f}% positive, {neu:.1f}% neutral, {neg:.1f}% negative. "
        f"Overall label: {overall.upper()}.\n\n"
        f"Write exactly 2-3 bullet points (each starting with •) highlighting the most important "
        f"takeaways about overall sentiment. Be specific with percentages.\n"
        f"Rules:\n"
        f"- Use **bold** for specific percentages (e.g. **67%**) and the sentiment label\n"
        f"- Keep each bullet to one sentence\n"
        f"- Frame for a consumer, not a business analyst"
    )
    try:
        r = llm.invoke(prompt)
        return r.content if hasattr(r, "content") else str(r)
    except Exception:
        return f"• Overall sentiment for {topic} is **{overall.upper()}** ({pos:.1f}% positive, {neg:.1f}% negative)."


def generate_timeline_insight(detailed_sentiments: List[Dict], topic: str, llm) -> str:
    summary = "Timeline data available."
    span_days = 0
    try:
        import pandas as pd
        from datetime import datetime

        records = []
        for s in detailed_sentiments:
            ts = s.get("created_at")
            if not ts:
                continue
            try:
                dt = datetime.fromtimestamp(float(str(ts).strip()))
            except Exception:
                continue
            records.append({"dt": dt, "label": s.get("label", "neutral")})

        if records:
            df = pd.DataFrame(records)
            df["date"] = df["dt"].dt.date
            daily = df.groupby(["date", "label"]).size().unstack(fill_value=0)
            for col in ["positive", "neutral", "negative"]:
                if col not in daily.columns:
                    daily[col] = 0
            daily["total"]   = daily[["positive", "neutral", "negative"]].sum(axis=1)
            daily["neg_pct"] = (daily["negative"] / daily["total"].replace(0, 1) * 100).round(1)
            peak_neg  = daily["neg_pct"].idxmax()
            recent_neg = daily.tail(7)["neg_pct"].mean().round(1)
            overall_neg = daily["neg_pct"].mean().round(1)
            span_days   = (daily.index[-1] - daily.index[0]).days + 1
            summary = (
                f"Dataset spans {span_days} days. "
                f"Peak negative: {daily.loc[peak_neg, 'neg_pct']:.1f}% on {peak_neg}. "
                f"Overall avg negative: {overall_neg}%. Recent 7-day avg: {recent_neg}%."
            )
    except Exception:
        pass

    prompt = (
        f"You are analyzing the sentiment timeline for Reddit discussions about '{topic}'.\n\n"
        f"Stats: {summary}\n\n"
        f"Write exactly 2-3 bullet points (each starting with •) about notable trends, "
        f"spikes, or patterns in the timeline.\n"
        f"Rules:\n"
        f"- Use **bold** for specific dates, percentages, and key turning points\n"
        f"- Frame for a curious consumer, not a data scientist\n"
        f"- If no clear trend exists, say so honestly"
    )
    try:
        r = llm.invoke(prompt)
        return r.content if hasattr(r, "content") else str(r)
    except Exception:
        return f"• Sentiment varied across {span_days or 'multiple'} days of Reddit activity."


def generate_wordcloud_insight(word_frequencies: Dict, topic: str, llm) -> str:
    """word_frequencies: {overall:[(w,c),...], positive:[...], negative:[...]}"""
    def _fmt(lst, n=5):
        return ", ".join(f"**{w}** ({c})" for w, c in (lst or [])[:n]) or "N/A"

    overall_str  = _fmt(word_frequencies.get("overall",  []))
    positive_str = _fmt(word_frequencies.get("positive", []))
    negative_str = _fmt(word_frequencies.get("negative", []))

    prompt = (
        f"You are interpreting word frequency data from Reddit comments about '{topic}'.\n\n"
        f"Top overall words (word: count): {overall_str}\n"
        f"Top positive-comment words: {positive_str}\n"
        f"Top negative-comment words: {negative_str}\n\n"
        f"Write exactly 2-3 bullet points (each starting with •) explaining what these "
        f"word patterns reveal about what people are actually talking about.\n"
        f"Rules:\n"
        f"- Use **bold** for specific words you reference\n"
        f"- Focus on contrast between positive and negative word sets\n"
        f"- Frame for a general reader deciding whether to buy/try {topic}"
    )
    try:
        r = llm.invoke(prompt)
        return r.content if hasattr(r, "content") else str(r)
    except Exception:
        return f"• The most frequently discussed terms in {topic} comments reflect key themes in user feedback."


def generate_absa_insight(aspect_analysis: Dict, topic: str, llm) -> str:
    top = sorted(
        [(a, d) for a, d in aspect_analysis.items() if a.lower() != "others"],
        key=lambda x: x[1].get("total_mentions", 0), reverse=True
    )[:7]

    aspect_lines = "\n".join(
        f"- {a}: {d.get('positive',{}).get('percentage',0):.0f}% pos, "
        f"{d.get('negative',{}).get('percentage',0):.0f}% neg "
        f"({d.get('total_mentions',0)} mentions)"
        for a, d in top
    )
    prompt = (
        f"You are interpreting aspect-based sentiment analysis for Reddit discussions about '{topic}'.\n\n"
        f"Top discussed aspects:\n{aspect_lines}\n\n"
        f"Write exactly 2-3 bullet points (each starting with •) highlighting the most "
        f"notable aspect sentiment patterns.\n"
        f"Rules:\n"
        f"- Use **bold** for aspect names and key percentages\n"
        f"- Call out the most praised and most criticized aspects specifically\n"
        f"- Frame for a consumer deciding whether to buy/try {topic}"
    )
    try:
        r = llm.invoke(prompt)
        return r.content if hasattr(r, "content") else str(r)
    except Exception:
        return f"• Multiple aspects of {topic} were discussed with varying sentiment patterns."


def generate_absa_recommendations(
    aspect_analysis: Dict, overall_sentiment: str, topic: str, llm
) -> List[str]:
    """Returns 4 consumer-framed strings labelled with bold headers."""
    top = sorted(
        [(a, d) for a, d in aspect_analysis.items() if a.lower() != "others"],
        key=lambda x: x[1].get("total_mentions", 0), reverse=True
    )[:8]

    aspect_lines = "\n".join(
        f"- {a}: {d.get('positive',{}).get('percentage',0):.0f}% pos, "
        f"{d.get('negative',{}).get('percentage',0):.0f}% neg "
        f"({d.get('total_mentions',0)} mentions)"
        for a, d in top
    )
    prompt = (
        f"You are summarizing what Reddit users think about '{topic}' for a curious "
        f"general reader who is considering buying or trying it.\n\n"
        f"Overall sentiment: {overall_sentiment.upper()}\n"
        f"Aspect breakdown:\n{aspect_lines}\n\n"
        f"Write exactly 4 short paragraphs, each starting with its label on its own line:\n"
        f"**What People Love**\n[1-2 sentences on the most praised aspects]\n\n"
        f"**Common Complaints**\n[1-2 sentences on the most criticized aspects]\n\n"
        f"**Mixed Opinions**\n[1-2 sentences on aspects with divided sentiment]\n\n"
        f"**Bottom Line**\n[1 sentence overall verdict for a potential buyer]\n\n"
        f"Rules:\n"
        f"- Use **bold** for specific aspect names\n"
        f"- Be direct and specific, not vague\n"
        f"- Speak as if summarizing what other people are saying, not as a business advisor"
    )
    try:
        r = llm.invoke(prompt)
        text = r.content if hasattr(r, "content") else str(r)
        labels = ["**What People Love**", "**Common Complaints**", "**Mixed Opinions**", "**Bottom Line**"]
        sections = []
        for i, label in enumerate(labels):
            if label not in text:
                continue
            start = text.index(label)
            next_starts = [text.index(l) for l in labels if l in text and text.index(l) > start]
            end = min(next_starts) if next_starts else len(text)
            sections.append(text[start:end].strip())
        return sections if sections else [text]
    except Exception:
        return _fallback_recommendations(aspect_analysis, overall_sentiment, topic)


def generate_suggested_questions(aspect_analysis: Dict, topic: str) -> List[str]:
    """Consumer-framed suggested questions based on aspect sentiment balance."""
    questions = []
    named = sorted(
        [(a, d) for a, d in aspect_analysis.items() if a.lower() != "others"],
        key=lambda x: x[1].get("total_mentions", 0), reverse=True
    )[:5]

    for aspect, data in named:
        neg = data.get("negative", {}).get("percentage", 0)
        pos = data.get("positive", {}).get("percentage", 0)
        if neg >= 45:
            questions.append(f"Why are people unhappy with the {aspect}?")
        elif pos >= 45:
            questions.append(f"What do people like about the {aspect}?")
        else:
            questions.append(f"What are people saying about the {aspect}?")

    questions += [
        f"Is {topic} worth it based on what people are saying?",
        f"What are the biggest complaints about {topic}?",
        f"What do most people like about {topic}?",
        f"Should I buy {topic} based on Reddit reviews?",
        f"How has opinion about {topic} changed over time?",
    ]

    seen, result = set(), []
    for q in questions:
        if q.lower() not in seen:
            seen.add(q.lower())
            result.append(q)
        if len(result) == 6:
            break
    return result


# ── Rule-based fallback (no LLM) ──────────────────────────────────────────────

def generate_aspect_recommendations(
    aspect_analysis: Dict, topic: str, overall_sentiment: str, min_mentions: int = 3
) -> List[str]:
    SKIP = {
        'a','lot','both','one','two','three','you','they','them','what','which',
        'this','that','it','is','are','have','has','was','were','be','been',
        'like','just','really','very','pretty','thing','stuff','bit','way','time',
    }
    try:
        valid = {
            a: d for a, d in aspect_analysis.items()
            if d.get("total_mentions", 0) >= min_mentions
            and a.lower() not in SKIP and len(a) > 2
        }
        if not valid:
            return [
                f"**What People Love** — Overall sentiment is {overall_sentiment.upper()}.",
                "**Common Complaints** — Not enough data to identify specific pain points.",
                "**Mixed Opinions** — Multiple aspects received divided feedback.",
                f"**Bottom Line** — Community discussion about {topic} is ongoing.",
            ]
        neg_s = sorted(valid.items(), key=lambda x: x[1]["negative"]["percentage"], reverse=True)
        pos_s = sorted(valid.items(), key=lambda x: x[1]["positive"]["percentage"], reverse=True)
        love      = pos_s[0][0].title() if pos_s else "performance"
        complaint = neg_s[0][0].title() if neg_s else "reliability"
        return [
            f"**What People Love** — **{love}** receives the highest positive mentions.",
            f"**Common Complaints** — **{complaint}** has the most negative feedback.",
            f"**Mixed Opinions** — Several aspects have divided community sentiment.",
            f"**Bottom Line** — Overall sentiment is **{overall_sentiment.upper()}**.",
        ]
    except Exception:
        return [f"**Bottom Line** — Overall sentiment for {topic} is {overall_sentiment.upper()}."]


def _fallback_recommendations(aspect_analysis, overall_sentiment, topic):
    return generate_aspect_recommendations(aspect_analysis, topic, overall_sentiment)
