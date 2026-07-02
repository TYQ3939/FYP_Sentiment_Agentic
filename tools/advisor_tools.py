"""LLM-powered insight and recommendation tools for the Advisor Agent."""

from typing import Dict, List


# ─── internal helper ──────────────────────────────────────────────────────────

def _call_llm_safe(llm, prompt: str, fallback: str = "") -> str:
    """Invoke the LLM; return fallback string on any error."""
    try:
        response = llm.invoke(prompt)
        return (response.content if hasattr(response, "content") else str(response)).strip()
    except Exception as exc:
        print(f"  [advisor] LLM call failed: {str(exc)[:120]}")
        return fallback


def _format_bullets(text: str) -> str:
    """
    Normalise LLM bullet-point output for Streamlit markdown rendering.

    The LLM sometimes returns all bullets on one line or separated by single
    newlines. Markdown collapses single newlines into spaces, so each bullet
    must be separated by a blank line (\n\n) to render on its own line.

    Handles both "• text • text" (same line) and "• text\n• text" (single LF).
    Also converts numbered lists ("1. text\n2. text") the same way.
    """
    if not text:
        return text

    # Split on bullet character '•' (ignoring empty fragments)
    if "•" in text:
        parts = [p.strip() for p in text.split("•") if p.strip()]
        return "\n\n".join(f"• {p}" for p in parts)

    # Numbered list: split on lines and rejoin with blank lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return "\n\n".join(lines)

    return text


# ─── per-section LLM insight functions ───────────────────────────────────────

def generate_absa_insight(aspect_analysis: dict, topic: str, llm) -> str:
    """
    Generate 2-3 bullet-point insights about the ABSA aspect breakdown.

    Summarises which aspects dominate, which have the strongest positive/negative
    signals, and what the pattern implies about what users care about most.

    Args:
        aspect_analysis : dict of {aspect_label -> {positive,neutral,negative,total_mentions}}
        topic           : the product / topic being analyzed
        llm             : LangChain-compatible LLM instance

    Returns:
        Multi-line string with bullet points (each starting with "•").
        Important numbers/aspects wrapped in **bold**, positive/negative
        sentiment words left for the frontend to colour.
    """
    named  = {k: v for k, v in aspect_analysis.items() if k != "Others"}
    others = aspect_analysis.get("Others")

    if not named:
        return f"• No named aspect categories were identified for {topic}."

    sorted_aspects = sorted(named.items(), key=lambda x: x[1].get("total_mentions", 0), reverse=True)[:7]

    lines = []
    for aspect, data in sorted_aspects:
        pos   = data.get("positive", {}).get("percentage", 0)
        neu   = data.get("neutral",  {}).get("percentage", 0)
        neg   = data.get("negative", {}).get("percentage", 0)
        total = data.get("total_mentions", 0)
        lines.append(
            f"  {aspect}: {pos:.0f}% positive / {neu:.0f}% neutral / "
            f"{neg:.0f}% negative  ({total} mentions)"
        )

    others_line = (
        f"\n  Others (uncategorized): {others.get('total_mentions', 0)} mentions"
        if others else ""
    )

    prompt = (
        f'You are an ABSA expert reviewing Reddit sentiment data about "{topic}".\n\n'
        f'Aspect breakdown:\n' + "\n".join(lines) + others_line +
        f'\n\nWrite exactly 2-3 concise bullet points:\n'
        f'1. Which aspect(s) dominate the discussion and what the sentiment split reveals\n'
        f'2. The most positively or negatively received aspect and what that implies\n'
        f'3. One key takeaway about what users care about most for "{topic}"\n\n'
        f'Rules:\n'
        f'- Start each bullet with "• "\n'
        f'- Use **bold** for specific percentages (e.g. **67%**) and key aspect names\n'
        f'- Keep each bullet to 1-2 sentences\n'
        f'- Reference specific aspect names and percentages from the data\n'
        f'- No introduction or conclusion sentence\n'
        f'Output only the bullet points.'
    )

    fallback_aspect = sorted_aspects[0][0] if sorted_aspects else "topics"
    return _format_bullets(_call_llm_safe(
        llm, prompt,
        fallback=f"• **{fallback_aspect}** is the most discussed aspect for {topic}."
    ))


def generate_summary_insight(analysis_data: dict, topic: str, llm) -> str:
    """
    Generate 2-3 bullet-point insights about the overall sentiment distribution.

    Args:
        analysis_data : dict saved as 'sentiment_results' in shared state
        topic         : the product / topic being analyzed
        llm           : LangChain-compatible LLM instance

    Returns:
        Multi-line string with bullet points (each starting with "•")
    """
    dist = analysis_data.get("sentiment_distribution", {})
    pos  = dist.get("positive", {})
    neu  = dist.get("neutral",  {})
    neg  = dist.get("negative", {})
    total = analysis_data.get("total_texts_analyzed", 0)

    prompt = (
        f'You are a social-media sentiment analyst. Analyze this Reddit sentiment data for "{topic}" '
        f'and write exactly 2-3 concise, actionable insights.\n\n'
        f'Overall sentiment : {analysis_data.get("overall_sentiment","N/A").upper()} '
        f'({analysis_data.get("confidence",0):.0%} confidence)\n'
        f'Positive : {pos.get("percentage",0):.1f}% ({pos.get("count",0)} comments)\n'
        f'Neutral  : {neu.get("percentage",0):.1f}% ({neu.get("count",0)} comments)\n'
        f'Negative : {neg.get("percentage",0):.1f}% ({neg.get("count",0)} comments)\n'
        f'Total analyzed : {total} texts\n\n'
        f'Rules:\n'
        f'- Start each bullet with "• "\n'
        f'- Reference specific numbers\n'
        f'- Use **bold** for specific percentages (e.g. **67%**) and the sentiment label\n'
        f'- Be specific about what the distribution reveals about public opinion on "{topic}"\n'
        f'- No introduction or conclusion sentence\n'
        f'Output only the bullet points.'
    )

    return _format_bullets(_call_llm_safe(
        llm, prompt,
        fallback=f"• Overall sentiment for {topic} is "
                 f"{analysis_data.get('overall_sentiment','neutral').upper()} "
                 f"based on {total} analyzed texts."
    ))


def generate_timeline_insight(detailed_sentiments: list, topic: str, llm) -> str:
    """
    Compute daily sentiment aggregations then ask the LLM for trend insights.

    Args:
        detailed_sentiments : list of {'label','confidence','text','created_at'} dicts
        topic               : the product / topic being analyzed
        llm                 : LangChain-compatible LLM instance

    Returns:
        Multi-line string with bullet points
    """
    from datetime import datetime as _dt

    # ── aggregate by day ──────────────────────────────────────────────────────
    groups: Dict[str, Dict[str, int]] = {}
    for s in detailed_sentiments:
        ts    = s.get("created_at", "")
        label = s.get("label", "neutral")
        try:
            key = _dt.fromtimestamp(float(ts)).strftime("%Y-%m-%d") if ts else None
        except (ValueError, TypeError, OSError):
            key = None
        if key is None:
            continue
        if key not in groups:
            groups[key] = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
        groups[key][label] = groups[key].get(label, 0) + 1
        groups[key]["total"] += 1

    if not groups:
        return f"• Insufficient timestamp data to analyze sentiment trends for {topic}."

    sorted_groups = sorted(groups.items())
    lines = []
    for date, c in sorted_groups:
        t = c["total"]
        if t > 0:
            lines.append(
                f"  {date}: {c['positive']/t*100:.0f}% positive, "
                f"{c['neutral']/t*100:.0f}% neutral, "
                f"{c['negative']/t*100:.0f}% negative  ({t} comments)"
            )

    # Trim to avoid token overflow (keep first 5, middle sample, last 5)
    if len(lines) > 15:
        mid   = len(lines) // 2
        lines = (lines[:5]
                 + [f"  ... ({len(lines)-10} more periods) ..."]
                 + lines[mid:mid+3]
                 + lines[-5:])

    prompt = (
        f'You are a social-media trend analyst. Here are daily sentiment breakdowns '
        f'from Reddit for "{topic}":\n\n'
        + "\n".join(lines)
        + f'\n\nWrite 2-3 bullet points:\n'
          f'1. Whether sentiment is improving, declining, or stable over time\n'
          f'2. Any notable spikes or drops\n'
          f'3. What the trend suggests about public perception of "{topic}"\n\n'
          f'Rules:\n'
          f'- Start each bullet with "• "\n'
          f'- Reference specific dates or periods from the data above\n'
          f'- Use **bold** for specific dates, percentages, and key turning points\n'
          f'- No introduction or conclusion\n'
          f'Output only the bullet points.'
    )

    return _format_bullets(_call_llm_safe(
        llm, prompt,
        fallback=f"• Sentiment trends analyzed across {len(groups)} time periods for {topic}."
    ))


def generate_wordcloud_insight(word_frequencies: dict, topic: str, llm) -> str:
    """
    Analyze the actual word-frequency data behind the wordclouds (top words per
    sentiment bucket, NOUN/PROPN/ADJ lemmas) and ask the LLM what the language
    patterns reveal — genuinely about word usage, not a re-statement of ABSA
    aspect data.

    Args:
        word_frequencies : dict from get_top_words_by_sentiment(), e.g.
                            {"overall": [(word, count), ...], "positive": [...], ...}
        topic            : the product / topic being analyzed
        llm              : LangChain-compatible LLM instance

    Returns:
        Multi-line string with bullet points
    """
    overall  = word_frequencies.get("overall",  [])
    positive = word_frequencies.get("positive", [])
    negative = word_frequencies.get("negative", [])

    if not overall:
        return f"• Insufficient data to generate word-frequency insights for {topic}."

    def _fmt(pairs):
        return ", ".join(f"{w} ({c})" for w, c in pairs) or "none identified"

    prompt = (
        f'You are a text-analysis expert reviewing word-frequency data from Reddit discussions '
        f'about "{topic}". These are the most frequent words actually used in comments, with '
        f'their occurrence counts.\n\n'
        f'Most frequent words overall      : {_fmt(overall[:12])}\n'
        f'Most frequent words in POSITIVE comments : {_fmt(positive[:8])}\n'
        f'Most frequent words in NEGATIVE comments : {_fmt(negative[:8])}\n\n'
        f'Write 2-3 bullet points about:\n'
        f'1. What the most frequent words reveal about what people focus on when discussing "{topic}"\n'
        f'2. Which words stand out in positive vs. negative comments and what that implies\n'
        f'3. One takeaway about the overall language/tone people use\n\n'
        f'Rules:\n'
        f'- Start each bullet with "• "\n'
        f'- Reference specific words from the data above\n'
        f'- Use **bold** for the specific words you reference\n'
        f'- No introduction or conclusion\n'
        f'Output only the bullet points.'
    )

    return _format_bullets(_call_llm_safe(
        llm, prompt,
        fallback=(
            "• Most frequent words: "
            + ", ".join(w for w, _ in overall[:5])
            + f" for {topic}."
        )
    ))


def generate_absa_recommendations(
    aspect_analysis: dict,
    topic: str,
    overall_sentiment: str,
    llm,
) -> List[str]:
    """
    Generate LLM-powered "what others are thinking" takeaways from aspect-level
    sentiment, written for a curious general-public reader (a potential buyer
    or follower of the topic) rather than a business/strategic audience.

    Returns a list of recommendation strings.
    Falls back to rule-based generate_aspect_recommendations() on LLM failure.
    """
    if not aspect_analysis:
        return [
            f"**Bottom Line**: Overall sentiment for {topic} is {overall_sentiment.upper()}, "
            "but not enough aspect-level detail was found to break it down further.",
        ]

    sorted_aspects = sorted(
        aspect_analysis.items(),
        key=lambda x: x[1].get("total_mentions", 0),
        reverse=True
    )[:10]

    aspect_lines = []
    for aspect, data in sorted_aspects:
        pos = data.get("positive", {}).get("percentage", 0)
        neu = data.get("neutral",  {}).get("percentage", 0)
        neg = data.get("negative", {}).get("percentage", 0)
        mentions = data.get("total_mentions", 0)
        aspect_lines.append(
            f"  - {aspect}: {pos:.0f}% positive, {neu:.0f}% neutral, "
            f"{neg:.0f}% negative ({mentions} mentions)"
        )

    prompt = (
        f'You are summarizing what Reddit users think about "{topic}" for a curious general reader — '
        f'someone who is not an industry insider, but wants a quick, honest sense of what other people '
        f'are saying before they decide whether to care about, buy, or follow "{topic}".\n\n'
        f'Community overall sentiment: {overall_sentiment.upper()}\n\n'
        f'Aspect-level breakdown:\n'
        + "\n".join(aspect_lines)
        + "\n\nWrite 4 short, natural takeaways a curious reader would actually want to know."
          "\n\nFormat each line EXACTLY like this:"
          "\n**What People Love**: one concise, natural sentence referencing a specific aspect and number."
          "\n**Common Complaints**: one concise, natural sentence referencing a specific aspect and number."
          "\n**Mixed Opinions**: one concise, natural sentence about an aspect with divided sentiment."
          "\n**Bottom Line**: one concise, practical takeaway — what someone should know before deciding "
          "whether to care about, buy, or follow this."
          "\n\nRules:"
          "\n- Use only these labels: What People Love, Common Complaints, Mixed Opinions, Bottom Line"
          "\n- Write for a general audience, not a business executive — no corporate jargon"
          "\n- One sentence per line, under 25 words, conversational tone"
          "\n- Reference specific aspect names and percentages from the data"
          "\n- No numbering, no brackets, no introduction or conclusion"
          "\n- Output only the four lines."
    )

    raw = _call_llm_safe(llm, prompt)

    if raw:
        lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
        # Accept lines that start with **Label**: (new format) or any bullet/number (fallback)
        recs = [
            ln for ln in lines
            if ln.startswith("**") or ln[0:1].isdigit()
            or ln.startswith("-") or ln.startswith("•")
        ]
        if recs:
            return recs
        if lines:
            return lines[:5]

    # LLM failed — fall back to rule-based
    return generate_aspect_recommendations(aspect_analysis, topic, overall_sentiment)


# ─── keep original rule-based functions as fallbacks ─────────────────────────

def generate_aspect_recommendations(
    aspect_analysis: Dict,
    topic: str,
    overall_sentiment: str,
    min_mentions: int = 3,
) -> List[str]:
    """
    Rule-based strategic recommendations based on aspect-level sentiment.
    Used as fallback when the LLM is unavailable.
    """
    recommendations = []

    MEANINGLESS = {
        'a','lot','both','one','two','three','you','they','them','what','which',
        'this','that','it','is','are','have','has','was','were','be','been',
        'like','just','really','very','pretty','thing','stuff','bit','way','time'
    }

    try:
        valid = {
            aspect: data for aspect, data in aspect_analysis.items()
            if data.get("total_mentions", 0) >= min_mentions
            and aspect.lower() not in MEANINGLESS
            and len(aspect) > 2
        }

        if not valid:
            return [
                f"**Bottom Line**: Overall sentiment is {overall_sentiment.upper()}, but there "
                "wasn't enough aspect-level discussion to break it down further.",
            ]

        strong = sorted(
            [
                {"aspect": a, "positive_pct": d.get("positive",{}).get("percentage",0),
                 "mentions": d.get("total_mentions",0)}
                for a, d in valid.items()
                if d.get("positive",{}).get("percentage",0) >= 60
            ],
            key=lambda x: x["positive_pct"], reverse=True
        )
        if strong:
            top = strong[0]
            recommendations.append(
                f"**What People Love**: {top['aspect'].title()} gets {top['positive_pct']:.1f}% "
                f"positive sentiment ({top['mentions']} mentions) from people discussing it."
            )

        critical = sorted(
            [
                {"aspect": a, "negative_pct": d.get("negative",{}).get("percentage",0),
                 "mentions": d.get("total_mentions",0)}
                for a, d in valid.items()
                if d.get("negative",{}).get("percentage",0) >= 50
            ],
            key=lambda x: x["negative_pct"], reverse=True
        )
        if critical:
            top = critical[0]
            recommendations.append(
                f"**Common Complaints**: {top['aspect'].title()} draws {top['negative_pct']:.1f}% "
                f"negative sentiment ({top['mentions']} mentions) — a recurring pain point."
            )

        mixed = sorted(
            [(a, d) for a, d in valid.items()
             if 40 <= d.get("positive",{}).get("percentage",0) < 60],
            key=lambda x: x[1].get("total_mentions",0), reverse=True
        )
        if mixed:
            top = mixed[0]
            recommendations.append(
                f"**Mixed Opinions**: {top[0].title()} has divided sentiment, "
                "so opinions on it really depend on who you ask."
            )

        recommendations.append(
            f"**Bottom Line**: Overall sentiment is {overall_sentiment.upper()} — "
            "worth weighing the praised aspects against the recurring complaints above."
        )

        return recommendations

    except Exception as exc:
        print(f"Error in generate_aspect_recommendations: {str(exc)}")
        return []


def generate_suggested_questions(aspect_analysis: Dict, topic: str) -> List[str]:
    """
    Generate suggested follow-up questions framed for a curious general-public
    reader who wants to know what other people on Reddit think about the
    topic/product/trend — not a business/marketing audience.
    """
    questions = []

    try:
        named = {k: v for k, v in aspect_analysis.items() if k != "Others"}
        sorted_aspects = sorted(
            named.items(),
            key=lambda x: x[1].get("total_mentions", 0),
            reverse=True
        )[:4]

        for aspect, data in sorted_aspects:
            name    = aspect.title()
            neg_pct = data.get("negative", {}).get("percentage", 0)
            pos_pct = data.get("positive", {}).get("percentage", 0)

            if neg_pct >= 45:
                questions.append(f"Why are people unhappy with the {name}?")
            elif pos_pct >= 45:
                questions.append(f"What do people like about the {name}?")
            else:
                questions.append(f"What are people saying about the {name}?")

        questions.extend([
            f"What's the overall buzz around {topic}?",
            f"What are people most excited about with {topic}?",
            f"What are the biggest complaints about {topic}?",
            f"Is {topic} worth it based on what people are saying?",
            f"What should I know before getting into {topic}?",
        ])

        return list(dict.fromkeys(questions))[:6]

    except Exception as exc:
        print(f"Error generating suggested questions: {str(exc)}")
        return [
            "What are people saying overall?",
            "What do people like most?",
            "What are the common complaints?",
            "Is this worth it based on Reddit discussions?",
        ]


# ─── compare mode ─────────────────────────────────────────────────────────────

def generate_compare_insight(
    topic_a: str,
    sentiment_a: dict,
    aspects_a: dict,
    topic_b: str,
    sentiment_b: dict,
    aspects_b: dict,
    llm,
) -> str:
    """
    Generate a competitive benchmark synthesis report comparing two topics.
    Returns markdown text with three sections:
      # Competitive Edge
      # System Vulnerabilities
      # Market Sentiment Verdict
    """

    def _dist_line(topic, sr):
        d    = sr.get("sentiment_distribution", {})
        pos  = d.get("positive", {}).get("percentage", 0)
        neu  = d.get("neutral",  {}).get("percentage", 0)
        neg  = d.get("negative", {}).get("percentage", 0)
        ov   = sr.get("overall_sentiment", "N/A").upper()
        conf = sr.get("confidence", 0)
        n    = sr.get("total_texts_analyzed", 0)
        return (
            f"{topic}: {ov} ({conf:.0%} confidence, {n} texts)\n"
            f"  Positive {pos:.1f}%  Neutral {neu:.1f}%  Negative {neg:.1f}%"
        )

    def _aspects_block(aspects, top_n=6):
        rows = sorted(
            [(k, v) for k, v in aspects.items() if k != "Others"],
            key=lambda x: x[1].get("total_mentions", 0), reverse=True
        )[:top_n]
        lines = []
        for name, data in rows:
            pos = data.get("positive", {}).get("percentage", 0)
            neg = data.get("negative", {}).get("percentage", 0)
            mentions = data.get("total_mentions", 0)
            lines.append(f"  {name}: {pos:.0f}% pos / {neg:.0f}% neg ({mentions} mentions)")
        return "\n".join(lines) if lines else "  (no aspects identified)"

    context = (
        "=== Sentiment Overview ===\n"
        f"{_dist_line(topic_a, sentiment_a)}\n"
        f"{_dist_line(topic_b, sentiment_b)}\n\n"
        f"=== Top Aspects — {topic_a} ===\n{_aspects_block(aspects_a)}\n\n"
        f"=== Top Aspects — {topic_b} ===\n{_aspects_block(aspects_b)}"
    )

    prompt = (
        f"You are a competitive market analyst reviewing Reddit sentiment data.\n"
        f"Topic A: {topic_a}\n"
        f"Topic B: {topic_b}\n\n"
        f"Data:\n{context}\n\n"
        f"Write a benchmark synthesis report using EXACTLY these three markdown section headers:\n\n"
        f"# Competitive Edge\n"
        f"2-3 bullet points identifying where one topic clearly outperforms the other "
        f"based on specific sentiment percentages and aspect scores.\n\n"
        f"# System Vulnerabilities\n"
        f"2-3 bullet points spotlighting the most critical weaknesses or recurring "
        f"complaint patterns for each topic.\n\n"
        f"# Market Sentiment Verdict\n"
        f"1 short paragraph concluding which topic currently dominates public "
        f"perception and why, based only on the data.\n\n"
        f"Rules:\n"
        f"- Use **bold** for specific numbers, topic names, and key aspects\n"
        f"- Start bullet points with '• '\n"
        f"- Be specific — cite percentages and aspect names from the data\n"
        f"- Write for a general audience, not corporate executives\n"
        f"- Output only the three sections, no extra text."
    )

    fallback = (
        f"# Competitive Edge\n"
        f"• Comparison between **{topic_a}** and **{topic_b}** loaded — "
        f"review the individual tabs for detailed breakdowns.\n\n"
        f"# System Vulnerabilities\n"
        f"• Check each topic's Aspect Analysis tab for complaint patterns.\n\n"
        f"# Market Sentiment Verdict\n"
        f"Unable to generate verdict — LLM unavailable at this time."
    )

    return _call_llm_safe(llm, prompt, fallback=fallback)
