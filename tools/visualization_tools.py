"""Comprehensive visualization tools for sentiment analysis - Streamlit compatible."""

import os
import json
from typing import List, Dict
from datetime import datetime
import warnings

# Set matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import plotly (always needed)
import plotly.graph_objects as go
import plotly.express as px

# NOTE: WordCloud is imported only when needed (lazy import in function)
# This prevents errors when wordcloud is not installed but other functions are used

warnings.filterwarnings('ignore')


def _nice_y_dtick(max_val: int, target_ticks: int = 8) -> int:
    """
    Return a 'round' tick interval so the y-axis shows clean values like
    0, 10, 20, 30 rather than irregular steps from integer division.
    Picks the smallest value from a human-readable set that gives at most
    target_ticks tick marks for the given max value.
    """
    if max_val <= 0:
        return 1
    rough = max_val / target_ticks
    nice_steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000]
    for step in nice_steps:
        if step >= rough:
            return step
    return nice_steps[-1]


def generate_timeline_chart(sentiments: List[Dict], use_timestamps: bool = True) -> go.Figure:
    """
    Generate timeline chart with dynamic time resolution.

    Chooses the grouping granularity (hourly / daily / weekly) based on
    the actual span of the data so the chart stays readable even when all
    comments were posted on the same day.
    """

    if not sentiments:
        return None

    try:
        from datetime import datetime, timedelta
        import pandas as pd

        records = []

        for s in sentiments:
            try:
                ts = s.get("created_at")
                dt = None
                if ts:
                    ts_str = str(ts).strip()
                    try:
                        dt = datetime.fromtimestamp(float(ts_str))
                    except (ValueError, OSError, OverflowError):
                        try:
                            dt = datetime.strptime(ts_str[:16], "%Y-%m-%dT%H:%M")
                        except ValueError:
                            try:
                                dt = datetime.strptime(ts_str[:10], "%Y-%m-%d")
                            except ValueError:
                                dt = None

                if dt is None:
                    # Spread synthetic entries across the past few days
                    dt = datetime.now() - timedelta(minutes=len(records) * 10)

                records.append({"dt": dt, "sentiment": s.get("label", "neutral")})

            except Exception as e:
                print(f"Error parsing sentiment timestamp: {str(e)[:50]}")
                continue

        if not records:
            return None

        df = pd.DataFrame(records)

        # Determine span of real data to pick granularity
        dt_min = df["dt"].min()
        dt_max = df["dt"].max()
        span_hours = max((dt_max - dt_min).total_seconds() / 3600, 0)

        if span_hours <= 48:
            # Hourly buckets
            df["period"] = df["dt"].apply(
                lambda d: d.strftime("%m-%d %H:00")
            )
            title = "Sentiment Timeline (Hourly)"
            xlabel = "Hour"
            ylabel = "Count per Hour"
        elif span_hours <= 24 * 90:
            # Daily buckets
            df["period"] = df["dt"].apply(lambda d: d.strftime("%Y-%m-%d"))
            title = "Sentiment Timeline (Daily)"
            xlabel = "Date"
            ylabel = "Count per Day"
        else:
            # Weekly buckets — use Monday of each ISO week
            df["period"] = df["dt"].apply(
                lambda d: (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")
            )
            title = "Sentiment Timeline (Weekly)"
            xlabel = "Week starting"
            ylabel = "Count per Week"

        grouped = df.groupby(["period", "sentiment"]).size().unstack(fill_value=0)

        for col in ["positive", "neutral", "negative"]:
            if col not in grouped.columns:
                grouped[col] = 0

        grouped = grouped.sort_index()[["positive", "neutral", "negative"]]
        periods = grouped.index.tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=periods,
            y=grouped["positive"].astype(int),
            mode="lines+markers",
            name="Positive",
            line=dict(color="#2ecc71", width=3),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=periods,
            y=grouped["neutral"].astype(int),
            mode="lines+markers",
            name="Neutral",
            line=dict(color="#95a5a6", width=3),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=periods,
            y=grouped["negative"].astype(int),
            mode="lines+markers",
            name="Negative",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=8),
        ))

        max_count = int(max(
            grouped["positive"].max(),
            grouped["neutral"].max(),
            grouped["negative"].max(),
        ))
        y_dtick   = _nice_y_dtick(max_count)
        y_max     = max(y_dtick, (max_count // y_dtick + 1) * y_dtick)

        # Tilt x-axis labels only when there are many periods
        n_periods = len(periods)
        x_angle = -45 if n_periods > 12 else (-30 if n_periods > 6 else 0)

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            height=420,
            hovermode="x unified",
            xaxis_tickangle=x_angle,
            yaxis=dict(
                tickformat="d",
                tick0=0,
                dtick=y_dtick,
                range=[0, y_max],
            ),
        )

        return fig

    except Exception as e:
        print(f"Error in generate_timeline_chart: {str(e)}")
        return None


# Generic product/category self-reference words that tend to dominate
# wordclouds without adding insight (e.g. "phone" for an "iPhone" topic,
# "smartphone" pulled from category_detail). Added to the topic stopwords
# whenever they appear as a substring of a topic/category token.
_GENERIC_SELF_REF_WORDS = {
    "phone", "smartphone", "device", "laptop", "tablet", "watch", "camera",
    "app", "application", "game", "show", "movie", "series", "song", "album",
    "car", "vehicle", "headphone", "headphones", "earbud", "earbuds",
    "speaker", "console", "computer", "pc", "tv", "television",
}


def _build_topic_stopwords(topic: str = "", category_detail: str = "") -> set:
    """
    Build a dynamic stopword set from the topic + category description so
    self-referential words (e.g. "phone" dominating an "iPhone 17 Pro Max"
    wordcloud) don't drown out genuinely informative words.
    """
    import re as _re

    tokens = set()
    for source in (topic or "", category_detail or ""):
        tokens.update(_re.findall(r"[a-z]+", source.lower()))

    stops = set(tokens)
    for generic in _GENERIC_SELF_REF_WORDS:
        if any(generic in tok for tok in tokens):
            stops.add(generic)

    return stops


def generate_wordcloud_by_sentiment(
    sentiment_data: Dict,
    processed_data: List[Dict],
    topic: str = "",
    category_detail: str = "",
) -> Dict[str, bytes]:
    """
    Generate wordclouds broken down by sentiment and POS type.

    Text source strategy (most-to-least reliable):
      1. Rebuild BERTweet-preprocessed texts from processed_data.preprocessing.sentiment
         and match sentiment labels from detailed_sentiments by index.
         These texts are ALWAYS full-length regardless of API payload truncation.
      2. Fall back to detailed_sentiments[i].text if processed_data yields nothing.

    Returns keys: "overall", "<sentiment>_noun", "<sentiment>_verb", "<sentiment>_adj"
    """

    try:
        from wordcloud import WordCloud
    except ImportError:
        return {}

    import io

    wordclouds = {}

    custom_stops = _build_topic_stopwords(topic, category_detail)

    nlp_wc = None
    try:
        import spacy
        nlp_wc = spacy.load("en_core_web_sm")
    except Exception:
        pass  # spaCy unavailable; fallback to regex tokenisation

    POS_GROUPS = {
        "noun": ["NOUN", "PROPN"],
        "verb": ["VERB"],
        "adj":  ["ADJ"],
    }
    COLORMAPS = {
        "positive": "Greens",
        "neutral":  "Blues",
        "negative": "Reds",
    }

    def _render_wordcloud(text, colormap):
        if not text or len(text.strip()) < 10:
            return None
        try:
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap=colormap,
                prefer_horizontal=0.7,
                min_font_size=10,
                max_words=100,
            ).generate(text)
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            return None

    def _extract_pos_words(texts, pos_tags):
        """POS-filter a list of texts and return a single space-joined token string."""
        words = []
        for text in texts:
            if not text:
                continue
            clean = text.replace("HTTPURL", "").replace("@USER", "").strip()
            if not clean:
                continue
            if nlp_wc:
                try:
                    doc = nlp_wc(clean)
                    for token in doc:
                        if (
                            token.pos_ in pos_tags
                            and not token.is_stop
                            and token.is_alpha
                            and len(token.lemma_) > 2
                            and token.lemma_.lower() not in custom_stops
                        ):
                            words.append(token.lemma_.lower())
                except Exception:
                    pass
            else:
                # Fallback: no POS distinction — keep all meaningful alpha tokens
                import re
                for w in re.findall(r"[a-zA-Z]{3,}", clean.lower()):
                    if w not in custom_stops:
                        words.append(w)
        return " ".join(words)

    try:
        detailed_sentiments = sentiment_data.get("detailed_sentiments", [])

        # ── Strategy 1: rebuild full texts from processed_data + index-match labels ──
        # processed_data.preprocessing.sentiment.comments are the BERTweet-preprocessed
        # texts in the same order AnalystAgent used to produce detailed_sentiments.
        all_sentiment_texts = []
        for source in processed_data:
            if not isinstance(source, dict):
                continue
            prep = source.get("preprocessing", {}).get("sentiment", {})
            all_sentiment_texts.extend(prep.get("comments", []))
            all_sentiment_texts.extend(prep.get("posts", []))

        sentiment_texts_map = {"positive": [], "neutral": [], "negative": []}

        if all_sentiment_texts and detailed_sentiments:
            for idx, text in enumerate(all_sentiment_texts):
                if idx < len(detailed_sentiments):
                    label = detailed_sentiments[idx].get("label", "neutral")
                else:
                    break  # no more labels available
                if label in sentiment_texts_map and text:
                    sentiment_texts_map[label].append(text)

        # ── Strategy 2: fall back to detailed_sentiments.text ──────────────────────
        if not any(sentiment_texts_map.values()) and detailed_sentiments:
            for s in detailed_sentiments:
                label = s.get("label", "neutral")
                text = s.get("text", "")
                if text and label in sentiment_texts_map:
                    sentiment_texts_map[label].append(text)

        # ── Per-sentiment × per-POS wordclouds (9 total) ───────────────────────────
        for sentiment in ["positive", "neutral", "negative"]:
            texts = sentiment_texts_map.get(sentiment, [])
            if not texts:
                continue

            colormap = COLORMAPS.get(sentiment, "viridis")

            for pos_name, pos_tags in POS_GROUPS.items():
                filtered_text = _extract_pos_words(texts, pos_tags)
                img = _render_wordcloud(filtered_text, colormap)
                if img:
                    wordclouds[f"{sentiment}_{pos_name}"] = img

        # ── Overall wordcloud from pre-processed NOUN/PROPN/ADJ lemma texts ────────
        all_wordcloud_texts = []
        for source in processed_data:
            if isinstance(source, dict):
                prep = source.get("preprocessing", {}).get("wordcloud", {})
                all_wordcloud_texts.extend(prep.get("comments", []))
                all_wordcloud_texts.extend(prep.get("posts", []))

        if all_wordcloud_texts:
            overall_words = [w for w in " ".join(all_wordcloud_texts).split() if w not in custom_stops]
            img = _render_wordcloud(" ".join(overall_words), "viridis")
            if img:
                wordclouds["overall"] = img

    except Exception as e:
        pass  # silently return whatever wordclouds were completed before the error

    return wordclouds


def get_top_words_by_sentiment(
    sentiment_data: Dict,
    processed_data: List[Dict],
    topic: str = "",
    category_detail: str = "",
    top_n: int = 12,
) -> Dict[str, list]:
    """
    Compute the most frequent content words per sentiment bucket from the same
    preprocessed NOUN/PROPN/ADJ lemma texts used to render the wordclouds, with
    the same dynamic topic stopwords — but returns frequency counts instead of
    rendered images. Used by the Advisor Agent to generate a genuine wordcloud
    / word-frequency insight (instead of reusing ABSA aspect data as a proxy).

    Returns: {"overall": [(word, count), ...], "positive": [...], "neutral": [...], "negative": [...]}
    """
    from collections import Counter

    custom_stops = _build_topic_stopwords(topic, category_detail)
    detailed_sentiments = sentiment_data.get("detailed_sentiments", [])

    all_wordcloud_texts = []
    for source in processed_data:
        if isinstance(source, dict):
            prep = source.get("preprocessing", {}).get("wordcloud", {})
            all_wordcloud_texts.extend(prep.get("comments", []))
            all_wordcloud_texts.extend(prep.get("posts", []))

    buckets = {
        "overall": Counter(), "positive": Counter(),
        "neutral": Counter(), "negative": Counter(),
    }

    for idx, text in enumerate(all_wordcloud_texts):
        if not text:
            continue
        words = [w for w in text.split() if w not in custom_stops and len(w) > 2]
        if not words:
            continue
        buckets["overall"].update(words)
        label = (
            detailed_sentiments[idx].get("label", "neutral")
            if idx < len(detailed_sentiments) else "neutral"
        )
        if label in buckets:
            buckets[label].update(words)

    return {key: counter.most_common(top_n) for key, counter in buckets.items()}


def calculate_total_sentiment_coverage(aspect_analysis: Dict, total_sentiments: int) -> Dict:
    """
    Calculate what percentage of total comments are covered by aspects.
    
    Args:
        aspect_analysis: Dictionary of aspects with sentiment data
        total_sentiments: Total number of sentiments analyzed
    
    Returns:
        Dictionary with coverage statistics
    """
    
    comments_with_aspects = sum(data.get("total_mentions", 0) for data in aspect_analysis.values())
    comments_without_aspects = total_sentiments - comments_with_aspects
    
    return {
        "total_analyzed": total_sentiments,
        "with_aspects": comments_with_aspects,
        "without_aspects": comments_without_aspects,
        "coverage_percentage": (comments_with_aspects / total_sentiments * 100) if total_sentiments > 0 else 0
    }

def perform_aspect_level_sentiment_analysis(processed_data: List[Dict], 
                                           detailed_sentiments: List[Dict], 
                                           use_tech_aspects: bool = True) -> Dict:
    """
    Perform aspect-level sentiment analysis using tech-specific aspects.
    NOW TRACKS all comments including those without specific aspects.
    
    For technology scope: Extracts aspects like "camera", "battery", "price", etc.
    Falls back to noun-based extraction if tech aspects insufficient.
    
    Args:
        processed_data: Processed data with texts
        detailed_sentiments: Detailed sentiment predictions
        use_tech_aspects: Whether to prioritize tech aspects
    
    Returns:
        Dictionary with aspect analysis (including overall sentiment for unmatched comments)
    """
    
    try:
        from tools.processor_tools import TECH_ASPECTS
        import spacy
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                         check=False, capture_output=True)
            nlp = spacy.load("en_core_web_sm")
        
        aspects_sentiment = {}
        comments_with_aspects = set()  # Track which comments have aspects
        
        # Extract aspects from all texts
        all_texts = []
        for source in processed_data:
            if isinstance(source, dict):
                all_texts.extend(source.get("comments", []))
                all_texts.extend(source.get("posts", []))
        
        text_contents = []
        for t in all_texts:
            if isinstance(t, dict):
                text_contents.append(t.get("text", ""))
            else:
                text_contents.append(str(t) if t else "")
        
        # Match sentiments with texts and extract aspects
        for i, text in enumerate(text_contents[:len(detailed_sentiments)]):
            if not text or len(text.strip()) < 5:
                continue
            
            try:
                sentiment_label = detailed_sentiments[i].get("label", "neutral")
                doc = nlp(text)
                
                # First, look for tech aspects
                text_lower = text.lower()
                found_tech_aspects = False
                
                for tech_aspect in TECH_ASPECTS:
                    if tech_aspect in text_lower:
                        if tech_aspect not in aspects_sentiment:
                            aspects_sentiment[tech_aspect] = {
                                "positive": 0, "neutral": 0, "negative": 0, "total": 0
                            }
                        
                        aspects_sentiment[tech_aspect][sentiment_label] += 1
                        aspects_sentiment[tech_aspect]["total"] += 1
                        comments_with_aspects.add(i)
                        found_tech_aspects = True
                
                # If no tech aspects found, use noun chunks
                if not found_tech_aspects:
                    for chunk in doc.noun_chunks:
                        aspect = chunk.text.lower().strip()
                        
                        if (len(aspect) > 2 and aspect.isalpha() and 
                            aspect not in TECH_ASPECTS and
                            aspect not in ["the", "a", "an", "you", "they", "them", "what", "both", "this", "that", "it"]):
                            
                            if aspect not in aspects_sentiment:
                                aspects_sentiment[aspect] = {
                                    "positive": 0, "neutral": 0, "negative": 0, "total": 0
                                }
                            
                            aspects_sentiment[aspect][sentiment_label] += 1
                            aspects_sentiment[aspect]["total"] += 1
                            comments_with_aspects.add(i)
            
            except Exception as e:
                print(f"⚠️ Error processing text: {str(e)[:50]}")
                continue
        
        # Calculate percentages and filter
        aspect_analysis = {}
        for aspect, sentiment_counts in aspects_sentiment.items():
            total = sentiment_counts["total"]
            if total >= 2:  # Only aspects mentioned at least twice
                aspect_analysis[aspect] = {
                    "positive": {
                        "count": sentiment_counts["positive"],
                        "percentage": (sentiment_counts["positive"] / total * 100) if total > 0 else 0
                    },
                    "neutral": {
                        "count": sentiment_counts["neutral"],
                        "percentage": (sentiment_counts["neutral"] / total * 100) if total > 0 else 0
                    },
                    "negative": {
                        "count": sentiment_counts["negative"],
                        "percentage": (sentiment_counts["negative"] / total * 100) if total > 0 else 0
                    },
                    "total_mentions": total
                }
        
        # Add "Other/Unspecified" category for comments without specific aspects
        comments_without_aspects_list = [
            detailed_sentiments[i] for i in range(len(detailed_sentiments)) 
            if i not in comments_with_aspects
        ]
        
        if comments_without_aspects_list:
            other_pos = sum(1 for s in comments_without_aspects_list if s.get("label") == "positive")
            other_neu = sum(1 for s in comments_without_aspects_list if s.get("label") == "neutral")
            other_neg = sum(1 for s in comments_without_aspects_list if s.get("label") == "negative")
            other_total = len(comments_without_aspects_list)
            
            aspect_analysis["Other/Unspecified"] = {
                "positive": {
                    "count": other_pos,
                    "percentage": (other_pos / other_total * 100) if other_total > 0 else 0
                },
                "neutral": {
                    "count": other_neu,
                    "percentage": (other_neu / other_total * 100) if other_total > 0 else 0
                },
                "negative": {
                    "count": other_neg,
                    "percentage": (other_neg / other_total * 100) if other_total > 0 else 0
                },
                "total_mentions": other_total
            }
        
        # Prioritize tech aspects in output, but include "Other/Unspecified"
        sorted_aspects = sorted(
            aspect_analysis.items(),
            key=lambda x: (
                x[0] == "Other/Unspecified",  # Put Other/Unspecified last
                x[0] not in TECH_ASPECTS,     # Then prioritize tech aspects
                -x[1]["total_mentions"]        # Then by total mentions
            )
        )[:16]  # Show top 16 (includes Other/Unspecified)
        
        return dict(sorted_aspects)
    
    except Exception as e:
        print(f"⚠️ Could not perform aspect-level analysis: {str(e)}")
        return {}


def generate_aspect_sentiment_chart(aspect_analysis: Dict) -> go.Figure:
    """
    Generate a grouped bar chart showing sentiment for each aspect.

    "Others" (HDBSCAN noise cluster −1) is always shown last with muted bar
    colours so it is visually distinct from named topic clusters.  A footnote
    annotation is added explaining what it represents.

    Tick angle adapts dynamically so labels are readable at any aspect count.
    """

    if not aspect_analysis:
        return None

    try:
        # Named aspects (top 9) + Others always at the end
        others_data   = aspect_analysis.get("Others")
        named_aspects = [a for a in aspect_analysis if a != "Others"][:9]
        aspects       = named_aspects + (["Others"] if others_data else [])

        if not aspects:
            return None

        has_others = "Others" in aspects

        # Per-bar colour lists: muted tones for the Others bar
        _muted = {"#2ecc71": "#a8d5b5", "#95a5a6": "#cccccc", "#e74c3c": "#e8a8a3"}

        def _colors(base: str) -> list:
            return [_muted[base] if a == "Others" else base for a in aspects]

        positive_counts = [aspect_analysis[a]["positive"]["count"] for a in aspects]
        neutral_counts  = [aspect_analysis[a]["neutral"]["count"]  for a in aspects]
        negative_counts = [aspect_analysis[a]["negative"]["count"] for a in aspects]

        fig = go.Figure(data=[
            go.Bar(name="Positive", x=aspects, y=positive_counts,
                   marker_color=_colors("#2ecc71")),
            go.Bar(name="Neutral",  x=aspects, y=neutral_counts,
                   marker_color=_colors("#95a5a6")),
            go.Bar(name="Negative", x=aspects, y=negative_counts,
                   marker_color=_colors("#e74c3c")),
        ])

        # Dynamic tick angle
        n             = len(aspects)
        max_label_len = max(len(a) for a in aspects)
        if n <= 4 and max_label_len <= 15:
            tick_angle = 0
        elif n <= 7 and max_label_len <= 20:
            tick_angle = -30
        else:
            tick_angle = -45

        chart_height  = 420 if tick_angle == 0 else (460 if tick_angle == -30 else 500)
        bottom_margin = 80 if has_others else 40

        fig.update_layout(
            title="Aspect-Level Sentiment Analysis",
            xaxis_title="Aspect",
            yaxis_title="Count",
            barmode="group",
            height=chart_height + (30 if has_others else 0),
            xaxis_tickangle=tick_angle,
            margin=dict(b=bottom_margin),
        )

        if has_others:
            fig.add_annotation(
                text="* Others: comments not assigned to any specific topic cluster (HDBSCAN noise, cluster −1)",
                xref="paper", yref="paper",
                x=0, y=-0.22,
                showarrow=False,
                font=dict(size=11, color="gray"),
                align="left",
            )

        return fig

    except Exception as e:
        print(f"Error in generate_aspect_sentiment_chart: {str(e)}")
        return None