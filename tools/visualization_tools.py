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


def generate_timeline_chart(sentiments: List[Dict], use_timestamps: bool = True) -> go.Figure:
    """
    Generate timeline chart with PROPER AGGREGATION by date.
    
    Groups sentiments by date and creates daily trend lines.
    Ensures readable visualization with aggregated daily counts.
    """
    
    if not sentiments:
        return None
    
    try:
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Step 1: Extract dates from sentiments
        sentiment_records = []
        
        for sentiment in sentiments:
            try:
                # Extract date
                timestamp = sentiment.get("created_at")
                if timestamp:
                    if isinstance(timestamp, (int, float)):
                        date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                    else:
                        date_str = str(timestamp)[:10]
                        try:
                            datetime.strptime(date_str, "%Y-%m-%d")
                            date = date_str
                        except:
                            date = None
                else:
                    date = None
                
                # If no date, create synthetic date
                if not date:
                    # Use position as synthetic date
                    base_date = datetime.now()
                    days_back = len(sentiment_records)
                    synthetic_date = base_date - timedelta(days=days_back)
                    date = synthetic_date.strftime("%Y-%m-%d")
                
                # Extract sentiment label
                label = sentiment.get("label", "neutral")
                
                sentiment_records.append({
                    "date": date,
                    "sentiment": label
                })
            
            except Exception as e:
                print(f"Error parsing sentiment: {str(e)[:50]}")
                continue
        
        if not sentiment_records:
            return None
        
        # Step 2: Create DataFrame for aggregation
        df = pd.DataFrame(sentiment_records)
        
        # Step 3: Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Step 4: Group by date and sentiment, count occurrences
        grouped = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        # Ensure all sentiment columns exist
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment not in grouped.columns:
                grouped[sentiment] = 0
        
        # Step 5: Sort by date and reset index
        grouped = grouped.sort_index()
        grouped = grouped[['positive', 'neutral', 'negative']]  # Ensure correct column order
        
        # Step 6: Create plotly figure with aggregated data
        dates = grouped.index.strftime("%Y-%m-%d").tolist()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=grouped['positive'].astype(int),
            mode='lines+markers',
            name='Positive',
            line=dict(color="#2ecc71", width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=grouped['neutral'].astype(int),
            mode='lines+markers',
            name='Neutral',
            line=dict(color="#95a5a6", width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=grouped['negative'].astype(int),
            mode='lines+markers',
            name='Negative',
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Sentiment Timeline (by Date) - Aggregated Daily Counts",
            xaxis_title="Date (YYYY-MM-DD)",
            yaxis_title="Count per Day",
            height=400,
            hovermode='x unified',
            xaxis_tickangle=-45,
            yaxis=dict(tickformat="d")
        )
        
        return fig
    
    except Exception as e:
        print(f"Error in generate_timeline_chart: {str(e)}")
        return None


def generate_wordcloud_by_sentiment(sentiment_data: Dict, processed_data: List[Dict], topic: str = "") -> Dict[str, bytes]:
    """
    Generate wordclouds with topic-aware custom stopwords filtering.

    Args:
        sentiment_data: Sentiment analysis results
        processed_data: Processed data with preprocessing info
        topic: Topic to filter from wordcloud

    Returns:
        Dictionary with wordcloud images
    """

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠️ WordCloud library not installed")
        return {}

    wordclouds = {}

    try:
        # Create custom stopwords based on topic
        custom_stops = set()
        if topic:
            # Use basic topic words as stopwords
            topic_words = set(topic.lower().split())
            custom_stops = topic_words
            print(f"Using {len(custom_stops)} topic words as stopwords")

        # Extract wordcloud texts from processed data
        all_wordcloud_texts = []
        for source in processed_data:
            if isinstance(source, dict):
                preprocessing = source.get("preprocessing", {})
                wordcloud_data = preprocessing.get("wordcloud", {})
                all_wordcloud_texts.extend(wordcloud_data.get("comments", []))
                all_wordcloud_texts.extend(wordcloud_data.get("posts", []))

        if not all_wordcloud_texts:
            print("⚠️ No wordcloud texts available")
            return {}

        # Extract sentiment-specific texts
        detailed_sentiments = sentiment_data.get("detailed_sentiments", [])
        if not detailed_sentiments:
            print("⚠️ No sentiment details available")
            return {}

        # Generate per-sentiment wordclouds
        for sentiment in ["positive", "neutral", "negative"]:
            sentiment_texts = [
                s.get("text", "") for s in detailed_sentiments 
                if s.get("label") == sentiment and s.get("text")
            ]

            if not sentiment_texts or len(" ".join(sentiment_texts)) < 20:
                continue

            text = " ".join(sentiment_texts)

            try:
                # Remove custom stopwords from text
                words = text.split()
                filtered_words = [w for w in words if w not in custom_stops]
                filtered_text = " ".join(filtered_words)

                if len(filtered_text.strip()) < 20:
                    continue

                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    colormap={"positive": "Greens", "neutral": "Blues", "negative": "Reds"}.get(sentiment, "viridis"),
                    prefer_horizontal=0.7,
                    min_font_size=10,
                    max_words=100
                ).generate(filtered_text)

                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')

                import io
                img_bytes = io.BytesIO()
                plt.savefig(img_bytes, format='png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                img_bytes.seek(0)

                wordclouds[sentiment] = img_bytes.getvalue()
                print(f"✅ Generated {sentiment} wordcloud")

            except Exception as e:
                print(f"⚠️ Error generating {sentiment} wordcloud: {str(e)[:80]}")
                continue

        # Generate overall wordcloud
        try:
            overall_text = " ".join(all_wordcloud_texts)
            if len(overall_text.strip()) >= 20:
                words = overall_text.split()
                filtered_words = [w for w in words if w not in custom_stops]
                filtered_text = " ".join(filtered_words)

                if len(filtered_text.strip()) >= 20:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        prefer_horizontal=0.7,
                        min_font_size=10,
                        max_words=100
                    ).generate(filtered_text)

                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(111)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')

                    import io
                    img_bytes = io.BytesIO()
                    plt.savefig(img_bytes, format='png', bbox_inches='tight', dpi=100)
                    plt.close(fig)
                    img_bytes.seek(0)

                    wordclouds["overall"] = img_bytes.getvalue()
                    print(f"✅ Generated overall wordcloud")

        except Exception as e:
            print(f"⚠️ Error generating overall wordcloud: {str(e)[:80]}")

    except Exception as e:
        print(f"Error in generate_wordcloud_by_sentiment: {str(e)}")

    return wordclouds

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
    Uses plotly only - no wordcloud dependency.
    """
    
    if not aspect_analysis:
        return None
    
    try:
        aspects = list(aspect_analysis.keys())[:10]
        
        if not aspects:
            return None
        
        positive_counts = [aspect_analysis[a]["positive"]["count"] for a in aspects]
        neutral_counts = [aspect_analysis[a]["neutral"]["count"] for a in aspects]
        negative_counts = [aspect_analysis[a]["negative"]["count"] for a in aspects]
        
        fig = go.Figure(data=[
            go.Bar(name='Positive', x=aspects, y=positive_counts, marker_color='#2ecc71'),
            go.Bar(name='Neutral', x=aspects, y=neutral_counts, marker_color='#95a5a6'),
            go.Bar(name='Negative', x=aspects, y=negative_counts, marker_color='#e74c3c')
        ])
        
        fig.update_layout(
            title="Aspect-Level Sentiment Analysis",
            xaxis_title="Aspect",
            yaxis_title="Count",
            barmode='group',
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    except Exception as e:
        print(f"Error in generate_aspect_sentiment_chart: {str(e)}")
        return None