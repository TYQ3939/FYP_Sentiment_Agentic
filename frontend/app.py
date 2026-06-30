# Windows asyncio fix MUST be first - import asyncio before using it
import sys
import os
import asyncio

# Setup project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set event loop policy for Windows AFTER importing asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Now import everything else
import streamlit as st
import requests
import time
from datetime import datetime

# ========== CONFIGURATION ==========

API_BASE_URL = "http://127.0.0.1:8000"


@st.cache_data(show_spinner=False)
def _cached_generate_wordclouds(job_id, _analysis_data, _processed_data, topic, category_detail=""):
    """
    Cache wordcloud generation per job so changing the noun/verb/adj filter
    (a widget rerun) doesn't regenerate all 10 wordcloud images from scratch.
    Args prefixed with "_" are excluded from Streamlit's cache-key hashing.
    """
    from tools.visualization_tools import generate_wordcloud_by_sentiment
    return generate_wordcloud_by_sentiment(
        _analysis_data, _processed_data, topic=topic, category_detail=category_detail
    )

# ========== PAGE CONFIG ==========

st.set_page_config(
    page_title="AI Sentiment Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Multi-Agent Sentiment Analysis")
st.write("Analyze social media sentiment using a team of specialized AI agents.")
st.write("💡 **Note:** Backend API must be running on `http://localhost:8000`")

# ========== INITIALIZE SESSION STATE ==========

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "main"
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "current_job_id" not in st.session_state:
    st.session_state.current_job_id = None

# ========== SIDEBAR ==========

with st.sidebar:
    st.header("📋 Session History")
    
    if st.session_state.session_history:
        st.subheader("Previous Analyses")
        for i, historical_topic in enumerate(st.session_state.session_history):
            if st.button(f"🔍 {historical_topic}", key=f"history_{i}"):
                st.session_state.current_topic = historical_topic
                st.session_state.current_tab = "results"
                st.rerun()
    else:
        st.write("No previous analyses yet.")
    
    st.divider()
    
    # API Status Check
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except requests.exceptions.ConnectionError:
        st.error("❌ API Offline\n\nStart backend with:\n```\npython run_backend.py\n```")
    except Exception as e:
        st.error(f"❌ API Error: {str(e)[:50]}")
    
    st.divider()
    
    if st.button("🗑️ Clear History", width='stretch'):
        st.session_state.session_history = []
        st.rerun()

# ========== MAIN TAB ==========

if st.session_state.current_tab == "main":
    st.header("Run Sentiment Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "Enter Topic:",
            value=st.session_state.current_topic or "",
            placeholder="e.g., iPhone 17 Pro Max"
        )
    
    with col2:
        start_button = st.button("Run Analysis", type="primary", width='stretch')
    
    # Optional: Custom subreddits
    with st.expander("🔧 Advanced Options"):
        subreddits_input = st.text_input(
            "Custom subreddits (comma-separated):",
            placeholder="e.g., iphone, apple, technology"
        )
        custom_subreddits = [s.strip() for s in subreddits_input.split(",")] if subreddits_input else None
    
    if start_button and topic:
        st.session_state.current_topic = topic
        
        # Add to history
        if topic not in st.session_state.session_history:
            st.session_state.session_history.insert(0, topic)
        
        try:
            # Call backend API
            with st.spinner("🚀 Submitting job to backend..."):
                response = requests.post(
                    f"{API_BASE_URL}/scrape/start",
                    json={
                        "topic": topic,
                        "subreddits": custom_subreddits
                    },
                    timeout=10
                )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data["job_id"]
                st.session_state.current_job_id = job_id

                st.success("✅ Analysis started! Redirecting to progress page...")
                
                # Store start time
                st.session_state.start_time = time.time()
                st.session_state.current_tab = "monitoring"
                st.rerun()
            else:
                st.error(f"❌ Failed to start job: {response.json().get('detail', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ **Cannot connect to backend API**\n\nMake sure to start the backend:\n```bash\npython run_backend.py\n```")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    elif not topic and start_button:
        st.warning("Please enter a topic first.")

# ========== MONITORING TAB ==========

elif st.session_state.current_tab == "monitoring":
    if st.session_state.current_job_id:
        topic_display = st.session_state.current_topic or "your topic"
        st.header(f"Analyzing: {topic_display}")

        job_id = st.session_state.current_job_id

        if st.button("← Cancel / Back to Input"):
            st.session_state.current_tab = "main"
            st.rerun()

        st.divider()

        job_status = None
        status_placeholder = st.empty()

        try:
            with status_placeholder.container():
                st.info("Checking progress...")
            response = requests.get(f"{API_BASE_URL}/scrape/status/{job_id}", timeout=20)
            status_placeholder.empty()
            if response.status_code == 200:
                job_status = response.json()
        except requests.exceptions.Timeout:
            status_placeholder.warning("Connection slow — retrying in 5 seconds...")
            time.sleep(5)
            st.rerun()
        except requests.exceptions.ConnectionError:
            status_placeholder.error("Cannot connect to the backend. Make sure it is running.")
        except Exception as e:
            status_placeholder.error(f"Error fetching status: {str(e)}")

        if job_status:
            progress = job_status.get('progress', 0)
            status   = job_status.get('status', 'pending')

            # ── Progress bar ──────────────────────────────────────────────────
            st.progress(progress / 100)
            st.caption(f"{progress}% complete")

            st.divider()

            # ── Status-specific messaging ─────────────────────────────────────
            if status == "completed":
                st.success("Analysis complete!")
                if st.button("View Results", type="primary"):
                    st.session_state.current_tab = "results"
                    st.session_state.analysis_complete = True
                    st.rerun()

            elif status == "error":
                st.error(f"Something went wrong: {job_status.get('error', 'Unknown error')}")
                if st.button("← Try Again"):
                    st.session_state.current_tab = "main"
                    st.rerun()

            else:
                elapsed = time.time() - st.session_state.get("start_time", time.time())

                started_at = ""
                try:
                    created_time = datetime.fromisoformat(job_status['created_at'])
                    started_at   = f"Started at {created_time.strftime('%H:%M:%S')}."
                except Exception:
                    pass

                # ── Taking too long? Show a soft warning after 12 minutes ──────
                if elapsed > 720:
                    st.warning(
                        f"This is taking longer than expected (over 12 minutes). "
                        f"The Reddit API may be slow right now, or **{topic_display}** "
                        f"may not have enough recent discussion. "
                        f"You can wait a little longer, or cancel and try a broader topic."
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Wait & Refresh"):
                            st.rerun()
                    with col2:
                        if st.button("Cancel & Try Again", type="primary"):
                            st.session_state.current_tab = "main"
                            st.rerun()
                else:
                    st.info(
                        f"The AI agents are working on **{topic_display}**. "
                        f"This typically takes 3–8 minutes depending on the amount of data. "
                        f"{started_at}"
                    )
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("Refresh Now"):
                            st.rerun()
                    with col2:
                        st.caption("Auto-refreshes every 5 seconds.")

                time.sleep(5)
                st.rerun()
    else:
        st.warning("No analysis in progress. Go back to start a new one.")
        if st.button("← Back"):
            st.session_state.current_tab = "main"
            st.rerun()

# ========== RESULTS TAB (UPDATED) ==========

elif st.session_state.current_tab == "results":
    if st.session_state.current_job_id:
        job_id = st.session_state.current_job_id

        # Fetch final results with extended timeout (results payload is large)
        # Use the dedicated /scrape/results/ endpoint which returns full data
        try:
            with st.spinner("📥 Loading results from server (this may take 10-30 seconds)..."):
                response = requests.get(f"{API_BASE_URL}/scrape/results/{job_id}", timeout=30)

            if response.status_code == 200:
                job_status = response.json()

                if job_status['status'] == "completed" and job_status['results']:
                    results = job_status['results']
                    
                    st.header(f"📈 Analysis Results: {st.session_state.current_topic}")
                    
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("🏠 New Analysis", type="primary"):
                            st.session_state.current_tab = "main"
                            st.session_state.analysis_complete = False
                            st.session_state.current_job_id = None
                            st.rerun()
                    
                    with col2:
                        if st.button("📥 Export Results"):
                            st.info("Export functionality coming soon!")
                    
                    st.divider()
                    
                    # Get analysis results
                    analyst_results       = results.get('analyst', {})
                    advisor_results       = results.get('advisor', {})
                    visualization_results = results.get('visualization', {})

                    analysis_data      = analyst_results.get('analysis', {})
                    visualization_data = visualization_results.get('visualization_data', {})
                    advisor_insights   = advisor_results.get('advisor_insights', {})
                    
                    # ── Insight text styling helper ───────────────────────
                    def _style_insight(text: str) -> str:
                        """
                        Post-process advisor insight text for richer Streamlit rendering.
                        Converts **bold** markdown to HTML <strong> and adds colour
                        spans for positive (green), negative (red) and neutral (grey)
                        sentiment keywords.  Requires unsafe_allow_html=True.
                        """
                        import re
                        # Convert **bold** → <strong>bold</strong> (reliable in HTML mode)
                        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                        # Colour positive-sentiment keywords green
                        text = re.sub(
                            r'\b(positive|positively|strength|strengths)\b',
                            r'<span style="color:#27ae60;font-weight:600">\1</span>',
                            text, flags=re.IGNORECASE
                        )
                        # Colour negative-sentiment keywords red
                        text = re.sub(
                            r'\b(negative|negatively|weakness|weaknesses|concern|concerning)\b',
                            r'<span style="color:#e74c3c;font-weight:600">\1</span>',
                            text, flags=re.IGNORECASE
                        )
                        # Colour neutral/mixed keywords grey
                        text = re.sub(
                            r'\b(neutral|mixed)\b',
                            r'<span style="color:#7f8c8d;font-weight:600">\1</span>',
                            text, flags=re.IGNORECASE
                        )
                        return text

                    # Display results in tabs
                    tabs = st.tabs([
                        "📊 Summary",
                        "📈 Timeline",
                        "☁️ Wordcloud",
                        "🎯 Aspect Analysis",
                        "💡 Recommendations"
                    ])
                    
                    # ============ TAB 1: SUMMARY ============
                    with tabs[0]:
                        st.subheader("Analysis Summary")

                        # Topic — displayed as full-width text so long names are never cut off
                        topic_label = visualization_data.get('topic', 'N/A')
                        st.markdown(f"**Topic:** {topic_label}")

                        st.divider()

                        # Key metrics — 3 columns (Posts removed: always 0)
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Total Comments",
                                visualization_data.get('total_comments', 0)
                            )

                        with col2:
                            st.metric(
                                "Data Sources",
                                f"{len(visualization_data.get('subreddits', []))} subreddits"
                            )

                        with col3:
                            sentiment = visualization_data.get('overall_sentiment', 'neutral').upper()
                            emoji = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}.get(sentiment, "❓")
                            st.metric(
                                "Overall Sentiment",
                                f"{emoji} {sentiment}"
                            )
                        
                        st.divider()
                        
                        # Data sources
                        st.subheader("Data Sources")
                        subreddits = visualization_data.get('subreddits', [])
                        st.write(f"**Analyzed {len(subreddits)} subreddit(s):**")
                        for subreddit in subreddits:
                            st.write(f"  • r/{subreddit}")
                        
                        st.divider()
                        
                        # Sentiment distribution with chart
                        st.subheader("Sentiment Distribution")
                        
                        distribution = analysis_data.get('sentiment_distribution', {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pos_data = distribution.get('positive', {})
                            st.metric(
                                "😊 Positive",
                                f"{pos_data.get('count', 0)}",
                                f"{pos_data.get('percentage', 0):.1f}%"
                            )
                        
                        with col2:
                            neu_data = distribution.get('neutral', {})
                            st.metric(
                                "😐 Neutral",
                                f"{neu_data.get('count', 0)}",
                                f"{neu_data.get('percentage', 0):.1f}%"
                            )
                        
                        with col3:
                            neg_data = distribution.get('negative', {})
                            st.metric(
                                "😞 Negative",
                                f"{neg_data.get('count', 0)}",
                                f"{neg_data.get('percentage', 0):.1f}%"
                            )
                        
                        st.metric("Confidence", f"{analysis_data.get('confidence', 0):.2%}")
                        
                        # Display sentiment chart
                        try:
                            import plotly.graph_objects as go
                            
                            labels = list(distribution.keys())
                            sizes = [distribution[label]["count"] for label in labels]
                            colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=labels,
                                values=sizes,
                                marker=dict(colors=[colors.get(label, "#999999") for label in labels]),
                                textposition="inside",
                                textinfo="label+percent"
                            )])
                            
                            fig.update_layout(
                                title="Sentiment Distribution Chart",
                                height=400
                            )
                            
                            st.plotly_chart(fig, width='stretch')
                        
                        except Exception as e:
                            st.error(f"Could not display chart: {str(e)}")

                        # ── Advisor insight ──────────────────────────────────
                        summary_insight = advisor_insights.get('summary', '')
                        if summary_insight:
                            st.divider()
                            with st.expander("AI Advisor Insight", expanded=True):
                                st.markdown(_style_insight(summary_insight), unsafe_allow_html=True)

                    # ============ TAB 2: TIMELINE ============
                    with tabs[1]:
                        st.subheader("Sentiment Timeline")
                        st.info("📈 Shows how sentiment evolved across time periods")
                        
                        try:
                            # Import directly without path issues
                            import sys
                            import os
                            
                            # Get the project root directory
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            if project_root not in sys.path:
                                sys.path.insert(0, project_root)
                            
                            from tools.visualization_tools import generate_timeline_chart
                            
                            detailed_sentiments = analysis_data.get('detailed_sentiments', [])
                            if detailed_sentiments:
                                fig = generate_timeline_chart(detailed_sentiments)
                                
                                if fig:
                                    st.plotly_chart(fig, width='stretch')
                                else:
                                    st.warning("Could not generate timeline chart")
                            else:
                                st.info("No sentiment data available for timeline")
                        
                        except ImportError as e:
                            st.error(f"Import error: {str(e)}")
                            st.info("Try running from project root: `streamlit run frontend/app.py`")
                        except Exception as e:
                            st.error(f"Error generating timeline: {str(e)}")

                        # ── Advisor insight ──────────────────────────────────
                        timeline_insight = advisor_insights.get('timeline', '')
                        if timeline_insight:
                            st.divider()
                            with st.expander("AI Advisor Insight", expanded=True):
                                st.markdown(_style_insight(timeline_insight), unsafe_allow_html=True)

                    # ============ TAB 3: WORDCLOUD ============
                    with tabs[2]:
                        st.subheader("Word Frequency Analysis")
                        st.info("☁️ Shows the most frequent words by sentiment and word type (nouns, verbs, adjectives)")

                        try:
                            processed_data  = results.get('state', {}).get('processed_data', [])
                            topic           = visualization_data.get('topic', '')
                            category_detail = visualization_data.get('category_detail', '')

                            if processed_data:
                                with st.spinner("Generating wordclouds..."):
                                    wordclouds = _cached_generate_wordclouds(
                                        job_id, analysis_data, processed_data, topic, category_detail
                                    )

                                if wordclouds:
                                    # Overall wordcloud at the top
                                    if "overall" in wordclouds:
                                        st.subheader("Overall Wordcloud")
                                        st.image(wordclouds["overall"], use_container_width=True)

                                    st.divider()

                                    # POS filter
                                    st.subheader("Wordcloud by Sentiment & Word Type")
                                    pos_filter = st.radio(
                                        "Filter by word type:",
                                        ["Nouns", "Verbs", "Adjectives"],
                                        horizontal=True,
                                        key="wordcloud_pos_filter"
                                    )
                                    pos_key = {"Nouns": "noun", "Verbs": "verb", "Adjectives": "adj"}[pos_filter]

                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        key = f"positive_{pos_key}"
                                        st.subheader("😊 Positive")
                                        if key in wordclouds:
                                            st.image(wordclouds[key], use_container_width=True)
                                        else:
                                            st.info("Not enough data")

                                    with col2:
                                        key = f"neutral_{pos_key}"
                                        st.subheader("😐 Neutral")
                                        if key in wordclouds:
                                            st.image(wordclouds[key], use_container_width=True)
                                        else:
                                            st.info("Not enough data")

                                    with col3:
                                        key = f"negative_{pos_key}"
                                        st.subheader("😞 Negative")
                                        if key in wordclouds:
                                            st.image(wordclouds[key], use_container_width=True)
                                        else:
                                            st.info("Not enough data")
                                else:
                                    st.info("Could not generate wordclouds")
                            else:
                                st.info("No processed data available for wordclouds")

                        except Exception as e:
                            st.error(f"Error generating wordclouds: {str(e)}")

                        # ── Advisor insight ──────────────────────────────────
                        wc_insight = advisor_insights.get('wordcloud', '')
                        if wc_insight:
                            st.divider()
                            with st.expander("AI Advisor Insight", expanded=True):
                                st.markdown(_style_insight(wc_insight), unsafe_allow_html=True)

                    # ============ TAB 4: ASPECT ANALYSIS ============
                    with tabs[3]:
                        st.subheader("Aspect-Level Sentiment Analysis")
                        st.info("🎯 Shows sentiment for specific aspects/topics mentioned")
                        
                        try:
                            import sys
                            import os
                            
                            # Get the project root directory
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            if project_root not in sys.path:
                                sys.path.insert(0, project_root)
                            
                            from tools.visualization_tools import generate_aspect_sentiment_chart
                            
                            aspect_analysis = visualization_data.get('aspect_analysis', {})
                            
                            if aspect_analysis:
                                # Display aspect chart
                                fig = generate_aspect_sentiment_chart(aspect_analysis)
                                if fig:
                                    st.plotly_chart(fig, width='stretch')
                                
                                # Display aspect details table — top 9 named + Others always at end
                                st.subheader("Top Aspects")

                                others_row  = aspect_analysis.get("Others")
                                named_items = [(a, d) for a, d in aspect_analysis.items()
                                               if a != "Others"][:9]
                                table_items = named_items + ([("Others", others_row)]
                                                             if others_row else [])

                                aspect_list = []
                                for aspect, data in table_items:
                                    aspect_list.append({
                                        "Aspect"  : aspect.title(),
                                        "Positive": f"{data['positive']['percentage']:.1f}%",
                                        "Neutral" : f"{data['neutral']['percentage']:.1f}%",
                                        "Negative": f"{data['negative']['percentage']:.1f}%",
                                        "Mentions": data['total_mentions']
                                    })

                                st.dataframe(aspect_list, width='stretch')
                            else:
                                st.info("No aspect analysis available")

                        except ImportError as e:
                            st.error(f"Import error: {str(e)}")
                            st.info("Try running from project root: `streamlit run frontend/app.py`")
                        except Exception as e:
                            st.error(f"Error displaying aspect analysis: {str(e)}")

                        # ── Advisor ABSA insight ──────────────────────────────
                        absa_insight = advisor_insights.get('absa_insight', '')
                        if absa_insight:
                            st.divider()
                            with st.expander("AI Advisor Insight", expanded=True):
                                st.markdown(_style_insight(absa_insight), unsafe_allow_html=True)

                    # ============ TAB 5: RECOMMENDATIONS & CHAT ============
                    with tabs[4]:
                        st.subheader("AI Advisor — Full Analysis")

                        try:
                            # ── Section-by-section insights overview ─────────
                            if advisor_insights:
                                with st.expander("Overall Sentiment Insight", expanded=True):
                                    st.markdown(_style_insight(advisor_insights.get('summary', '_Not available_')), unsafe_allow_html=True)

                                with st.expander("Sentiment Timeline Insight", expanded=False):
                                    st.markdown(_style_insight(advisor_insights.get('timeline', '_Not available_')), unsafe_allow_html=True)

                                with st.expander("Discussion Theme Insight", expanded=False):
                                    st.markdown(_style_insight(advisor_insights.get('wordcloud', '_Not available_')), unsafe_allow_html=True)
                            else:
                                st.info("Advisor insights not available for this job.")

                            st.divider()

                            # ── ABSA-based strategic recommendations ─────────
                            st.subheader("Strategic Recommendations")
                            recommendations = advisor_results.get('recommendations', [])
                            if recommendations:
                                for rec in recommendations:
                                    st.markdown(_style_insight(rec), unsafe_allow_html=True)
                            else:
                                st.info("No strategic recommendations available.")

                            st.divider()

                            # ── Interactive chat ─────────────────────────────
                            st.subheader("Ask a Question")

                            if "chat_history" not in st.session_state:
                                st.session_state.chat_history = []

                            # Render conversation history
                            for q, a in st.session_state.chat_history:
                                with st.chat_message("user"):
                                    st.write(q)
                                with st.chat_message("assistant"):
                                    st.write(a)

                            # Chat input
                            user_question = st.chat_input(
                                "Ask anything about the analysis...",
                                key=f"chat_input_{job_id}"
                            )

                            # ── Suggested questions (shown under the chat input) ──
                            suggested_questions = advisor_results.get('suggested_questions', [])
                            if suggested_questions:
                                st.caption("Or try one of these:")
                                cols = st.columns(2)
                                for idx, question in enumerate(suggested_questions):
                                    with cols[idx % 2]:
                                        if st.button(question, width='stretch', key=f"suggested_{idx}"):
                                            if "chat_history" not in st.session_state:
                                                st.session_state.chat_history = []
                                            with st.spinner("Thinking..."):
                                                try:
                                                    from agents.advisor_agent import AdvisorAgent
                                                    advisor = AdvisorAgent()
                                                    answer = advisor.answer_question(question)
                                                    st.session_state.chat_history.append((question, answer))
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Error: {str(e)[:200]}")

                            if user_question:
                                with st.chat_message("user"):
                                    st.write(user_question)
                                with st.chat_message("assistant"):
                                    with st.spinner("Thinking..."):
                                        try:
                                            project_root = os.path.dirname(
                                                os.path.dirname(os.path.abspath(__file__))
                                            )
                                            if project_root not in sys.path:
                                                sys.path.insert(0, project_root)

                                            from agents.advisor_agent import AdvisorAgent
                                            advisor = AdvisorAgent()
                                            answer  = advisor.answer_question(user_question)
                                            st.session_state.chat_history.append((user_question, answer))
                                            st.write(answer)
                                        except Exception as e:
                                            err = f"Error answering question: {str(e)[:200]}"
                                            st.error(err)
                                            st.session_state.chat_history.append((user_question, err))

                        except Exception as e:
                            st.error(f"Error in Recommendations tab: {str(e)}")
                
                else:
                    st.error("Results not available yet or job failed.")

        except requests.exceptions.Timeout:
            st.error("⏱️ **Request Timeout**: Server took too long to respond. Results are likely still being processed.")
            st.info("💡 Try these solutions:\n1. Wait a few more seconds\n2. Click the back button and monitor the job\n3. Refresh the page manually")
        except requests.exceptions.ConnectionError:
            st.error("❌ **Connection Error**: Cannot reach the backend API.\n\nMake sure the backend is running:\n```bash\npython run_backend.py\n```")
        except Exception as e:
            st.error(f"❌ Error fetching results: {str(e)}")
            st.info("If this persists, try:\n1. Going back to monitor the job\n2. Checking the backend logs for errors")
    else:
        st.warning("No results available. Go to the main tab to start a new analysis.")