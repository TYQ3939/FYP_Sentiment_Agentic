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
import re as _re
from datetime import datetime


def _style_insight(text: str) -> str:
    """Convert **bold** markers and colour pos/neg keywords for st.markdown HTML."""
    # **bold** → <strong>
    text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # colour positive/negative words
    for word in ("positive", "positively", "praised"):
        text = _re.sub(rf"\b({word})\b", r"<span style='color:#2ecc71'>\1</span>", text, flags=_re.IGNORECASE)
    for word in ("negative", "negatively", "criticized", "complaints", "unhappy"):
        text = _re.sub(rf"\b({word})\b", r"<span style='color:#e74c3c'>\1</span>", text, flags=_re.IGNORECASE)
    return text


# ========== CONFIGURATION ==========

API_BASE_URL = "http://127.0.0.1:8000"

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
                
                st.success(f"✅ Job started: `{job_id}`")
                st.info("Polling for results... This may take several minutes.")
                
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
        st.header("📊 Job Monitoring")
        
        job_id = st.session_state.current_job_id
        
        # Back button
        if st.button("← Back to Input"):
            st.session_state.current_tab = "main"
            st.rerun()
        
        st.divider()

        # Auto-refresh status with visual feedback
        job_status = None
        status_placeholder = st.empty()

        try:
            with status_placeholder.container():
                st.info("🔄 Checking job status...")

            response = requests.get(f"{API_BASE_URL}/scrape/status/{job_id}", timeout=20)

            status_placeholder.empty()  # Clear the loading message

            if response.status_code == 200:
                job_status = response.json()
        except requests.exceptions.Timeout:
            status_placeholder.warning("⏱️ Request timeout - retrying automatically in 5 seconds...")
            time.sleep(5)
            st.rerun()
        except requests.exceptions.ConnectionError:
            status_placeholder.error("❌ Cannot connect to backend API. Make sure it's running:\n```bash\npython run_backend.py\n```")
        except Exception as e:
            status_placeholder.error(f"Error fetching status: {str(e)}")
        
        if job_status:
            _mon_topic = st.session_state.get("current_topic", "your topic")
            st.markdown(f"### Analyzing: {_mon_topic}")

            # Progress bar
            _pct = job_status['progress']
            st.progress(_pct / 100)
            st.caption(f"{_pct}% complete")

            # Status badge (no Job ID shown to users)
            _jstatus = job_status['status']
            if _jstatus == "completed":
                st.success("Analysis complete!")
            elif _jstatus == "error":
                st.error(f"Something went wrong: {job_status.get('error', 'Unknown error')}")
            elif _jstatus == "running":
                _stage_msg = {
                    10: "Collecting data...",
                    30: "Preprocessing text...",
                    50: "Running sentiment model...",
                    70: "Analyzing aspects...",
                    85: "Generating recommendations...",
                    95: "Finishing up...",
                }.get(
                    max((k for k in [10, 30, 50, 70, 85, 95] if k <= _pct), default=10),
                    "Processing...",
                )
                st.info(_stage_msg)
            else:
                st.warning("Queued — starting soon...")

            st.caption(f"Started at {datetime.fromisoformat(job_status['created_at']).strftime('%H:%M:%S')}")
            st.divider()

            if _jstatus == "completed":
                if st.button("View Results", type="primary"):
                    st.session_state.current_tab = "results"
                    st.session_state.analysis_complete = True
                    st.rerun()
            elif _jstatus != "error":
                if st.button("Refresh Now"):
                    st.rerun()
                time.sleep(5)
                st.rerun()
    else:
        st.warning("No job in progress. Go to the main tab to start a new analysis.")

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
                    analyst_results = results.get('analyst', {})
                    advisor_results = results.get('advisor', {})
                    visualization_results = results.get('visualization', {})
                    
                    analysis_data = analyst_results.get('analysis', {})
                    visualization_data = visualization_results.get('visualization_data', {})
                    
                    # Display results in tabs
                    tabs = st.tabs([
                        "📊 Summary",
                        "📈 Timeline",
                        "☁️ Wordcloud",
                        "🎯 Aspect Analysis",
                        "💡 Recommendations"
                    ])
                    
                    # Shared: advisor insights for expanders
                    advisor_insights = advisor_results.get('advisor_insights', {})

                    # ============ TAB 1: SUMMARY ============
                    with tabs[0]:
                        st.subheader("Analysis Summary")

                        # Topic — full width so long names don't truncate
                        _topic_display = visualization_data.get('topic', 'N/A')
                        st.markdown(f"### {_topic_display}")

                        # Key metrics (without "Total Posts" which is always 0 for comment-only scrapes)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Comments", visualization_data.get('total_comments', 0))
                        with col2:
                            _conf = analysis_data.get('confidence', 0)
                            st.metric("Model Confidence", f"{_conf:.1%}")
                        with col3:
                            _sent = visualization_data.get('overall_sentiment', 'neutral').upper()
                            _sent_emoji = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}.get(_sent, "")
                            st.metric("Overall Sentiment", f"{_sent_emoji} {_sent}")

                        st.divider()

                        # Data sources
                        st.subheader("Data Sources")
                        subreddits = visualization_data.get('subreddits', [])
                        st.write(f"**Analyzed {len(subreddits)} source(s):**")
                        for subreddit in subreddits:
                            st.write(f"  • r/{subreddit}" if not str(subreddit).lower().startswith("youtube") else f"  • {subreddit}")

                        st.divider()

                        # Sentiment distribution with chart
                        st.subheader("Sentiment Distribution")

                        distribution = analysis_data.get('sentiment_distribution', {})
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            pos_data = distribution.get('positive', {})
                            st.metric("😊 Positive", f"{pos_data.get('count', 0)}", f"{pos_data.get('percentage', 0):.1f}%")

                        with col2:
                            neu_data = distribution.get('neutral', {})
                            st.metric("😐 Neutral", f"{neu_data.get('count', 0)}", f"{neu_data.get('percentage', 0):.1f}%")

                        with col3:
                            neg_data = distribution.get('negative', {})
                            st.metric("😞 Negative", f"{neg_data.get('count', 0)}", f"{neg_data.get('percentage', 0):.1f}%")

                        # Distribution pie chart
                        try:
                            import plotly.graph_objects as go

                            labels = list(distribution.keys())
                            sizes  = [distribution[label]["count"] for label in labels]
                            colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

                            fig = go.Figure(data=[go.Pie(
                                labels=labels,
                                values=sizes,
                                marker=dict(colors=[colors.get(label, "#999999") for label in labels]),
                                textposition="inside",
                                textinfo="label+percent",
                            )])
                            fig.update_layout(title="Sentiment Distribution Chart", height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Could not display chart: {str(e)}")

                        # AI insight expander
                        if advisor_insights.get("summary"):
                            with st.expander("AI Advisor Insight", expanded=False):
                                st.markdown(
                                    _style_insight(advisor_insights["summary"]),
                                    unsafe_allow_html=True,
                                )
                    
                    # ============ TAB 2: TIMELINE ============
                    with tabs[1]:
                        st.subheader("Sentiment Timeline")
                        st.info("📈 Shows how sentiment evolved across time periods. Red triangles mark unusual negative spikes.")

                        try:
                            from tools.visualization_tools import generate_timeline_chart
                            from tools.rca_tools import detect_anomalies

                            detailed_sentiments = analysis_data.get('detailed_sentiments', [])
                            if detailed_sentiments:
                                # Detect anomalies (pure math, no API calls)
                                anomalies = detect_anomalies(detailed_sentiments)

                                fig = generate_timeline_chart(
                                    detailed_sentiments,
                                    anomaly_dates=anomalies if anomalies else None,
                                )

                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Could not generate timeline chart")

                                # ── Root Cause Analysis panel ─────────────────
                                if anomalies:
                                    st.divider()
                                    st.subheader("⚠️ Negative Sentiment Anomalies")
                                    st.caption(
                                        f"Found **{len(anomalies)}** date(s) where negative sentiment "
                                        "spiked unusually high (rolling Z-score ≥ 2.5). "
                                        "Click a date to investigate the root cause."
                                    )

                                    # Anomaly date buttons
                                    btn_cols = st.columns(min(3, len(anomalies)))
                                    for idx, anom in enumerate(anomalies):
                                        with btn_cols[idx % 3]:
                                            label = (
                                                f"📍 {anom['date']}\n"
                                                f"{anom['neg_pct']}% negative  •  Z = {anom['z_score']}"
                                            )
                                            if st.button(
                                                label,
                                                key=f"rca_btn_{job_id}_{anom['date']}",
                                                use_container_width=True,
                                            ):
                                                st.session_state[f"rca_selected_{job_id}"] = anom["date"]

                                    # Show RCA result for selected date
                                    selected = st.session_state.get(f"rca_selected_{job_id}")
                                    if selected:
                                        st.markdown(f"---\n### Root Cause Analysis — {selected}")
                                        cache_key = f"rca_result_{job_id}_{selected}"

                                        if cache_key not in st.session_state:
                                            with st.spinner(f"Searching for root cause of {selected} spike..."):
                                                try:
                                                    rca_resp = requests.post(
                                                        f"{API_BASE_URL}/rca/{job_id}",
                                                        json={"date": selected},
                                                        timeout=60,
                                                    )
                                                    if rca_resp.status_code == 200:
                                                        st.session_state[cache_key] = rca_resp.json()
                                                    else:
                                                        detail = rca_resp.json().get("detail", "Unknown error")
                                                        st.error(f"RCA failed: {detail}")
                                                except Exception as rca_err:
                                                    st.error(f"RCA request error: {str(rca_err)[:120]}")

                                        rca = st.session_state.get(cache_key)
                                        if rca:
                                            status = rca.get("status", "UNCERTAIN")
                                            status_colour = {
                                                "MATCH":     "green",
                                                "MISMATCH":  "orange",
                                                "UNCERTAIN": "blue",
                                                "NO_DATA":   "grey",
                                                "ERROR":     "red",
                                            }.get(status, "grey")

                                            col_a, col_b = st.columns([1, 3])
                                            with col_a:
                                                st.markdown(
                                                    f"**Verdict**  \n"
                                                    f"<span style='color:{status_colour};font-size:1.1em;font-weight:bold'>"
                                                    f"{status}</span>",
                                                    unsafe_allow_html=True,
                                                )
                                            with col_b:
                                                aspects = rca.get("aspects_analyzed", [])
                                                if aspects:
                                                    st.markdown(f"**Spike aspects:** {', '.join(aspects)}")
                                                src = rca.get("search_source", "none")
                                                n   = rca.get("snippets_found", 0)
                                                st.caption(f"Web search: {n} result(s) via {src}")

                                            st.markdown(f"**Analysis:** {rca.get('reasoning', '—')}")
                                            st.info(f"**Root Cause:** {rca.get('root_cause_summary', '—')}")

                                elif len(detailed_sentiments) > 0 and len(detailed_sentiments) < 14:
                                    st.caption(
                                        "Anomaly detection requires at least 14 days of data "
                                        f"(current dataset spans fewer days)."
                                    )
                            else:
                                st.info("No sentiment data available for timeline")

                            # AI insight expander (inside try so import errors are caught)
                            if advisor_insights.get("timeline"):
                                with st.expander("AI Advisor Insight", expanded=False):
                                    st.markdown(
                                        _style_insight(advisor_insights["timeline"]),
                                        unsafe_allow_html=True,
                                    )

                        except ImportError as e:
                            st.error(f"Import error: {str(e)}")
                            st.info("Try running from project root: `streamlit run frontend/app.py`")
                        except Exception as e:
                            st.error(f"Error generating timeline: {str(e)}")

                    # ============ TAB 3: WORDCLOUD ============
                    with tabs[2]:
                        st.subheader("Word Frequency Analysis")
                        st.info("Shows the most frequent words by sentiment and word type (nouns, verbs, adjectives)")

                        try:
                            from tools.visualization_tools import generate_wordcloud_by_sentiment

                            _wc_processed = results.get('state', {}).get('processed_data', [])
                            _wc_topic     = visualization_data.get('topic', '')
                            _wc_cat       = visualization_data.get('category_detail', '')

                            # Cache wordclouds per job so they aren't regenerated on every widget interaction
                            @st.cache_data(show_spinner=False)
                            def _cached_wordclouds(_job_id, _topic, _cat):
                                return generate_wordcloud_by_sentiment(
                                    analysis_data, _wc_processed,
                                    topic=_topic, category_detail=_cat,
                                )

                            if _wc_processed:
                                with st.spinner("Generating wordclouds..."):
                                    wordclouds = _cached_wordclouds(job_id, _wc_topic, _wc_cat)

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
                                        key="wordcloud_pos_filter",
                                    )
                                    pos_key = {"Nouns": "noun", "Verbs": "verb", "Adjectives": "adj"}[pos_filter]

                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        wc_k = f"positive_{pos_key}"
                                        st.subheader("Positive")
                                        if wc_k in wordclouds:
                                            st.image(wordclouds[wc_k], use_container_width=True)
                                        else:
                                            st.info("Not enough data")

                                    with col2:
                                        wc_k = f"neutral_{pos_key}"
                                        st.subheader("Neutral")
                                        if wc_k in wordclouds:
                                            st.image(wordclouds[wc_k], use_container_width=True)
                                        else:
                                            st.info("Not enough data")

                                    with col3:
                                        wc_k = f"negative_{pos_key}"
                                        st.subheader("Negative")
                                        if wc_k in wordclouds:
                                            st.image(wordclouds[wc_k], use_container_width=True)
                                        else:
                                            st.info("Not enough data")

                                    # AI insight expander
                                    if advisor_insights.get("wordcloud"):
                                        with st.expander("AI Advisor Insight", expanded=False):
                                            st.markdown(
                                                _style_insight(advisor_insights["wordcloud"]),
                                                unsafe_allow_html=True,
                                            )
                                else:
                                    st.info("Could not generate wordclouds")
                            else:
                                st.info("No processed data available for wordclouds")

                        except Exception as e:
                            st.error(f"Error generating wordclouds: {str(e)}")
                    
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
                                
                                # Display aspect details table
                                st.subheader("Top Aspects")
                                
                                aspect_list = []
                                for aspect, data in list(aspect_analysis.items())[:10]:
                                    aspect_list.append({
                                        "Aspect": aspect.title(),
                                        "Positive": f"{data['positive']['percentage']:.1f}%",
                                        "Neutral": f"{data['neutral']['percentage']:.1f}%",
                                        "Negative": f"{data['negative']['percentage']:.1f}%",
                                        "Mentions": data['total_mentions']
                                    })
                                
                                st.dataframe(aspect_list, use_container_width=True)

                                # AI insight expander
                                if advisor_insights.get("absa"):
                                    with st.expander("AI Advisor Insight", expanded=False):
                                        st.markdown(
                                            _style_insight(advisor_insights["absa"]),
                                            unsafe_allow_html=True,
                                        )
                            else:
                                st.info("No aspect analysis available")

                        except ImportError as e:
                            st.error(f"Import error: {str(e)}")
                            st.info("Try running from project root: `streamlit run frontend/app.py`")
                        except Exception as e:
                            st.error(f"Error displaying aspect analysis: {str(e)}")
                    
                    # ============ TAB 5: RECOMMENDATIONS & CHAT ============
                    with tabs[4]:
                        st.subheader("What People Are Saying")

                        try:
                            # Consumer-framed recommendations (4 bold-labelled sections)
                            recommendations = advisor_results.get('recommendations', [])

                            if recommendations:
                                for rec in recommendations:
                                    st.markdown(
                                        _style_insight(rec),
                                        unsafe_allow_html=True,
                                    )
                                    st.write("")  # spacing between sections
                            else:
                                st.info("No recommendations available. Run the analysis to generate insights.")

                            st.divider()

                            # ── AI Advisor Chat ──────────────────────────────────────
                            st.subheader("Ask the AI Advisor")

                            if "chat_history" not in st.session_state:
                                st.session_state.chat_history = []

                            # Display chat history
                            for _q, _a in st.session_state.chat_history:
                                with st.chat_message("user"):
                                    st.write(_q)
                                with st.chat_message("assistant"):
                                    st.write(_a)

                            # Chat input (always at the bottom of the conversation)
                            user_question = st.chat_input(
                                "Ask anything about the analysis...",
                                key=f"chat_input_{job_id}",
                            )

                            if user_question:
                                with st.chat_message("user"):
                                    st.write(user_question)
                                with st.chat_message("assistant"):
                                    with st.spinner("Thinking..."):
                                        try:
                                            from agents.advisor_agent import AdvisorAgent
                                            _advisor = AdvisorAgent()
                                            _answer  = _advisor.answer_question(user_question)
                                            st.session_state.chat_history.append((user_question, _answer))
                                            st.write(_answer)
                                        except ImportError as e:
                                            _err = f"Import error: {str(e)}"
                                            st.error(_err)
                                            st.session_state.chat_history.append((user_question, _err))
                                        except Exception as e:
                                            _err = f"Error answering question: {str(e)[:150]}"
                                            st.error(_err)
                                            st.session_state.chat_history.append((user_question, _err))

                            # Suggested questions below the chat input
                            suggested_questions = advisor_results.get('suggested_questions', [])
                            if suggested_questions:
                                st.caption("Try one of these questions:")
                                _sq_cols = st.columns(2)
                                for _si, _sq in enumerate(suggested_questions):
                                    with _sq_cols[_si % 2]:
                                        if st.button(_sq, key=f"sq_{job_id}_{_si}", use_container_width=True):
                                            with st.spinner("Thinking..."):
                                                try:
                                                    from agents.advisor_agent import AdvisorAgent
                                                    _advisor = AdvisorAgent()
                                                    _answer  = _advisor.answer_question(_sq)
                                                    st.session_state.chat_history.append((_sq, _answer))
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Error: {str(e)[:100]}")

                        except Exception as e:
                            st.error(f"Error in recommendations tab: {str(e)}")
                
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