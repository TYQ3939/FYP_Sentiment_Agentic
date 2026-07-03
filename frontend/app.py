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

API_BASE_URL = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")


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
if "compare_mode_enabled" not in st.session_state:
    st.session_state.compare_mode_enabled = False
if "compare_topic_b" not in st.session_state:
    st.session_state.compare_topic_b = None
if "compare_job_id_b" not in st.session_state:
    st.session_state.compare_job_id_b = None

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
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.session_history = []
        st.rerun()

# ========== MAIN TAB ==========

if st.session_state.current_tab == "main":
    st.header("Run Sentiment Analysis")
    
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        topic = st.text_input(
            "Enter Topic:",
            value=st.session_state.current_topic or "",
            placeholder="e.g., iPhone 17 Pro Max"
        )

    with col2:
        start_button = st.button("Run Analysis", type="primary", use_container_width=True)

    with col3:
        compare_label = "- Remove Compare" if st.session_state.compare_mode_enabled else "+ Add Compare"
        if st.button(compare_label, use_container_width=True):
            st.session_state.compare_mode_enabled = not st.session_state.compare_mode_enabled
            if not st.session_state.compare_mode_enabled:
                st.session_state.compare_topic_b = None
            st.rerun()

    # Second topic input (only shown in compare mode)
    topic_b = None
    if st.session_state.compare_mode_enabled:
        topic_b = st.text_input(
            "Compare with (Topic B):",
            value=st.session_state.compare_topic_b or "",
            placeholder="e.g., Samsung Galaxy S25 Ultra",
            key="topic_b_input"
        )
        st.caption("Both topics will be scraped and analyzed in parallel then shown side-by-side.")

    # Optional: Custom subreddits
    with st.expander("Advanced Options"):
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
            if st.session_state.compare_mode_enabled and topic_b and topic_b.strip():
                # ── Compare mode: fire two jobs ──────────────────────────────
                st.session_state.compare_topic_b = topic_b.strip()
                if topic_b.strip() not in st.session_state.session_history:
                    st.session_state.session_history.insert(0, topic_b.strip())

                with st.spinner("Submitting two analysis jobs..."):
                    resp_a = requests.post(
                        f"{API_BASE_URL}/scrape/start",
                        json={"topic": topic, "subreddits": custom_subreddits, "mode": "compare"},
                        timeout=10
                    )
                    resp_b = requests.post(
                        f"{API_BASE_URL}/scrape/start",
                        json={"topic": topic_b.strip(), "subreddits": custom_subreddits, "mode": "compare"},
                        timeout=10
                    )

                if resp_a.status_code == 200 and resp_b.status_code == 200:
                    st.session_state.current_job_id   = resp_a.json()["job_id"]
                    st.session_state.compare_job_id_b = resp_b.json()["job_id"]
                    st.session_state.start_time       = time.time()
                    st.session_state.current_tab      = "monitoring"
                    st.rerun()
                else:
                    failed = []
                    if resp_a.status_code != 200:
                        failed.append(f"Topic A: {resp_a.json().get('detail','Unknown error')}")
                    if resp_b.status_code != 200:
                        failed.append(f"Topic B: {resp_b.json().get('detail','Unknown error')}")
                    st.error("Failed to start jobs: " + "; ".join(failed))

            else:
                # ── Single mode: fire one job ─────────────────────────────────
                st.session_state.compare_job_id_b = None
                with st.spinner("Submitting job to backend..."):
                    response = requests.post(
                        f"{API_BASE_URL}/scrape/start",
                        json={"topic": topic, "subreddits": custom_subreddits},
                        timeout=10
                    )

                if response.status_code == 200:
                    st.session_state.current_job_id = response.json()["job_id"]
                    st.session_state.start_time     = time.time()
                    st.session_state.current_tab    = "monitoring"
                    st.rerun()
                else:
                    st.error(f"Failed to start job: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend API. Start the backend with:\n```bash\npython run_backend.py\n```")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    elif not topic and start_button:
        st.warning("Please enter a topic first.")

# ========== MONITORING TAB ==========

elif st.session_state.current_tab == "monitoring":
    if st.session_state.current_job_id:
        job_id_a      = st.session_state.current_job_id
        compare_job_b = st.session_state.get("compare_job_id_b")
        topic_a       = st.session_state.current_topic or "your topic"
        topic_b       = st.session_state.get("compare_topic_b") or "Topic B"

        if st.button("← Cancel / Back to Input"):
            st.session_state.current_tab = "main"
            st.rerun()

        st.divider()

        # ── Compare-mode monitoring ───────────────────────────────────────────
        if compare_job_b:
            st.header(f"Comparing: {topic_a}  vs  {topic_b}")

            try:
                resp_a = requests.get(f"{API_BASE_URL}/scrape/status/{job_id_a}", timeout=20)
                resp_b = requests.get(f"{API_BASE_URL}/scrape/status/{compare_job_b}", timeout=20)
                status_a = resp_a.json() if resp_a.status_code == 200 else None
                status_b = resp_b.json() if resp_b.status_code == 200 else None
            except Exception as poll_err:
                st.warning(f"Connection slow — retrying in 5 seconds... ({str(poll_err)[:60]})")
                time.sleep(5)
                st.rerun()
                status_a = status_b = None

            if status_a and status_b:
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown(f"**{topic_a}**")
                    prog_a = status_a.get('progress', 0)
                    st.progress(prog_a / 100)
                    st.caption(f"{prog_a}%  —  {status_a.get('status', 'running')}")
                with col_r:
                    st.markdown(f"**{topic_b}**")
                    prog_b = status_b.get('progress', 0)
                    st.progress(prog_b / 100)
                    st.caption(f"{prog_b}%  —  {status_b.get('status', 'running')}")

                both_done  = (status_a['status'] == 'completed' and
                              status_b['status'] == 'completed')
                either_err = (status_a['status'] == 'error' or
                              status_b['status'] == 'error')

                st.divider()

                if both_done:
                    st.success("Both analyses complete!")
                    if st.button("View Comparison", type="primary"):
                        st.session_state.current_tab = "compare_results"
                        st.session_state.analysis_complete = True
                        st.rerun()

                elif either_err:
                    for lbl, s in [(topic_a, status_a), (topic_b, status_b)]:
                        if s['status'] == 'error':
                            st.error(f"**{lbl}** failed: {s.get('error', 'Unknown error')}")
                    if st.button("← Try Again"):
                        st.session_state.current_tab = "main"
                        st.rerun()

                else:
                    elapsed = time.time() - st.session_state.get("start_time", time.time())
                    if elapsed > 1500:
                        st.warning(
                            "This is taking longer than expected. Two parallel jobs typically "
                            "take 10-25 minutes. You can wait or cancel and try again."
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
                            f"Both AI pipelines are running in parallel. "
                            f"This typically takes 10-25 minutes. "
                            f"Results appear once both are complete."
                        )
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("Refresh Now"):
                                st.rerun()
                        with col2:
                            st.caption("Auto-refreshes every 5 seconds.")
                    time.sleep(5)
                    st.rerun()

        # ── Single-mode monitoring ────────────────────────────────────────────
        else:
            topic_display = topic_a
            st.header(f"Analyzing: {topic_display}")

            job_id     = job_id_a
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

                st.progress(progress / 100)
                st.caption(f"{progress}% complete")
                st.divider()

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

                    if elapsed > 1500:
                        st.warning(
                            f"This is taking longer than expected (over 25 minutes). "
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
                        with st.popover("📥 Export Results", use_container_width=True):
                            import io, csv as _csv, json as _json
                            from datetime import datetime as _dt

                            _topic   = st.session_state.get("current_topic", "analysis")
                            _slug    = _topic.replace(" ", "_").replace("/", "_")[:40]
                            _analyst = results.get("analyst", {})
                            _analysis = _analyst.get("analysis", {})
                            _detailed = _analysis.get("detailed_sentiments", [])
                            _aspects  = (
                                results.get("advisor", {}).get("aspect_analysis")
                                or results.get("visualization", {})
                                         .get("visualization_data", {})
                                         .get("aspect_analysis", {})
                                or {}
                            )
                            _advisor = results.get("advisor", {})

                            st.markdown("**Choose export format:**")

                            # ── 1. Sentiment data CSV ─────────────────────
                            if _detailed:
                                _buf = io.StringIO()
                                _w = _csv.writer(_buf)
                                _w.writerow(["text", "sentiment", "confidence", "created_at"])
                                for _r in _detailed:
                                    _w.writerow([
                                        _r.get("text", ""),
                                        _r.get("label", ""),
                                        f'{_r.get("confidence", 0):.4f}',
                                        _r.get("created_at", ""),
                                    ])
                                st.download_button(
                                    "📄 Sentiment Data (.csv)",
                                    data=_buf.getvalue(),
                                    file_name=f"sentiment_{_slug}.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="dl_sentiment_csv",
                                )
                            else:
                                st.caption("No sentiment data available")

                            # ── 2. Aspect analysis CSV ────────────────────
                            if _aspects:
                                _buf2 = io.StringIO()
                                _w2 = _csv.writer(_buf2)
                                _w2.writerow([
                                    "Aspect", "Positive %", "Positive Count",
                                    "Neutral %", "Neutral Count",
                                    "Negative %", "Negative Count", "Total Mentions",
                                ])
                                for _asp, _d in sorted(
                                    _aspects.items(),
                                    key=lambda x: x[1].get("total_mentions", 0),
                                    reverse=True,
                                ):
                                    _w2.writerow([
                                        _asp,
                                        f"{_d.get('positive', {}).get('percentage', 0):.1f}",
                                        _d.get('positive', {}).get('count', 0),
                                        f"{_d.get('neutral',  {}).get('percentage', 0):.1f}",
                                        _d.get('neutral',  {}).get('count', 0),
                                        f"{_d.get('negative', {}).get('percentage', 0):.1f}",
                                        _d.get('negative', {}).get('count', 0),
                                        _d.get('total_mentions', 0),
                                    ])
                                st.download_button(
                                    "📊 Aspect Analysis (.csv)",
                                    data=_buf2.getvalue(),
                                    file_name=f"aspects_{_slug}.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="dl_aspects_csv",
                                )
                            else:
                                st.caption("No aspect data available")

                            # ── 3. Full summary JSON ──────────────────────
                            _summary = {
                                "topic": _topic,
                                "exported_at": _dt.now().isoformat(),
                                "overall_sentiment": _analysis.get("overall_sentiment"),
                                "confidence": _analysis.get("confidence"),
                                "total_texts_analyzed": _analysis.get("total_texts_analyzed"),
                                "sentiment_distribution": _analysis.get("sentiment_distribution"),
                                "advisor_insights": _advisor.get("advisor_insights", {}),
                                "recommendations": _advisor.get("recommendations", []),
                                "aspect_analysis": _aspects,
                            }
                            st.download_button(
                                "📋 Full Summary (.json)",
                                data=_json.dumps(_summary, indent=2, ensure_ascii=False),
                                file_name=f"summary_{_slug}.json",
                                mime="application/json",
                                use_container_width=True,
                                key="dl_summary_json",
                            )

                    with col3:
                        if st.session_state.get("compare_job_id_b"):
                            if st.button("← Back to Comparison"):
                                st.session_state.current_tab = "compare_results"
                                st.rerun()
                    
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
                        _DIST_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                        _sorted_dist = sorted(distribution.items(), key=lambda x: x[1].get('count', 0), reverse=True)
                        col1, col2, col3 = st.columns(3)
                        for _dcol, (_dlabel, _ddata) in zip([col1, col2, col3], _sorted_dist):
                            with _dcol:
                                st.metric(
                                    f"{_DIST_EMOJI.get(_dlabel, '')} {_dlabel.title()}",
                                    f"{_ddata.get('count', 0)}",
                                    f"{_ddata.get('percentage', 0):.1f}%"
                                )
                        
                        st.metric("Confidence", f"{analysis_data.get('confidence', 0):.2%}")
                        
                        # Display sentiment chart
                        try:
                            import plotly.graph_objects as go
                            
                            _pie_items = sorted(distribution.items(), key=lambda x: x[1].get("count", 0), reverse=True)
                            labels = [i[0] for i in _pie_items]
                            sizes  = [i[1].get("count", 0) for i in _pie_items]
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
                            
                            st.plotly_chart(fig, use_container_width=True)

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
                        st.info("Shows how sentiment evolved over time. Red triangles mark unusual negative spikes.")

                        try:
                            from tools.visualization_tools import generate_timeline_chart
                            from tools.rca_tools import detect_anomalies

                            detailed_sentiments = analysis_data.get('detailed_sentiments', [])
                            if detailed_sentiments:
                                anomalies = detect_anomalies(detailed_sentiments)
                                fig = generate_timeline_chart(
                                    detailed_sentiments,
                                    anomaly_dates=anomalies if anomalies else None,
                                )

                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Could not generate timeline chart")

                                # ── Root Cause Analysis panel ─────────────
                                if anomalies:
                                    st.divider()
                                    st.subheader("Negative Sentiment Anomalies")
                                    st.caption(
                                        f"Found **{len(anomalies)}** date(s) with unusually high negative sentiment "
                                        "(rolling Z-score >= 2.5). Click a date to investigate."
                                    )

                                    btn_cols = st.columns(min(3, len(anomalies)))
                                    for idx, anom in enumerate(anomalies):
                                        with btn_cols[idx % 3]:
                                            label = (
                                                f"{anom['date']}\n"
                                                f"{anom['neg_pct']}% negative  Z={anom['z_score']}"
                                            )
                                            if st.button(label, key=f"rca_btn_{job_id}_{anom['date']}", use_container_width=True):
                                                st.session_state[f"rca_selected_{job_id}"] = anom["date"]

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
                                                        st.error(f"RCA failed: {rca_resp.json().get('detail', 'Unknown error')}")
                                                except Exception as rca_err:
                                                    st.error(f"RCA request error: {str(rca_err)[:120]}")

                                        rca = st.session_state.get(cache_key)
                                        if rca:
                                            status = rca.get("status", "UNCERTAIN")
                                            status_colour = {
                                                "MATCH"    : "green",
                                                "MISMATCH" : "orange",
                                                "UNCERTAIN": "blue",
                                                "NO_DATA"  : "grey",
                                                "ERROR"    : "red",
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

                                elif len(detailed_sentiments) > 0:
                                    st.caption("Anomaly detection needs at least 14 days of data to work.")
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
                                        st.image(wordclouds["overall"], use_column_width=True)

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
                                            st.image(wordclouds[key], use_column_width=True)
                                        else:
                                            st.info("Not enough data")

                                    with col2:
                                        key = f"neutral_{pos_key}"
                                        st.subheader("😐 Neutral")
                                        if key in wordclouds:
                                            st.image(wordclouds[key], use_column_width=True)
                                        else:
                                            st.info("Not enough data")

                                    with col3:
                                        key = f"negative_{pos_key}"
                                        st.subheader("😞 Negative")
                                        if key in wordclouds:
                                            st.image(wordclouds[key], use_column_width=True)
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
                                    st.plotly_chart(fig, use_container_width=True)

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

                                st.table(aspect_list)
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

                            # Chat input (text_input + button avoids ChatInput.js module load error)
                            _ci_col, _ci_btn_col = st.columns([5, 1])
                            with _ci_col:
                                _ci_draft = st.text_input(
                                    "Question",
                                    label_visibility="collapsed",
                                    placeholder="Ask anything about the analysis...",
                                    key=f"chat_input_{job_id}",
                                )
                            with _ci_btn_col:
                                _ci_send = st.button("Ask", key=f"chat_send_{job_id}",
                                                     use_container_width=True)
                            user_question = _ci_draft.strip() if (_ci_send and _ci_draft.strip()) else None

                            # ── Suggested questions (shown under the chat input) ──
                            suggested_questions = advisor_results.get('suggested_questions', [])
                            if suggested_questions:
                                st.caption("Or try one of these:")
                                cols = st.columns(2)
                                for idx, question in enumerate(suggested_questions):
                                    with cols[idx % 2]:
                                        if st.button(question, use_container_width=True, key=f"suggested_{idx}"):
                                            if "chat_history" not in st.session_state:
                                                st.session_state.chat_history = []
                                            with st.spinner("Thinking..."):
                                                try:
                                                    resp = requests.post(
                                                        f"{API_BASE_URL}/advisor/question/{job_id}",
                                                        json={"question": question},
                                                        timeout=60,
                                                    )
                                                    resp.raise_for_status()
                                                    answer = resp.json().get("answer", "No answer returned.")
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
                                            resp = requests.post(
                                                f"{API_BASE_URL}/advisor/question/{job_id}",
                                                json={"question": user_question},
                                                timeout=60,
                                            )
                                            resp.raise_for_status()
                                            answer = resp.json().get("answer", "No answer returned.")
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

# ========== COMPARE RESULTS TAB ==========

elif st.session_state.current_tab == "compare_results":
    job_id_a = st.session_state.get("current_job_id")
    job_id_b = st.session_state.get("compare_job_id_b")

    if not job_id_a or not job_id_b:
        st.warning("No comparison available.")
        if st.button("← Back to Main"):
            st.session_state.current_tab = "main"
            st.rerun()
    else:
        # Fetch both result payloads
        try:
            with st.spinner("Loading comparison results..."):
                resp_a = requests.get(f"{API_BASE_URL}/scrape/results/{job_id_a}", timeout=30)
                resp_b = requests.get(f"{API_BASE_URL}/scrape/results/{job_id_b}", timeout=30)
            job_a = resp_a.json() if resp_a.status_code == 200 else None
            job_b = resp_b.json() if resp_b.status_code == 200 else None
        except Exception as fetch_err:
            st.error(f"Error loading results: {str(fetch_err)}")
            if st.button("← Back"):
                st.session_state.current_tab = "main"
                st.rerun()
            job_a = job_b = None

        if not (job_a and job_b and job_a.get('results') and job_b.get('results')):
            st.error("Could not load results for one or both topics.")
            if st.button("← Back"):
                st.session_state.current_tab = "main"
                st.rerun()
        else:
            results_a = job_a['results']
            results_b = job_b['results']

            analyst_a  = results_a.get('analyst', {})
            advisor_a  = results_a.get('advisor', {})
            viz_a      = results_a.get('visualization', {})
            analysis_a = analyst_a.get('analysis', {})
            viz_data_a = viz_a.get('visualization_data', {})

            analyst_b  = results_b.get('analyst', {})
            advisor_b  = results_b.get('advisor', {})
            viz_b      = results_b.get('visualization', {})
            analysis_b = analyst_b.get('analysis', {})
            viz_data_b = viz_b.get('visualization_data', {})

            topic_a = viz_data_a.get('topic', st.session_state.current_topic or "Topic A")
            topic_b = viz_data_b.get('topic', st.session_state.get("compare_topic_b") or "Topic B")

            st.header(f"Comparison: {topic_a}  vs  {topic_b}")

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                if st.button("New Analysis", type="primary"):
                    st.session_state.current_tab        = "main"
                    st.session_state.analysis_complete  = False
                    st.session_state.current_job_id     = None
                    st.session_state.compare_job_id_b   = None
                    st.session_state.compare_mode_enabled = False
                    st.rerun()
            with col_btn2:
                if st.button("View Topic A Results"):
                    st.session_state.current_tab = "results"
                    st.rerun()

            st.divider()

            # ── Shared styling helper ────────────────────────────────────────
            def _style_insight(text: str) -> str:
                import re
                text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                text = re.sub(r'\b(positive|positively|strength|strengths)\b',
                              r'<span style="color:#27ae60;font-weight:600">\1</span>',
                              text, flags=re.IGNORECASE)
                text = re.sub(r'\b(negative|negatively|weakness|weaknesses|concern|concerning)\b',
                              r'<span style="color:#e74c3c;font-weight:600">\1</span>',
                              text, flags=re.IGNORECASE)
                text = re.sub(r'\b(neutral|mixed)\b',
                              r'<span style="color:#7f8c8d;font-weight:600">\1</span>',
                              text, flags=re.IGNORECASE)
                return text

            # ── Compare tabs ─────────────────────────────────────────────────
            ctabs = st.tabs([
                "Summary",
                "Timeline",
                "Wordcloud",
                "Aspect Analysis",
                "AI Advisor",
            ])

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — SUMMARY
            # ════════════════════════════════════════════════════════════════
            with ctabs[0]:
                st.subheader("Sentiment Overview")

                # Side-by-side donut charts
                try:
                    import plotly.graph_objects as go
                    COLORS = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

                    def _donut(analysis_data, topic_label):
                        dist   = analysis_data.get('sentiment_distribution', {})
                        _items = sorted(dist.items(), key=lambda x: x[1].get('count', 0), reverse=True)
                        labels = [i[0] for i in _items]
                        values = [i[1].get('count', 0) for i in _items]
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.45,
                            marker=dict(colors=[COLORS.get(l, "#999") for l in labels]),
                            textposition="inside",
                            textinfo="label+percent",
                        )])
                        fig.update_layout(
                            title=topic_label,
                            height=340,
                            margin=dict(t=50, b=10, l=10, r=10),
                            showlegend=False,
                        )
                        return fig

                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.plotly_chart(_donut(analysis_a, topic_a), use_container_width=True)
                    with col_r:
                        st.plotly_chart(_donut(analysis_b, topic_b), use_container_width=True)

                except Exception as e:
                    st.error(f"Chart error: {str(e)}")

                st.divider()

                # Key metrics
                col_l, col_r = st.columns(2)
                EMOJI_MAP = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}
                for col, anal, topic_label in [(col_l, analysis_a, topic_a),
                                               (col_r, analysis_b, topic_b)]:
                    with col:
                        st.markdown(f"**{topic_label}**")
                        ov  = anal.get('overall_sentiment', 'neutral').upper()
                        emo = EMOJI_MAP.get(ov, "❓")
                        dist = anal.get('sentiment_distribution', {})
                        st.metric("Overall Sentiment", f"{emo} {ov}")
                        st.metric("Confidence", f"{anal.get('confidence', 0):.1%}")
                        total_a_comments = anal.get('total_texts_analyzed', 0)
                        st.metric("Comments Analyzed", total_a_comments)
                        pos_p = dist.get('positive', {}).get('percentage', 0)
                        neu_p = dist.get('neutral',  {}).get('percentage', 0)
                        neg_p = dist.get('negative', {}).get('percentage', 0)
                        st.caption(f"Pos {pos_p:.1f}%  /  Neu {neu_p:.1f}%  /  Neg {neg_p:.1f}%")

                st.divider()

                # Variance sentence
                pos_a = analysis_a.get('sentiment_distribution', {}).get('positive', {}).get('percentage', 0)
                pos_b = analysis_b.get('sentiment_distribution', {}).get('positive', {}).get('percentage', 0)
                diff  = round(abs(pos_a - pos_b), 1)
                if diff > 0.5:
                    winner   = topic_a if pos_a > pos_b else topic_b
                    loser    = topic_b if pos_a > pos_b else topic_a
                    st.info(
                        f"**{winner}** holds a **{diff}% higher positive public sentiment score** "
                        f"than **{loser}**."
                    )
                else:
                    st.info(
                        f"**{topic_a}** and **{topic_b}** have nearly identical positive "
                        f"sentiment scores ({pos_a:.1f}% vs {pos_b:.1f}%)."
                    )

                # ── Per-topic summary insights ────────────────────────────
                ins_a = advisor_a.get('advisor_insights', {}).get('summary', '')
                ins_b = advisor_b.get('advisor_insights', {}).get('summary', '')
                if ins_a or ins_b:
                    st.divider()
                    col_l, col_r = st.columns(2)
                    with col_l:
                        if ins_a:
                            with st.expander(f"AI Advisor — {topic_a}", expanded=True):
                                st.markdown(_style_insight(ins_a), unsafe_allow_html=True)
                    with col_r:
                        if ins_b:
                            with st.expander(f"AI Advisor — {topic_b}", expanded=True):
                                st.markdown(_style_insight(ins_b), unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — TIMELINE
            # ════════════════════════════════════════════════════════════════
            with ctabs[1]:
                st.subheader("Sentiment Timeline Comparison")
                st.caption(
                    "Each topic has its own graph with independent anomaly detection. "
                    "Red triangles mark unusual negative spikes."
                )

                from tools.visualization_tools import generate_timeline_chart
                from tools.rca_tools import detect_anomalies

                _advisor_map     = {job_id_a: advisor_a, job_id_b: advisor_b}
                _topic_label_map = {job_id_a: topic_a,   job_id_b: topic_b}

                for topic_label, anal, jid in [
                    (topic_a, analysis_a, job_id_a),
                    (topic_b, analysis_b, job_id_b),
                ]:
                    st.markdown(f"#### {topic_label}")
                    detailed = anal.get('detailed_sentiments', [])
                    try:
                        if detailed:
                            anomalies = detect_anomalies(detailed)
                            fig = generate_timeline_chart(
                                detailed,
                                anomaly_dates=anomalies if anomalies else None,
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                            if anomalies:
                                st.caption(
                                    f"Found **{len(anomalies)}** anomaly date(s) — click to investigate."
                                )
                                btn_cols = st.columns(min(3, len(anomalies)))
                                for idx, anom in enumerate(anomalies):
                                    with btn_cols[idx % 3]:
                                        label = (
                                            f"{anom['date']}\n"
                                            f"{anom['neg_pct']}% neg  Z={anom['z_score']}"
                                        )
                                        rca_key = f"rca_sel_{jid}"
                                        if st.button(label, key=f"rca_{jid}_{anom['date']}",
                                                     use_container_width=True):
                                            st.session_state[rca_key] = anom["date"]

                                selected = st.session_state.get(f"rca_sel_{jid}")
                                if selected:
                                    st.markdown(f"**RCA — {selected}**")
                                    cache_key = f"rca_result_{jid}_{selected}"
                                    if cache_key not in st.session_state:
                                        with st.spinner("Searching for root cause..."):
                                            try:
                                                rca_resp = requests.post(
                                                    f"{API_BASE_URL}/rca/{jid}",
                                                    json={"date": selected},
                                                    timeout=60,
                                                )
                                                if rca_resp.status_code == 200:
                                                    st.session_state[cache_key] = rca_resp.json()
                                            except Exception as rca_err:
                                                st.error(str(rca_err)[:120])

                                    rca = st.session_state.get(cache_key)
                                    if rca:
                                        s_col = {"MATCH":"green","MISMATCH":"orange",
                                                 "UNCERTAIN":"blue","NO_DATA":"grey","ERROR":"red"}
                                        status_color = s_col.get(rca.get("status","UNCERTAIN"), "grey")
                                        st.markdown(
                                            f"Verdict: <span style='color:{status_color};"
                                            f"font-weight:bold'>{rca.get('status','?')}</span>  "
                                            f"— {rca.get('root_cause_summary','—')}",
                                            unsafe_allow_html=True
                                        )
                        else:
                            st.info(f"No timeline data for {topic_label}")
                    except Exception as e:
                        st.error(f"Timeline error for {topic_label}: {str(e)[:100]}")

                    _tl_ins = _advisor_map[jid].get('advisor_insights', {}).get('timeline', '')
                    if _tl_ins:
                        with st.expander(f"AI Advisor Insight — {topic_label}", expanded=False):
                            st.markdown(_style_insight(_tl_ins), unsafe_allow_html=True)

                    st.divider()

            # ════════════════════════════════════════════════════════════════
            # TAB 3 — WORDCLOUD
            # ════════════════════════════════════════════════════════════════
            with ctabs[2]:
                st.subheader("Word Frequency Comparison")

                wc_view = st.radio(
                    "View:",
                    ["Side-by-Side Overview", f"Deep Dive: {topic_a}", f"Deep Dive: {topic_b}"],
                    horizontal=True,
                    key="compare_wc_view",
                )

                proc_a = results_a.get('state', {}).get('processed_data', [])
                proc_b = results_b.get('state', {}).get('processed_data', [])
                cat_detail_a = viz_data_a.get('category_detail', '')
                cat_detail_b = viz_data_b.get('category_detail', '')

                try:
                    with st.spinner("Generating wordclouds..."):
                        wc_a = _cached_generate_wordclouds(
                            job_id_a, analysis_a, proc_a, topic_a, cat_detail_a
                        ) if proc_a else {}
                        wc_b = _cached_generate_wordclouds(
                            job_id_b, analysis_b, proc_b, topic_b, cat_detail_b
                        ) if proc_b else {}

                    if wc_view == "Side-by-Side Overview":
                        master_keys = [
                            ("Overall",  "overall"),
                            ("Positive", "positive_noun"),
                            ("Neutral",  "neutral_noun"),
                            ("Negative", "negative_noun"),
                        ]
                        for label, key in master_keys:
                            st.markdown(f"**{label}**")
                            col_l, col_r = st.columns(2)
                            with col_l:
                                st.caption(topic_a)
                                if key in wc_a:
                                    st.image(wc_a[key], use_column_width=True)
                                else:
                                    st.info("Not enough data")
                            with col_r:
                                st.caption(topic_b)
                                if key in wc_b:
                                    st.image(wc_b[key], use_column_width=True)
                                else:
                                    st.info("Not enough data")
                            st.divider()

                    else:
                        # Deep dive for one topic
                        wc_sel   = wc_a if wc_view.endswith(topic_a) else wc_b
                        t_sel    = topic_a if wc_view.endswith(topic_a) else topic_b
                        st.markdown(f"#### {t_sel} — Full Wordcloud Grid")

                        if "overall" in wc_sel:
                            st.subheader("Overall")
                            st.image(wc_sel["overall"], use_column_width=True)
                            st.divider()

                        pos_filter = st.radio(
                            "Word type:",
                            ["Nouns", "Verbs", "Adjectives"],
                            horizontal=True,
                            key=f"dd_pos_{t_sel}"
                        )
                        pos_key = {"Nouns": "noun", "Verbs": "verb", "Adjectives": "adj"}[pos_filter]

                        dd_col1, dd_col2, dd_col3 = st.columns(3)
                        for col, sentiment, emoji in [
                            (dd_col1, "positive", "Positive"),
                            (dd_col2, "neutral",  "Neutral"),
                            (dd_col3, "negative", "Negative"),
                        ]:
                            with col:
                                st.subheader(emoji)
                                key = f"{sentiment}_{pos_key}"
                                if key in wc_sel:
                                    st.image(wc_sel[key], use_column_width=True)
                                else:
                                    st.info("Not enough data")

                except Exception as e:
                    st.error(f"Wordcloud error: {str(e)}")

                # ── Per-topic wordcloud insights ──────────────────────────
                wc_ins_a = advisor_a.get('advisor_insights', {}).get('wordcloud', '')
                wc_ins_b = advisor_b.get('advisor_insights', {}).get('wordcloud', '')
                if wc_ins_a or wc_ins_b:
                    st.divider()
                    col_l, col_r = st.columns(2)
                    with col_l:
                        if wc_ins_a:
                            with st.expander(f"AI Advisor — {topic_a}", expanded=True):
                                st.markdown(_style_insight(wc_ins_a), unsafe_allow_html=True)
                    with col_r:
                        if wc_ins_b:
                            with st.expander(f"AI Advisor — {topic_b}", expanded=True):
                                st.markdown(_style_insight(wc_ins_b), unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════════════
            # TAB 4 — ABSA HEATMAPS
            # ════════════════════════════════════════════════════════════════
            with ctabs[3]:
                st.subheader("Aspect Sentiment Comparison")
                st.caption(
                    "Each cell shows the percentage of comments for that aspect-sentiment pair. "
                    "Darker red = higher concentration."
                )

                try:
                    from tools.visualization_tools import generate_absa_heatmap

                    aspects_a = viz_data_a.get('aspect_analysis', {})
                    aspects_b = viz_data_b.get('aspect_analysis', {})

                    col_l, col_r = st.columns(2)
                    with col_l:
                        fig_a = generate_absa_heatmap(aspects_a, topic_a)
                        if fig_a:
                            st.plotly_chart(fig_a, use_container_width=True)
                        else:
                            st.info(f"No aspect data for {topic_a}")
                    with col_r:
                        fig_b = generate_absa_heatmap(aspects_b, topic_b)
                        if fig_b:
                            st.plotly_chart(fig_b, use_container_width=True)
                        else:
                            st.info(f"No aspect data for {topic_b}")

                    # Data tables
                    st.divider()
                    col_l, col_r = st.columns(2)
                    for col, aspects_dict, t_label in [
                        (col_l, aspects_a, topic_a),
                        (col_r, aspects_b, topic_b),
                    ]:
                        with col:
                            st.markdown(f"**{t_label} — Top Aspects**")
                            others_row  = aspects_dict.get("Others")
                            named_items = [(a, d) for a, d in aspects_dict.items()
                                           if a != "Others"][:9]
                            table_items = named_items + ([("Others", others_row)]
                                                         if others_row else [])
                            if table_items:
                                rows = []
                                for aspect, data in table_items:
                                    rows.append({
                                        "Aspect"  : aspect.title(),
                                        "Positive": f"{data['positive']['percentage']:.0f}%",
                                        "Neutral" : f"{data['neutral']['percentage']:.0f}%",
                                        "Negative": f"{data['negative']['percentage']:.0f}%",
                                        "Mentions": data['total_mentions'],
                                    })
                                st.table(rows)
                            else:
                                st.info("No aspect data")

                    # ── Per-topic ABSA insights ──────────────────────────
                    absa_ins_a = advisor_a.get('advisor_insights', {}).get('absa_insight', '')
                    absa_ins_b = advisor_b.get('advisor_insights', {}).get('absa_insight', '')
                    if absa_ins_a or absa_ins_b:
                        st.divider()
                        col_l, col_r = st.columns(2)
                        with col_l:
                            if absa_ins_a:
                                with st.expander(f"AI Advisor — {topic_a}", expanded=True):
                                    st.markdown(_style_insight(absa_ins_a), unsafe_allow_html=True)
                        with col_r:
                            if absa_ins_b:
                                with st.expander(f"AI Advisor — {topic_b}", expanded=True):
                                    st.markdown(_style_insight(absa_ins_b), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Aspect heatmap error: {str(e)}")

            # ════════════════════════════════════════════════════════════════
            # TAB 5 — AI ADVISOR (BENCHMARK SYNTHESIS + FULL ADVISOR)
            # ════════════════════════════════════════════════════════════════
            with ctabs[4]:
                st.subheader("AI Advisor — Comparison")

                # ── Competitive Benchmark Report ──────────────────────────────
                st.markdown("### Competitive Benchmark Report")
                st.caption(
                    "The AI advisor compares both topics simultaneously across sentiment, "
                    "aspects, and trends to produce a structured competitive report."
                )

                bench_key = f"compare_insight_{job_id_a}_{job_id_b}"
                if bench_key not in st.session_state:
                    with st.spinner("Generating comparative AI analysis..."):
                        try:
                            from langchain_groq import ChatGroq
                            import os as _os
                            _groq_key = _os.getenv("GROQ_API_KEY", "")
                            if not _groq_key:
                                st.error("GROQ_API_KEY not set.")
                                st.session_state[bench_key] = None
                            else:
                                _llm = ChatGroq(model="llama-3.3-70b-versatile",
                                                temperature=0.1, groq_api_key=_groq_key)
                                from tools.advisor_tools import generate_compare_insight
                                st.session_state[bench_key] = generate_compare_insight(
                                    topic_a, analysis_a, viz_data_a.get('aspect_analysis', {}),
                                    topic_b, analysis_b, viz_data_b.get('aspect_analysis', {}),
                                    _llm,
                                )
                        except Exception as ce:
                            st.error(f"Compare insight error: {str(ce)[:200]}")
                            st.session_state[bench_key] = None

                bench_text = st.session_state.get(bench_key)
                if bench_text:
                    st.markdown(_style_insight(bench_text), unsafe_allow_html=True)
                if st.button("Regenerate Benchmark Report"):
                    if bench_key in st.session_state:
                        del st.session_state[bench_key]
                    st.rerun()

                st.divider()

                # ── Per-topic Strategic Recommendations ───────────────────────
                st.subheader("What People Think")
                col_l, col_r = st.columns(2)
                for _col, _adv, _t in [(col_l, advisor_a, topic_a),
                                        (col_r, advisor_b, topic_b)]:
                    with _col:
                        st.markdown(f"**{_t}**")
                        _recs = _adv.get('recommendations', [])
                        if _recs:
                            for _rec in _recs:
                                st.markdown(_style_insight(_rec), unsafe_allow_html=True)
                        else:
                            st.info("Recommendations not available.")

                st.divider()

                # ── Per-topic Suggested Questions ─────────────────────────────
                st.subheader("Suggested Questions")
                col_l, col_r = st.columns(2)
                for _col, _adv, _t, _jid in [
                    (col_l, advisor_a, topic_a, job_id_a),
                    (col_r, advisor_b, topic_b, job_id_b),
                ]:
                    with _col:
                        st.markdown(f"**{_t}**")
                        _qs = _adv.get('suggested_questions', [])
                        if _qs:
                            st.caption("Click to ask:")
                            for _qi, _q in enumerate(_qs):
                                if st.button(_q, key=f"cmp_sq_{_jid}_{_qi}",
                                             use_container_width=True):
                                    st.session_state["_cmp_chat_pending"] = _q
                                    st.rerun()
                        else:
                            st.info("Questions not available.")

                st.divider()

                # ── Chat — ask about either or both topics ────────────────────
                st.subheader("Ask About the Comparison")

                _chat_key = f"cmp_chat_{job_id_a}_{job_id_b}"
                if _chat_key not in st.session_state:
                    st.session_state[_chat_key] = []

                for _cq, _ca in st.session_state[_chat_key]:
                    with st.chat_message("user"):
                        st.write(_cq)
                    with st.chat_message("assistant"):
                        st.write(_ca)

                _pending_q = st.session_state.pop("_cmp_chat_pending", None)
                # text_input + button avoids the ChatInput.js module load error
                _cmp_ci_col, _cmp_ci_btn = st.columns([5, 1])
                with _cmp_ci_col:
                    _typed_q_draft = st.text_input(
                        "Question",
                        label_visibility="collapsed",
                        placeholder="Ask anything about either or both topics...",
                        key=f"cmp_chat_input_{job_id_a}",
                    )
                with _cmp_ci_btn:
                    _typed_send = st.button("Ask", key=f"cmp_chat_send_{job_id_a}",
                                            use_container_width=True)
                _typed_q = _typed_q_draft.strip() if (_typed_send and _typed_q_draft.strip()) else None
                _ask = _pending_q or _typed_q

                if _ask:
                    with st.chat_message("user"):
                        st.write(_ask)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                import os as _os2
                                from langchain_groq import ChatGroq
                                _gk = _os2.getenv("GROQ_API_KEY", "")
                                if not _gk:
                                    _ans = "GROQ_API_KEY not configured."
                                else:
                                    _cllm = ChatGroq(model="llama-3.3-70b-versatile",
                                                     temperature=0.3, groq_api_key=_gk)

                                    def _ds(a):
                                        d  = a.get('sentiment_distribution', {})
                                        ov = a.get('overall_sentiment', 'N/A').upper()
                                        return (
                                            f"{ov} — "
                                            f"Pos {d.get('positive',{}).get('percentage',0):.1f}%, "
                                            f"Neu {d.get('neutral',{}).get('percentage',0):.1f}%, "
                                            f"Neg {d.get('negative',{}).get('percentage',0):.1f}%"
                                        )

                                    _asp_a = ", ".join(
                                        list(viz_data_a.get('aspect_analysis', {}).keys())[:5]
                                    ) or "N/A"
                                    _asp_b = ", ".join(
                                        list(viz_data_b.get('aspect_analysis', {}).keys())[:5]
                                    ) or "N/A"

                                    _ctx_prompt = (
                                        f"You are a sentiment analysis expert comparing two topics "
                                        f"based on Reddit data.\n\n"
                                        f"Topic A — {topic_a}: {_ds(analysis_a)}\n"
                                        f"  Top aspects: {_asp_a}\n\n"
                                        f"Topic B — {topic_b}: {_ds(analysis_b)}\n"
                                        f"  Top aspects: {_asp_b}\n\n"
                                        f"Question: {_ask}\n\n"
                                        f"Answer in 3-5 concise sentences. "
                                        f"Reference specific data when relevant. Be direct."
                                    )
                                    _resp = _cllm.invoke(_ctx_prompt)
                                    _ans  = (_resp.content
                                             if hasattr(_resp, 'content')
                                             else str(_resp)).strip()

                                st.write(_ans)
                                st.session_state[_chat_key].append((_ask, _ans))
                            except Exception as _ce:
                                _err = f"Error: {str(_ce)[:200]}"
                                st.error(_err)
                                st.session_state[_chat_key].append((_ask, _err))

                st.divider()

                # ── Full per-topic insights (all 4 sections, collapsed) ────────
                for _t, _adv in [(topic_a, advisor_a), (topic_b, advisor_b)]:
                    _ai = _adv.get('advisor_insights', {})
                    with st.expander(f"Full AI Insights — {_t}", expanded=False):
                        if _ai:
                            for _sec, _lbl in [
                                ('summary',      'Overall Sentiment'),
                                ('timeline',     'Timeline Trends'),
                                ('wordcloud',    'Discussion Themes'),
                                ('absa_insight', 'Aspect Analysis'),
                            ]:
                                _v = _ai.get(_sec, '')
                                if _v:
                                    st.markdown(f"**{_lbl}**")
                                    st.markdown(_style_insight(_v), unsafe_allow_html=True)
                        else:
                            st.info("Insights not available.")