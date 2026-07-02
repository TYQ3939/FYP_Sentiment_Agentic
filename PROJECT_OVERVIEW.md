# Multi-Agent Reddit Sentiment Analysis System

## System
Python multi-agent pipeline that scrapes Reddit comments for any topic, runs fine-tuned BERTweet sentiment analysis, discovers aspects via BERTopic+UMAP+HDBSCAN, and displays results in a Streamlit UI backed by FastAPI. When Reddit returns no data, the pipeline falls back to YouTube comments, then to a locally cached dataset.

Two processes run simultaneously:
- **Backend** `python run_backend.py` → FastAPI on `http://127.0.0.1:8000`
- **Frontend** `streamlit run frontend/app.py` → Streamlit on `http://localhost:8501`

Frontend never runs agents directly. It submits a topic via REST, polls job progress, then fetches results.

## Directory
```
sentiment_agentic/
├── agents/
│   ├── base_agent.py          # superclass: LLM init, shared_state.json I/O
│   ├── scraper_agent.py       # Arctic Shift → YouTube → local cache fallback chain
│   ├── processor_agent.py     # dedup, wordcloud/sentiment preprocessing
│   ├── analyst_agent.py       # BERTweet inference + BERTopic ABSA pipeline
│   ├── advisor_agent.py       # LLM-powered per-section insights + Q&A
│   └── visualization_agent.py # reads pre-computed state, packages viz data
├── tools/
│   ├── scraper_tools.py       # API calls, LLM subreddit/keyword inference, YouTube scraper, versioned paths
│   ├── processor_tools.py     # text cleaning, dedup, BERTweet preprocessing
│   ├── analyst_tools.py       # BERTweet batch, BERTopic ABSA, LLM label verify
│   ├── advisor_tools.py       # LLM insight/recommendation/question generation
│   ├── visualization_tools.py # Plotly charts, wordcloud, word-frequency helpers
│   ├── rca_tools.py           # anomaly detection, web search, LLM root cause verdict
│   └── common_tools.py        # shared utilities
├── backend/
│   ├── api.py                 # FastAPI routes (incl. /rca endpoint)
│   ├── tasks.py               # background job orchestrator
│   └── database.py            # thread-safe job store → jobs_db.json
├── frontend/
│   └── app.py                 # Streamlit UI
├── data/filtered_data/        # ScraperAgent output / ProcessorAgent input
├── data/analysis/             # AnalystAgent JSON exports per topic
├── shared_state.json          # inter-agent blackboard
├── jobs_db.json               # job progress persistence
├── run_backend.py             # uvicorn entry point
├── .env.example               # template for all required API keys
└── requirements.txt
```

## Agent Pipeline
Sequential, single daemon thread. Progress written to `jobs_db.json`.

```
ScraperAgent (20-30%)
  → ProcessorAgent (40-50%)
    → AnalystAgent (65-75%)
      → AdvisorAgent (85-90%)
        → VisualizationAgent (95-100%)
```

### ScraperAgent `agents/scraper_agent.py`
Tool: `scraper_tools.py`

1. `_extract_topic(user_request)` — strips "Collect data about " prefix
2. `infer_topic_structure(topic, llm)` — LLM classifies topic type and returns exactly **3 subreddits** (best-first) + 4-6 keywords.
   - Prompt enforces: 1 dedicated sub + 1 category sub + 1 broader sub
   - Spec qualifiers (Pro Max, Ultra, Gen 2, S25+) preserved in every keyword
   - Returns `{subreddits, keywords, category, category_detail}`
3. **Fallback chain** — tries each source in order, moves to next if `comments_count == 0`:
   - **Primary — Arctic Shift (Reddit):** `scrape_with_api(subreddits, keywords)` scrolls posts via `before=<unix_ts>` cursor; local keyword filter; fetches comments per post. Stops at 500 comments, 24-week cap, or **10-minute wall-clock timeout** (`_SCRAPE_TIMEOUT_SECS=600`). Retry on 5xx and `JSONDecodeError` (up to 3 attempts, 8/16/24 s backoff). Returns `timed_out: bool` in result.
   - **Fallback 1 — YouTube Data API v3:** `scrape_with_youtube(topic, keywords)` — Phase 1: `/v3/search` (up to 4 pages × 50 = 200 videos); Phase 2: `/v3/commentThreads` per video (403 → skip). Maps comments to the same dict schema as Reddit scrape. Requires `YOUTUBE_API_KEY` in `.env`; raises `EnvironmentError` if absent.
   - **Fallback 2 — Local cache:** `find_local_dataset(topic)` scans `data/filtered_data/` for `combined_<safe_topic>*_filtered*.json` and returns the newest file by mtime. Loads comments directly from that file without re-running any API calls.
4. `export_to_csv(result, filepath)` — exports deduped CSV using a **versioned filename** so the same topic never overwrites a prior run.
5. `_save_filtered_data(result, topic)` — writes `combined_<topic>_filtered_N.json` (versioned), saves exact path to `shared_state["filtered_data_path"]`.

**`versioned_path(directory, base_name, ext)`** — helper that appends `_1`, `_2`, … to avoid filename collisions: `combined_iphone_filtered.json` → `combined_iphone_filtered_1.json` → …

**State written:** `raw_data`, `filtered_data_path`, `metadata` (includes `category`, `category_detail`, `data_source`)

`data_source` is one of `"reddit"` | `"youtube"` | `"local_cache"`.

API guardrails (Reddit): 1.5 s sleep per request · `limit=100` · single `link_id` per comment request · sticky posts skipped · 60 s per-request timeout

---

### ProcessorAgent `agents/processor_agent.py`
Tool: `processor_tools.py`

Reads `shared_state["filtered_data_path"]` (exact file, not glob — prevents mixing stale runs).

Per comment:
1. `remove_duplicates` by `text` field
2. Wordcloud preprocessing — spaCy POS tag + lemmatise, remove stopwords + topic noise (LLM-assisted)
3. BERTweet preprocessing — URL→`HTTPURL`, emoji→text, `@mention`→`@USER`, whitespace normalise; preserves case/punctuation
4. Processes one comment at a time to keep `(text, created_at)` aligned

**State written:** `processed_data`
```
processed_data[i].preprocessing.wordcloud.comments          # NOUN/PROPN/ADJ lemma texts (for overall WC)
processed_data[i].preprocessing.sentiment.comments          # BERTweet-preprocessed texts
processed_data[i].preprocessing.sentiment.comment_timestamps # unix ts, parallel to above
```

---

### AnalystAgent `agents/analyst_agent.py`
Tool: `analyst_tools.py`

**Sentiment:** `analyze_sentiment_batch(texts)` — fine-tuned BERTweet (`tools/models/bertweet_finetuned/`), GPU if available. `created_at` timestamps attached from parallel timestamp list. BERTweet is the **sole source of all sentiment labels** — both overall and per-aspect.

**ABSA** `discover_aspects_bertopic(texts, sentiments)`:
- Phase 1: clean text (strip r/, u/, URLs, punctuation, digits), extended `_SM_STOP` set (reddit noise: httpurl, lol, yeah, dont, etc.)
- Phase 2: `all-MiniLM-L6-v2` sentence embeddings (384D)
- Phase 3: `UMAP(n_neighbors=min(15,n-1), n_components=5, metric=cosine, random_state=42)`
- Phase 4: `HDBSCAN(min_cluster_size=max(5,n//20), min_samples=3, prediction_data=True)`
- Phase 5: `BERTopic` with `CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_features=5000)` — prevents function words from dominating c-TF-IDF topic keywords
- Cluster `-1` (HDBSCAN noise) → always labelled `"Others"`, never renamed, always last in output
- Named clusters: label = `words[0].title()` (best content keyword/bigram from c-TF-IDF)
- Falls back to K-Means+c-TF-IDF on `ImportError` (missing bertopic/umap-learn/hdbscan)

**LLM verification** `verify_aspects_with_llm(raw_aspects, topic, llm, category, category_detail)`:
- "Others" excluded before LLM call, re-appended at end
- Prompt sends `_top_words` keyword hints + type-specific examples (Consumer Electronics → Battery Life/Camera Quality; Software → UI/Bugs/Performance; Person → Music/Live Shows; Media → Plot/Acting; etc.)
- LLM outputs single **1-3 word Title Case** labels (no `&`, no long phrases)
- Duplicate labels merged (counts summed, percentages recalculated)

**State written:** `sentiment_results`, `aspect_analysis`

Also exports per-run JSON to `data/analysis/sentiment_<topic>.json` and `data/analysis/absa_<topic>.json`.

---

### AdvisorAgent `agents/advisor_agent.py`
Tool: `advisor_tools.py` — fully LLM-powered (Groq llama-3.3-70b-versatile).

Generates a per-section insight for every visualization tab, plus recommendations and suggested questions. All calls wrapped in `_safe_generate(fn, *args, fallback=...)` to avoid crashing the pipeline on any single LLM failure.

| Function | Input | Output |
|---|---|---|
| `generate_summary_insight` | sentiment_results | 2-3 bullets on overall distribution |
| `generate_timeline_insight` | detailed_sentiments (aggregated by day) | 2-3 bullets on trend/spikes |
| `generate_wordcloud_insight` | **word_frequencies** (from `get_top_words_by_sentiment`) | 2-3 bullets on actual word usage |
| `generate_absa_insight` | top-7 named aspects | 2-3 bullets on dominant aspects/sentiment patterns |
| `generate_absa_recommendations` | aspect_analysis + overall_sentiment | 4 consumer-framed takeaways |
| `generate_suggested_questions` | aspect_analysis | up to 6 consumer-framed questions |

All insight prompts instruct `**bold**` for key values; frontend `_style_insight()` converts `**bold**` → `<strong>` and applies green/red/grey colour spans for sentiment keywords.

**Recommendations labels:** `What People Love` / `Common Complaints` / `Mixed Opinions` / `Bottom Line` (consumer-framing, not business SWOT).

**Suggested questions:** aspect-specific phrasing varies by sentiment balance (≥45% neg → "Why are people unhappy with X?"; ≥45% pos → "What do people like about X?"; balanced → "What are people saying about X?"). Plus 5 topic-level consumer questions; max 6 total returned.

**Q&A:** `answer_question(question)` — LLM uses full shared-state context (sentiment dist, top aspects, section insights) to answer follow-up questions in 3-5 sentences.

**State written:** `advisor_insights`, `recommendations`, `suggested_questions`

---

### VisualizationAgent `agents/visualization_agent.py`
Tool: `visualization_tools.py`

Reads pre-computed `aspect_analysis` from shared state. Packages `visualization_data` dict including `topic`, `category`, `category_detail`, `sentiment_distribution`, `aspect_analysis`, `overall_sentiment`, `total_comments`, `subreddits`.

**State written:** `visuals_ready: true` (via visualization_data in results)

---

## Root Cause Analysis `tools/rca_tools.py`

Triggered on-demand from the frontend Timeline tab when the user clicks an anomaly date. Results are cached in the job record so repeated requests for the same date are instant.

### `detect_anomalies(detailed_sentiments, window_days=7, z_threshold=2.5)`
Rolling Z-score on daily negative-sentiment percentage. Returns anomaly dicts sorted by Z-score descending:
```
[{date, neg_count, total, neg_pct, z_score, mean_baseline}, ...]
```
Requires at least `window_days + 1` days of data. Uses pandas rolling mean/std.

### `get_spike_aspects(aspect_analysis, spike_date, detailed_sentiments, top_n=3)`
Weights each aspect by `(mentions on spike day among negative comments) × neg_pct`. Falls back to globally worst-negative aspects when day-level data is sparse. Excludes "Others" / "Other/Unspecified" cluster.

### `search_with_fallback(query, spike_date) → (snippets, source_label)`
Three-tier web search — silently skips any tier whose key is absent or is still a placeholder value:
1. **Tavily** (`TAVILY_API_KEY`) — `search_depth="basic"`, 5 results
2. **Serper** (`SERPER_API_KEY`) — `google.serper.dev/search`, 5 results
3. **Google Custom Search** (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`) — `/customsearch/v1`, 5 results

Returns `([], "none")` if all tiers fail or are unconfigured.

### `analyze_root_cause(snippets, aspects, topic, spike_date, llm)`
LLM prompt enforces exactly 3 output lines:
```
STATUS: MATCH | MISMATCH | UNCERTAIN
REASONING: <one sentence>
ROOT_CAUSE_SUMMARY: <one sentence for display>
```
Returns `{"status", "reasoning", "root_cause_summary"}`. Status `NO_DATA` when snippets list is empty; `ERROR` on LLM failure.

### `run_rca(topic, spike_date, aspect_analysis, detailed_sentiments, llm)`
Full orchestrator: `get_spike_aspects` → build search query → `search_with_fallback` → `analyze_root_cause`. Returns a flat dict suitable for caching:
```
{spike_date, aspects_analyzed, search_query, search_source, snippets_found,
 status, reasoning, root_cause_summary}
```

**API endpoint:** `POST /rca/{job_id}` body `{"date": "YYYY-MM-DD"}`. Validates job is completed, builds Groq LLM, runs `run_rca`, caches result in `db` under `rca_cache[spike_date]`, also writes to `shared_state["rca_cache"][spike_date]`.

---

## Shared State Schema `shared_state.json`
Inter-agent blackboard. All agents read/write via `BaseAgent.save_state(key, value)` / `load_state()`.

```
metadata
  .topic                    str
  .category                 str   # e.g. "Consumer Electronics"
  .category_detail          str   # e.g. "flagship smartphone by Apple"
  .subreddits_scraped       list[str]
  .keywords_used            list[str]
  .total_comments           int
  .data_source              str   # "reddit" | "youtube" | "local_cache"

filtered_data_path          str   # exact path to current job's filtered JSON

processed_data              list[dict]
  [i].subreddit             str
  [i].comments              list[{text, author, score, created_at, post_id}]
  [i].preprocessing
       .wordcloud.comments              list[str]  # NOUN/PROPN/ADJ lemma texts
       .sentiment.comments              list[str]  # BERTweet-preprocessed texts
       .sentiment.comment_timestamps    list[str]  # unix ts, parallel index

sentiment_results
  .overall_sentiment        "positive"|"neutral"|"negative"
  .confidence               float
  .sentiment_distribution   {positive,neutral,negative: {count, percentage}}
  .detailed_sentiments      list[{text, label, confidence, created_at}]

aspect_analysis             dict[label → {positive,neutral,negative:{count,percentage}, total_mentions}]
  # "Others" always present as last key (HDBSCAN noise cluster -1)

advisor_insights            dict
  .summary                  str   # bullet points
  .timeline                 str
  .wordcloud                str   # based on real word-frequency data
  .absa_insight             str
  .absa                     list[str]  # same as recommendations

recommendations             list[str]  # consumer-framed takeaways
suggested_questions         list[str]  # consumer-framed questions

rca_cache                   dict[YYYY-MM-DD → {spike_date, aspects_analyzed,
                                                search_query, search_source,
                                                snippets_found, status,
                                                reasoning, root_cause_summary}]
```

---

## Visualization Tools `tools/visualization_tools.py`

### `generate_timeline_chart(sentiments, use_timestamps=True, anomaly_dates=None)`
- Auto-selects hourly / daily / weekly grouping based on data span
- Y-axis uses `_nice_y_dtick()` → picks from `[1,2,5,10,20,25,50,100,…]` so ticks are always round numbers; top of range padded to next tick
- When `anomaly_dates` is provided: maps each date to its display period label and overlays red triangle-down markers (`marker_symbol="triangle-down"`) on the negative-sentiment line at each anomaly point

### `generate_aspect_sentiment_chart(aspect_analysis)`
- Top 9 named aspects + "Others" always appended last
- Muted bar colours for "Others"; footnote explaining cluster -1

### `generate_wordcloud_by_sentiment(sentiment_data, processed_data, topic, category_detail)`
- 9 POS-filtered wordclouds (positive/neutral/negative × noun/verb/adj) + 1 overall
- Dynamic topic stopwords via `_build_topic_stopwords(topic, category_detail)`:
  - Literal tokens from topic + category_detail strings
  - `_GENERIC_SELF_REF_WORDS` (`phone`, `smartphone`, `device`, `app`, `game`, …) added when they appear as a substring of any topic token (e.g. "iphone" → also stops "phone")
- spaCy for POS filtering; regex fallback if unavailable

### `get_top_words_by_sentiment(sentiment_data, processed_data, topic, category_detail, top_n=12)`
- Lightweight word-frequency counts (no image rendering) from `preprocessing.wordcloud` lemma texts
- Same dynamic stopwords as wordcloud generation
- Returns `{overall, positive, neutral, negative}: [(word, count), …]`
- Used by `AdvisorAgent` to feed genuine word-frequency data into the wordcloud LLM insight

---

## Frontend `frontend/app.py`

### Loading / Monitoring UI
- Job ID never shown to user; header shows topic name
- Progress bar + percentage only; started-at time in readable sentence
- After **12 minutes** elapsed: warning message + "Wait & Refresh" / "Cancel & Try Again" buttons
- Error jobs show the descriptive error message (e.g. "timed out", "no comments collected")

### Result Tabs

| Tab | Content |
|---|---|
| Summary | Topic (full text, not truncated), Total Comments, Data Source, Overall Sentiment; AI Advisor insight expander |
| Timeline | Plotly line chart (hourly/daily/weekly adaptive) with red triangle markers on anomaly dates; **RCA panel** below chart; AI Advisor insight expander |
| Wordcloud | Overall wordcloud + per-sentiment × per-POS filter; **cached** via `@st.cache_data` keyed on `job_id` so filter changes don't regenerate images; AI Advisor insight expander |
| Aspect Analysis | Plotly grouped bar; data table (top 9 named + Others always last); AI Advisor insight expander |
| Recommendations & Chat | Overview expanders (summary/timeline/wordcloud insights); Strategic Recommendations (consumer-framed); chat input; Suggested Questions shown **below** the chat input |

### Timeline Tab — RCA Panel
Located below the timeline chart. Runs `detect_anomalies` on the job's `detailed_sentiments`. If anomalies exist:
- Displays each anomaly date as a clickable button (Z-score + neg% shown)
- On click: `POST /rca/{job_id}` with `{"date": "YYYY-MM-DD"}`
- Displays verdict card: status badge (`MATCH` / `MISMATCH` / `UNCERTAIN` / `NO_DATA`), search source used, number of snippets, reasoning, root cause summary

### `_style_insight(text)` helper
Converts LLM output for all insight blocks:
- `**bold**` → `<strong>bold</strong>`
- Positive/strength keywords → green span
- Negative/weakness/concern keywords → red span
- Neutral/mixed keywords → grey span

Applied via `st.markdown(..., unsafe_allow_html=True)` in every advisor insight expander.

---

## Backend `backend/`

### `api.py` — port 8000
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | |
| POST | `/scrape/start` | returns `job_id` immediately; job runs in background thread |
| GET | `/scrape/status/{job_id}` | lightweight — no results payload |
| GET | `/scrape/results/{job_id}` | full results; call once on completion |
| GET | `/scrape/jobs` | list all |
| GET | `/scrape/jobs/{status}` | filter by status |
| DELETE | `/scrape/jobs/{job_id}` | |
| POST | `/rca/{job_id}` | body `{"date": "YYYY-MM-DD"}`; triggers RCA for that anomaly; result cached in job record |

Job fields: `id, topic, subreddits, status, progress, results, error, rca_cache, created_at, started_at, completed_at`
Persisted to `jobs_db.json` on every write (thread-safe lock).

### `tasks.py`
Runs the 5-agent pipeline sequentially in a background thread. All `print()` statements use plain ASCII prefixes (`[OK]`, `[ERROR]`, `[WARN]`, `[Step N]`) — no emoji, safe on Windows `charmap` encoding.

After scraping: validates `comments > 0`; if zero + `timed_out` flag → raises descriptive error ("timed out after 10 minutes"); if zero without timeout → raises "no comments collected" error. Both surface as `status="error"` with a human-readable message.

---

## Key Dependencies
| Package | Role |
|---|---|
| `fastapi` + `uvicorn` | REST API server |
| `streamlit` | web UI |
| `langchain-groq` | Groq llama-3.3-70b-versatile LLM (all LLM calls) |
| `transformers` + `torch` | fine-tuned BERTweet sentiment model (GPU if available) |
| `sentence-transformers` | all-MiniLM-L6-v2 embeddings for BERTopic ABSA |
| `bertopic` | topic modelling for ABSA aspect discovery |
| `umap-learn` | dimensionality reduction for BERTopic |
| `hdbscan` | density clustering for BERTopic (noise → "Others") |
| `scikit-learn` | CountVectorizer (c-TF-IDF in BERTopic), K-Means fallback |
| `spacy` + `en_core_web_sm` | POS tagging (wordcloud preprocessing + ABSA noun filter) |
| `requests` | Arctic Shift + Serper + Google CSE API calls |
| `google-api-python-client` | YouTube Data API v3 |
| `tavily-python` | Tavily web search (RCA tier 1) |
| `plotly` | interactive charts |
| `wordcloud` + `matplotlib` | wordcloud image generation |
| `pandas` + `numpy` | data manipulation, rolling Z-score anomaly detection |
| `python-dotenv` | `.env` / API key loading |

---

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Copy .env.example to .env and fill in your keys:
cp .env.example .env
```

Required keys in `.env`:
| Key | Required for |
|---|---|
| `GROQ_API_KEY` | All LLM calls (advisor insights, topic inference, ABSA label verify) |
| `HUGGINGFACE_API_KEY` | Downloading fine-tuned BERTweet model weights |
| `YOUTUBE_API_KEY` | YouTube scraper fallback (optional — skipped if absent) |
| `TAVILY_API_KEY` | RCA web search tier 1 (optional — skipped if absent) |
| `SERPER_API_KEY` | RCA web search tier 2 (optional — skipped if absent) |
| `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | RCA web search tier 3 (optional — skipped if absent) |

```bash
# Terminal 1 — backend:
python run_backend.py

# Terminal 2 — frontend:
streamlit run frontend/app.py
```
