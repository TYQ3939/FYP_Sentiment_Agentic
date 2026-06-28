# Multi-Agent Reddit Sentiment Analysis System

## System
Python multi-agent pipeline that scrapes Reddit comments for any topic, runs BERTweet sentiment analysis, discovers aspects via K-Means+c-TF-IDF, and displays results in a Streamlit UI backed by FastAPI.

Two processes run simultaneously:
- **Backend** `python run_backend.py` → FastAPI on `http://127.0.0.1:8000`
- **Frontend** `streamlit run frontend/app.py` → Streamlit on `http://localhost:8501`

Frontend never runs agents directly. It submits a topic via REST, polls job progress, then fetches results.

## Directory
```
sentiment_agentic/
├── agents/
│   ├── base_agent.py          # superclass: LLM init, shared_state.json I/O
│   ├── scraper_agent.py       # Reddit scraping via Arctic Shift API
│   ├── processor_agent.py     # dedup, wordcloud/sentiment preprocessing
│   ├── analyst_agent.py       # BERTweet inference + ABSA pipeline
│   ├── advisor_agent.py       # rule-based recommendations + Q&A
│   └── visualization_agent.py # reads pre-computed state, packages viz data
├── tools/
│   ├── scraper_tools.py       # API calls, keyword/subreddit inference
│   ├── processor_tools.py     # text cleaning, dedup, BERTweet preprocessing
│   ├── analyst_tools.py       # BERTweet batch, K-Means ABSA, LLM label verify
│   ├── advisor_tools.py       # recommendation + question generation
│   ├── visualization_tools.py # Plotly charts, wordcloud
│   └── common_tools.py        # shared utilities
├── backend/
│   ├── api.py                 # FastAPI routes
│   ├── tasks.py               # background job orchestrator
│   └── database.py            # thread-safe job store → jobs_db.json
├── frontend/
│   └── app.py                 # Streamlit UI
├── data/filtered_data/        # ScraperAgent output / ProcessorAgent input
├── shared_state.json          # inter-agent blackboard
├── jobs_db.json               # job progress persistence
├── run_backend.py             # uvicorn entry point
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
2. `infer_topic_structure(topic, llm)` — LLM returns subreddits + keywords. Prompt enforces spec qualifiers (Pro Max, Ultra, Gen 2) are preserved in every keyword variation.
3. `scrape_with_api(subreddits, keywords)` — scrolls posts backwards via `before=<unix_ts>` cursor; local keyword filter on title+selftext; fetches comments per post (one `link_id` at a time). Stops at 500 comments or 24-week cap. Ctrl+C saves collected data.
4. `export_to_csv(result, filepath)` — deduped CSV to `data/filtered_data/`
5. `_save_filtered_data(result, topic)` — writes `combined_<topic>_filtered.json`, saves exact path to `shared_state["filtered_data_path"]`

**State written:** `raw_data`, `filtered_data_path`, `metadata`

API guardrails: 1.5 s sleep per request · `limit=100` · single `link_id` per comment request · sticky posts skipped

### ProcessorAgent `agents/processor_agent.py`
Tool: `processor_tools.py`

Reads `shared_state["filtered_data_path"]` (exact file, not glob — prevents mixing stale runs).

Per comment:
1. `remove_duplicates` by `text` field
2. Wordcloud preprocessing — spaCy POS tag + lemmatise, remove stopwords + topic noise (LLM-assisted)
3. BERTweet preprocessing — URL→`HTTPURL`, emoji→text, `@mention`→`@USER`, whitespace normalise; preserves case/punctuation
4. Processes one comment at a time to keep `(text, created_at)` aligned — batch preprocessing would break alignment when short texts are filtered out

**State written:** `processed_data`
```
processed_data[i].preprocessing.sentiment.comments          # list[str]
processed_data[i].preprocessing.sentiment.comment_timestamps # list[str] unix ts, parallel to above
```

### AnalystAgent `agents/analyst_agent.py`
Tool: `analyst_tools.py`

**Sentiment:** `analyze_sentiment_batch(texts)` — fine-tuned BERTweet, GPU if available, batch inference. `created_at` timestamps attached to each result from the parallel timestamp list.

**ABSA** `discover_aspects_kmeans_ctfidf(texts, sentiments)`:
- Phase 1: clean (strip r/, u/, URLs), drop < 4 tokens
- Phase 2: `all-MiniLM-L6-v2` sentence embeddings (384D); falls back to TF-IDF if `sentence-transformers` missing
- Phase 3: PCA → 5D
- Phase 4: K-Means K=2..6, pick highest silhouette score
- Phase 5: concatenate texts per cluster into macro-documents
- Phase 6: c-TF-IDF (sklearn `TfidfVectorizer`) on macro-documents
- Phase 7: top-2 noun tokens (spaCy POS filtered) → `"Token1 & Token2"` label

**LLM verification** `verify_aspects_with_llm(raw_aspects, topic, llm)`:
- Sends raw labels + topic to Groq LLM
- LLM removes noise, polishes to `"Noun & Noun"` Title Case relevant to topic
- Duplicate new labels merged (counts summed, percentages recalculated)

**State written:** `sentiment_results`, `aspect_analysis`

### AdvisorAgent `agents/advisor_agent.py`
Tool: `advisor_tools.py` — no LLM required.

- `generate_aspect_recommendations(aspect_analysis, topic, overall_sentiment, min_mentions=3)` — rule-based, highlights high-negative/high-positive aspects
- `generate_suggested_questions(aspect_analysis, topic)` — question strings from aspect labels
- `answer_question(question)` — keyword-matched data-driven answers (positive/negative/overall/aspect queries)

**State written:** `recommendations`, `suggested_questions`

### VisualizationAgent `agents/visualization_agent.py`
Tool: `visualization_tools.py`

Reads pre-computed `aspect_analysis` from shared state (computed by AnalystAgent — does NOT recompute). Packages `visualization_data` dict for frontend consumption.

**State written:** `visuals_ready: true`

## Shared State Schema `shared_state.json`
Inter-agent blackboard. All agents read/write via `BaseAgent.save_state(key, value)` / `load_state()`.

```
metadata
  .topic                    str
  .subreddits_scraped       list[str]
  .keywords_used            list[str]
  .total_comments           int

filtered_data_path          str  # exact path to current job's filtered JSON

processed_data              list[dict]
  [i].subreddit             str
  [i].comments              list[{text, author, score, created_at, post_id}]
  [i].preprocessing
       .wordcloud.comments  list[str]
       .sentiment.comments              list[str]   # preprocessed texts
       .sentiment.comment_timestamps    list[str]   # unix ts, parallel index

sentiment_results
  .overall_sentiment        "positive"|"neutral"|"negative"
  .confidence               float
  .sentiment_distribution   {positive,neutral,negative: {count, percentage}}
  .detailed_sentiments      list[{text, label, confidence, created_at}]

aspect_analysis             dict[label → {positive,neutral,negative: {count,percentage}, total_mentions}]
recommendations             list[str]
suggested_questions         list[str]
visuals_ready               bool
```

## Data File `data/filtered_data/combined_<topic>_filtered.json`
```
{subreddit, topic, posts[], comments[{text, author, score, created_at, post_id}],
 metadata{posts_count, comments_count, scraped_at, target_reached}}
```
`<topic>` = `topic.replace(" ","_").replace("/","_")[:60]`

## Backend API `backend/api.py` — port 8000
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | |
| POST | `/scrape/start` | returns `job_id` immediately; job runs in background thread |
| GET | `/scrape/status/{job_id}` | lightweight — no results payload |
| GET | `/scrape/results/{job_id}` | full results; call once on completion |
| GET | `/scrape/jobs` | list all |
| GET | `/scrape/jobs/{status}` | filter by status |
| DELETE | `/scrape/jobs/{job_id}` | |

Job fields: `id, topic, subreddits, status, progress, results, error, created_at, started_at, completed_at`
Persisted to `jobs_db.json` on every write (thread-safe lock).

## Frontend `frontend/app.py`
- Sidebar: session history, API health indicator
- Main: topic input → polling loop (every 5 s `GET /scrape/status`) → on completion fetch `GET /scrape/results` → render tabs

| Tab | Content |
|---|---|
| Overview | overall sentiment, confidence, stats |
| Timeline | Plotly line chart: daily pos/neu/neg counts (unix ts parsed from `created_at`) |
| Aspects | Plotly grouped bar: sentiment per ABSA aspect |
| Wordcloud | matplotlib wordcloud per sentiment |
| Recommendations | per-aspect text recommendations |
| Q&A | keyword-matched answers from AdvisorAgent |

## Key Dependencies
| Package | Role |
|---|---|
| `fastapi` + `uvicorn` | REST API server |
| `streamlit` | web UI |
| `langchain-groq` | Groq Llama-3.3-70b-versatile LLM |
| `transformers` + `torch` | fine-tuned BERTweet sentiment model |
| `sentence-transformers` | all-MiniLM-L6-v2 for ABSA embeddings |
| `scikit-learn` | KMeans, PCA, silhouette, TfidfVectorizer |
| `spacy` + `en_core_web_sm` | POS tagging (wordcloud preprocessing + ABSA noun filter) |
| `requests` | Arctic Shift API calls |
| `plotly` | interactive charts |
| `wordcloud` + `matplotlib` | wordcloud images |
| `pandas` | data manipulation, CSV export |
| `python-dotenv` | `.env` / `GROQ_API_KEY` loading |

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo GROQ_API_KEY=your_key > .env
# terminal 1:
python run_backend.py
# terminal 2:
streamlit run frontend/app.py
```
