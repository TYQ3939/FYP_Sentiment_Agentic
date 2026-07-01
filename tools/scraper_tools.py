"""
Reddit data collector using the Arctic Shift live API
(https://arctic-shift.photon-reddit.com).

CORE API RULES & SAFEGUARDS
────────────────────────────
1. Throttling      : time.sleep(1.5) after every single network request.
2. Chunk cap       : limit=100 on every API call (server maximum).
3. Cursor bypass   : use `before=<oldest_created_utc>` to scroll past 1 000-row
                     depth limit without re-querying the same rows.
4. Lucene format   : multi-word keywords are wrapped in double-quotes and joined
                     with ' OR ' (e.g. '"iphone 17 pro" OR "iphone 17 max"').
                     Applied as local filter on title+selftext because the
                     `/api/posts/search` endpoint rejects the `q` parameter.
5. ID chunk size   : max 20 post IDs per comment request to prevent HTTP 414.
6. Sticky filter   : skip any post whose stickied / pinned / is_pinned flag is
                     True to avoid infinite megathread comment loops.
"""

import json
import time
import requests
import pandas as pd
from datetime import datetime

# ─── API endpoints ─────────────────────────────────────────────────────────────
_API_BASE         = "https://arctic-shift.photon-reddit.com"
_POSTS_ENDPOINT   = f"{_API_BASE}/api/posts/search"
_COMMENTS_ENDPOINT = f"{_API_BASE}/api/comments/search"

# ─── Guardrail constants ───────────────────────────────────────────────────────
_SLEEP            = 1.5    # seconds between every request  (rule 1)
_MAX_LIMIT        = 100    # items per request               (rule 2)

# ─── Collection parameters ─────────────────────────────────────────────────────
_MIN_COMMENTS       = 500
_MAX_LOOKBACK_WEEKS = 24    # safety cap on outer loop
_POST_POOL_TARGET   = 50    # minimum matching posts before fetching comments
_SCRAPE_TIMEOUT_SECS = 600  # hard wall-clock limit for the whole scrape (10 min)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — TOPIC STRUCTURING (LLM ROUTER)
# ═══════════════════════════════════════════════════════════════════════════════

def infer_topic_structure(topic: str, llm) -> dict:
    """
    Two-step LLM routing:
      1. Classify what TYPE of thing the topic is (phone product, singer,
         politician, sports event, TV show, etc.).
      2. Use that classification to directly pick the 5 best subreddits for
         this topic, ordered from best (most active/relevant) to good.

    No downstream validation step — the LLM's own judgement of community
    activity is used directly so scraping can start immediately.

    Returns:
        {
          "subreddits"     : [...],   # 3 subreddits, no r/ prefix, best-first
          "keywords"       : [...],   # 4-6 search terms
          "category"       : str,     # e.g. "Consumer Electronics"
          "category_detail": str,     # e.g. "flagship smartphone by Apple"
        }
    """
    prompt = f"""You are a Reddit research expert specialising in community discovery.

USER INPUT: "{topic}"

─── STEP 1: UNDERSTAND & CLASSIFY ───────────────────────────────────────────
Identify the exact core subject and classify it into ONE of these categories:
  • Consumer Electronics     — smartphone, laptop, tablet, headphones, camera, smartwatch, gaming console
  • Software / App / Game    — mobile app, video game, PC software, OS, web platform, SaaS tool
  • Person – Celebrity       — singer, rapper, band, actor, actress, comedian, YouTube creator, influencer
  • Person – Public Figure   — politician, CEO, athlete, scientist, activist, business leader
  • Event                    — sports match/tournament, concert/festival, awards show, product launch, news event
  • Media Content            — movie, TV show, anime, book, podcast, streaming series
  • Company / Brand          — tech company, retailer, car brand, food brand, financial institution
  • Fashion / Lifestyle      — clothing brand, beauty product, food/drink brand, fitness trend
  • Concept / Trend          — financial concept (crypto/stock), social movement, technology trend, cultural phenomenon
  • Other                    — anything that does not fit above

─── STEP 2: GENERATE SUBREDDITS ─────────────────────────────────────────────
Think about WHO discusses this topic on Reddit and WHERE, then pick the
BEST 3 subreddits — no candidate list, no validation step, just your best
final answer.
Return EXACTLY 3 subreddit names (no "r/" prefix), ordered from BEST to GOOD
(start with the most active and most relevant, end with the least):
  • 1 subreddit DEDICATED to this exact subject (brand sub, fan sub, product sub, official community)
  • 1 CATEGORY subreddit (e.g. r/smartphones, r/Music, r/gaming, r/television)
  • 1 BROADER interest sub that regularly features this topic (r/technology, r/news, r/entertainment)
Only include subreddits you are confident are real, currently active, and have
100k+ subscribers. Prefer fewer, high-confidence subreddits over guessing.

─── STEP 3: KEYWORDS ────────────────────────────────────────────────────────
Return 4-6 lowercase keywords users type when searching for this topic.
Rules:
  1. ALWAYS include the exact topic string as the first keyword.
  2. Preserve version/tier qualifiers (Pro Max, Ultra, Gen 2, S25+, etc.) in EVERY keyword — never drop them.
  3. Include 1-2 informal abbreviations only if they keep the qualifier.
  4. The LAST keyword may be the broad parent term for fallback coverage.

─── OUTPUT ──────────────────────────────────────────────────────────────────
Return ONLY valid JSON (no markdown, no explanation):
{{
  "topic"           : "cleaned exact topic name",
  "category"        : "category from the list above",
  "category_detail" : "one-phrase description, e.g. \\"flagship smartphone by Apple\\" or \\"K-pop idol (BTS member)\\"",
  "subreddits"      : ["sub1", "sub2", "sub3"],
  "keywords"        : ["keyword1", ..., "keyword6"]
}}"""

    try:
        print(f"  LLM classifying and routing topic: '{topic}'...")
        response = llm.invoke(prompt)
        text     = response.content if hasattr(response, "content") else str(response)

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            parsed          = json.loads(text[start:end])
            subreddits      = parsed.get("subreddits",       [])
            keywords        = parsed.get("keywords",         [topic.lower()])
            category        = parsed.get("category",         "")
            category_detail = parsed.get("category_detail",  "")
            print(f"  Category   : {category} — {category_detail}")
            print(f"  Subreddits : {subreddits}")
            print(f"  Keywords   : {keywords}")
            return {
                "subreddits"      : subreddits,
                "keywords"        : keywords,
                "category"        : category,
                "category_detail" : category_detail,
            }

        print("  Could not parse LLM response — using fallback")

    except Exception as exc:
        print(f"  LLM error: {str(exc)[:150]}")

    return {
        "subreddits"      : ["technology", "gadgets", "reviews"],
        "keywords"        : [topic.lower()],
        "category"        : "",
        "category_detail" : "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _format_lucene_query(keywords: list) -> str:
    """
    Format keywords as a Lucene OR query string.
    Multi-word terms are wrapped in double-quotes.

    Example: ["iphone 17 pro", "iphone 17"] -> '"iphone 17 pro" OR "iphone 17"'
    """
    return " OR ".join(f'"{k}"' for k in keywords)


def _keyword_match(post: dict, keywords: list) -> bool:
    """
    Return True if any keyword appears (case-insensitive) in the post's
    combined title + selftext.  This is the local Lucene-equivalent filter
    (rule 4) applied after retrieving posts from the API.
    """
    text = (str(post.get("title",    "")) + " " +
            str(post.get("selftext", ""))).lower()
    return any(kw.lower() in text for kw in keywords)


def _is_stickied(post: dict) -> bool:
    """Return True for pinned/stickied posts that should be skipped (rule 6)."""
    return bool(
        post.get("stickied") or
        post.get("pinned")   or
        post.get("is_pinned")
    )


_REQUEST_TIMEOUT = 60   # seconds; Arctic Shift historical searches can be slow
_MAX_RETRIES     = 3    # retry count for transient timeouts / network hiccups


def _fetch_submissions(subreddit: str, before_ts: int = None) -> list:
    """
    GET /api/posts/search for one page of posts.

    Applies rule 1 (sleep) and rule 2 (limit=100).
    Uses `before` cursor for rule 3 pagination.

    Returns:
        List of post dicts, or [] if the API is genuinely empty or all
        retry attempts are exhausted.
    """
    params = {
        "subreddit": subreddit,
        "sort"     : "desc",
        "limit"    : _MAX_LIMIT,
    }
    if before_ts is not None:
        params["before"] = before_ts

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.get(_POSTS_ENDPOINT, params=params, timeout=_REQUEST_TIMEOUT)
            time.sleep(_SLEEP)  # rule 1 — always throttle

            if r.status_code == 200:
                return r.json().get("data") or []

            # Retry on 5xx server errors (Arctic Shift transient failures)
            if r.status_code >= 500:
                wait = attempt * 8
                print(f"    Posts API {r.status_code} (attempt {attempt}/{_MAX_RETRIES})"
                      f" — retrying in {wait}s...")
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    continue

            print(f"    Posts API {r.status_code}: {r.text[:120]}")
            return []

        except Exception as exc:
            exc_msg = str(exc).lower()
            is_transient = (
                "timed out"  in exc_msg
                or "timeout" in exc_msg
                or isinstance(exc, json.JSONDecodeError)
                or isinstance(exc, requests.exceptions.ChunkedEncodingError)
            )
            if is_transient:
                wait = attempt * 8
                print(f"    Posts response incomplete (attempt {attempt}/{_MAX_RETRIES})"
                      f" — retrying in {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(_SLEEP)
                print(f"    Posts request error: {str(exc)[:120]}")
                return []

    print(f"    Posts request failed after {_MAX_RETRIES} attempts — skipping page")
    return []


def _fetch_comments_chunk(link_id: str, before_ts: int = None) -> list:
    """
    GET /api/comments/search for one page of comments for a single post.

    The API accepts exactly one base36 link_id per request (e.g. 't3_abc123').
    Applies rule 1 (sleep) and rule 2 (limit=100).

    Returns:
        List of comment dicts, or [] if genuinely empty or all retries exhausted.
    """
    params = {
        "link_id": link_id,
        "sort"   : "desc",
        "limit"  : _MAX_LIMIT,
    }
    if before_ts is not None:
        params["before"] = before_ts

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.get(_COMMENTS_ENDPOINT, params=params, timeout=_REQUEST_TIMEOUT)
            time.sleep(_SLEEP)  # rule 1

            if r.status_code == 200:
                return r.json().get("data") or []

            # Retry on 5xx server errors
            if r.status_code >= 500:
                wait = attempt * 8
                print(f"    Comments API {r.status_code} (attempt {attempt}/{_MAX_RETRIES})"
                      f" — retrying in {wait}s...")
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    continue

            print(f"    Comments API {r.status_code}: {r.text[:120]}")
            return []

        except Exception as exc:
            exc_msg = str(exc).lower()
            is_transient = (
                "timed out"  in exc_msg
                or "timeout" in exc_msg
                or isinstance(exc, json.JSONDecodeError)
                or isinstance(exc, requests.exceptions.ChunkedEncodingError)
            )
            if is_transient:
                wait = attempt * 8
                print(f"    Comments response incomplete (attempt {attempt}/{_MAX_RETRIES})"
                      f" — retrying in {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(_SLEEP)
                print(f"    Comments request error: {str(exc)[:120]}")
                return []

    print(f"    Comments request failed after {_MAX_RETRIES} attempts — skipping post")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 + 3 — MAIN SCRAPING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_with_api(
    subreddits   : list,
    keywords     : list,
    min_comments : int = _MIN_COMMENTS,
) -> dict:
    """
    Collect Reddit comments by:
      1. Scanning submissions backwards in time per subreddit (local keyword
         filter applied after each 100-item page — rule 4).
      2. Batching matched post IDs into chunks of 20 (rule 5) and pulling all
         linked comments with cursor pagination (rule 3).

    Stops as soon as `min_comments` valid comments are collected or the
    MAX_LOOKBACK_WEEKS safety cap is hit.

    Returns:
        A result dict compatible with the downstream ProcessorAgent.
    """
    lucene_q     = _format_lucene_query(keywords)
    all_comments = []
    start_ts     = int(datetime.now().timestamp())
    deadline     = time.time() + _SCRAPE_TIMEOUT_SECS
    timed_out    = False

    print(f"\n{'='*60}")
    print(f"Arctic Shift API collection started")
    print(f"  Subreddits : {subreddits}")
    print(f"  Keywords   : {keywords}")
    print(f"  Lucene q   : {lucene_q}")
    print(f"  Target     : {min_comments} comments")
    print(f"  Timeout    : {_SCRAPE_TIMEOUT_SECS}s")
    print(f"{'='*60}")

    try:
        for subreddit in subreddits:
            if len(all_comments) >= min_comments:
                break
            if time.time() >= deadline:
                timed_out = True
                print(f"  [timeout] Wall-clock limit reached — stopping scrape early")
                break

            print(f"\n--- r/{subreddit} ---")

            # ── STEP 2: Build matching post ID pool ─────────────────────────
            matching_ids  = []
            post_cursor   = start_ts
            weeks_checked = 0

            while len(matching_ids) < _POST_POOL_TARGET and weeks_checked < _MAX_LOOKBACK_WEEKS:
                if time.time() >= deadline:
                    timed_out = True
                    print(f"  [timeout] Wall-clock limit reached during post scan — stopping")
                    break
                weeks_checked += 1
                posts = _fetch_submissions(subreddit, before_ts=post_cursor)

                if not posts:
                    print(f"  [posts] No more posts — stopping r/{subreddit} scan")
                    break

                matched_this_page = 0
                for post in posts:
                    if _is_stickied(post):
                        continue
                    if _keyword_match(post, keywords):
                        matching_ids.append("t3_" + post["id"])
                        matched_this_page += 1

                oldest_ts = min(p["created_utc"] for p in posts)
                print(f"  [posts] page {weeks_checked}: {len(posts)} fetched, "
                      f"{matched_this_page} matched | pool={len(matching_ids)} | "
                      f"cursor={oldest_ts}")

                post_cursor = oldest_ts

                if len(posts) < _MAX_LIMIT:
                    print(f"  [posts] Reached end of r/{subreddit} feed")
                    break

            if not matching_ids:
                print(f"  No matching posts found in r/{subreddit}")
                continue

            # ── STEP 3: Fetch comments for matched posts (one post at a time) ─
            print(f"\n  Fetching comments for {len(matching_ids)} matched posts...")

            done = False
            for post_num, link_id in enumerate(matching_ids, start=1):
                if done:
                    break
                if time.time() >= deadline:
                    timed_out = True
                    print(f"  [timeout] Wall-clock limit reached during comment fetch — stopping")
                    break

                comment_cursor = None
                page_num       = 0

                while True:
                    if time.time() >= deadline:
                        timed_out = True
                        print(f"  [timeout] Wall-clock limit reached — stopping comment pages")
                        done = True
                        break

                    page_num += 1
                    raw_comments = _fetch_comments_chunk(link_id, before_ts=comment_cursor)

                    if not raw_comments:
                        break

                    added = 0
                    for c in raw_comments:
                        body = c.get("body", "")
                        if body in ("[deleted]", "[removed]") or not body:
                            continue
                        all_comments.append({
                            "text"      : body,
                            "author"    : c.get("author",      ""),
                            "score"     : c.get("score",        0),
                            "created_at": str(c.get("created_utc", "")),
                            "post_id"   : c.get("link_id",     ""),
                        })
                        added += 1

                    print(f"  [comments] post {post_num}/{len(matching_ids)} "
                          f"({link_id}), page {page_num}: {added} added | "
                          f"total={len(all_comments)}")

                    if len(all_comments) >= min_comments:
                        print(f"\n  Target reached: {len(all_comments)} comments")
                        done = True
                        break

                    if len(raw_comments) < _MAX_LIMIT:
                        break

                    comment_cursor = min(c["created_utc"] for c in raw_comments)

    except KeyboardInterrupt:
        print(f"\n\n  [STOPPED] Ctrl+C received — saving {len(all_comments)} collected comments.")

    if timed_out:
        print(f"  [timeout] Scrape stopped after {_SCRAPE_TIMEOUT_SECS}s "
              f"— collected {len(all_comments)} comments.")
    elif len(all_comments) < min_comments and len(all_comments) > 0:
        print(f"  Collected {len(all_comments)}/{min_comments} comments before stopping.")

    return {
        "subreddit"     : "+".join(subreddits),
        "topic"         : "+".join(keywords),
        "posts"         : [],
        "comments"      : all_comments,
        "posts_count"   : 0,
        "comments_count": len(all_comments),
        "target_reached": len(all_comments) >= min_comments,
        "timed_out"     : timed_out,
        "scraped_at"    : datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — EXPORT UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_csv(result: dict, filepath: str) -> None:
    """
    Convert the scraper result's comments list to a deduplicated CSV file.

    Args:
        result   : dict returned by scrape_with_api
        filepath : destination CSV path
    """
    comments = result.get("comments", [])
    if not comments:
        print(f"  No comments to export")
        return

    df = pd.DataFrame(comments)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  Exported {len(df)} rows to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def versioned_path(directory: str, base_name: str, ext: str) -> str:
    """
    Return a file path that does not overwrite an existing file.
    If <directory>/<base_name><ext> exists, tries <base_name>_1<ext>,
    <base_name>_2<ext>, ... until a free name is found.

    Example:
        versioned_path("./data", "iphone_comments", ".csv")
        -> "./data/iphone_comments.csv"   (if it doesn't exist)
        -> "./data/iphone_comments_1.csv" (if the first does)
    """
    import os
    candidate = os.path.join(directory, f"{base_name}{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate = os.path.join(directory, f"{base_name}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def find_local_dataset(topic: str, data_dir: str = "./data/filtered_data"):
    """
    Search data_dir for an existing filtered dataset whose filename contains
    a sanitised version of the topic string (case-insensitive prefix match).

    Returns the path of the NEWEST matching file, or None.
    """
    import os, glob as _glob

    if not os.path.isdir(data_dir):
        return None

    safe = topic.replace(" ", "_").replace("/", "_").lower()[:40]
    pattern = os.path.join(data_dir, f"combined_{safe}*_filtered*.json")
    matches = _glob.glob(pattern)

    if not matches:
        # Wider case-insensitive scan
        for f in os.listdir(data_dir):
            if f.lower().endswith(".json") and safe[:10] in f.lower():
                matches.append(os.path.join(data_dir, f))

    if not matches:
        return None

    return max(matches, key=os.path.getmtime)


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE FALLBACK SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_with_youtube(
    topic: str,
    keywords: list,
    min_comments: int = _MIN_COMMENTS,
) -> dict:
    """
    Collect video comments from YouTube Data API v3 when Arctic Shift fails.

    Phase 1 — Search:  GET /v3/search  (up to 4 pages × 50 = 200 videos)
    Phase 2 — Comments: GET /v3/commentThreads per video (paginates until
                        min_comments reached or no more pages)

    Returns the same schema as scrape_with_api() so downstream agents need
    no changes.

    Raises EnvironmentError if YOUTUBE_API_KEY is absent.
    """
    import os, requests as _req

    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise EnvironmentError(
            "YOUTUBE_API_KEY not set — add it to your .env file to enable "
            "the YouTube fallback scraper."
        )

    SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
    COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
    SLEEP        = 0.5  # YouTube quota is generous; 0.5s is enough

    # ── Phase 1: collect video IDs ───────────────────────────────────────────
    query      = topic
    video_ids  = []
    page_token = None
    pages      = 0

    while pages < 4:
        params = {
            "key"        : api_key,
            "q"          : query,
            "part"       : "id",
            "type"       : "video",
            "maxResults" : 50,
            "relevanceLanguage": "en",
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            r = _req.get(SEARCH_URL, params=params, timeout=20)
            time.sleep(SLEEP)
            if r.status_code != 200:
                print(f"  YouTube search API {r.status_code}: {r.text[:120]}")
                break
            data = r.json()
            for item in data.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    video_ids.append(vid)
            page_token = data.get("nextPageToken")
            pages += 1
            if not page_token:
                break
        except Exception as e:
            print(f"  YouTube search error: {str(e)[:100]}")
            break

    print(f"  YouTube: found {len(video_ids)} videos for '{topic}'")

    if not video_ids:
        return {"comments": [], "comments_count": 0, "posts": [], "source": "youtube"}

    # ── Phase 2: fetch comments from each video ───────────────────────────────
    all_comments = []
    kw_lower     = [k.lower() for k in keywords]

    for vid_id in video_ids:
        if len(all_comments) >= min_comments:
            break

        page_token = None
        while True:
            params = {
                "key"       : api_key,
                "videoId"   : vid_id,
                "part"      : "snippet",
                "maxResults": 100,
                "textFormat": "plainText",
                "order"     : "relevance",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                r = _req.get(COMMENTS_URL, params=params, timeout=20)
                time.sleep(SLEEP)

                if r.status_code == 403:
                    break  # comments disabled on this video
                if r.status_code != 200:
                    break

                data = r.json()
                for item in data.get("items", []):
                    snip = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    text = snip.get("textDisplay", "").strip()
                    if not text:
                        continue

                    # Keyword relevance filter (same spirit as Reddit local filter)
                    text_lower = text.lower()
                    if kw_lower and not any(k in text_lower for k in kw_lower):
                        continue

                    published = snip.get("publishedAt", "")
                    try:
                        ts = datetime.strptime(published[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
                    except Exception:
                        ts = time.time()

                    all_comments.append({
                        "text"      : text,
                        "author"    : snip.get("authorDisplayName", ""),
                        "score"     : int(snip.get("likeCount", 0)),
                        "created_at": ts,
                        "post_id"   : vid_id,
                    })

                page_token = data.get("nextPageToken")
                if not page_token or len(all_comments) >= min_comments:
                    break

            except Exception as e:
                print(f"  YouTube comments error (video {vid_id}): {str(e)[:80]}")
                break

    print(f"  YouTube: collected {len(all_comments)} comments")

    return {
        "comments"      : all_comments,
        "comments_count": len(all_comments),
        "posts"         : [],
        "posts_count"   : 0,
        "source"        : "youtube",
        "target_reached": len(all_comments) >= min_comments,
        "scraped_at"    : datetime.now().isoformat(),
    }
