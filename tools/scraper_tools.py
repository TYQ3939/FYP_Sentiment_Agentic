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
import os
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

# ─── API endpoints ─────────────────────────────────────────────────────────────
_API_BASE         = "https://arctic-shift.photon-reddit.com"
_POSTS_ENDPOINT   = f"{_API_BASE}/api/posts/search"
_COMMENTS_ENDPOINT = f"{_API_BASE}/api/comments/search"

# ─── Guardrail constants ───────────────────────────────────────────────────────
_SLEEP            = 1.5    # seconds between every request  (rule 1)
_MAX_LIMIT        = 100    # items per request               (rule 2)
_REQUEST_TIMEOUT  = 60     # seconds per request before giving up
_MAX_RETRIES      = 3      # retry attempts for transient errors (5xx, timeout)

# ─── Collection parameters ─────────────────────────────────────────────────────
_MIN_COMMENTS     = 500
_MAX_LOOKBACK_WEEKS = 24   # safety cap on outer loop
_POST_POOL_TARGET = 50     # minimum matching posts before fetching comments


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — TOPIC STRUCTURING (LLM ROUTER)
# ═══════════════════════════════════════════════════════════════════════════════

def infer_topic_structure(topic: str, llm) -> dict:
    """
    Ask the LLM to return the top 3 subreddits and 2-4 keyword variations for
    the given topic.

    Returns:
        {"subreddits": [...], "keywords": [...]}
    """
    prompt = f"""Given the topic: "{topic}"

Return ONLY a JSON object with this exact schema:
{{
    "subreddits": ["subreddit1", "subreddit2", "subreddit3"],
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"]
}}

Rules:
- subreddits: TOP 3 most relevant, active subreddits (no 'r/' prefix).
- keywords: 3-5 lowercase search terms real users type when discussing this topic.
  Follow these keyword rules strictly:
  1. ALWAYS include the full topic string (or its closest natural form) as the
     first keyword.
  2. If the topic contains a tier, version, edition, or spec qualifier
     (e.g. "Pro Max", "Ultra", "Plus", "5G", "Gen 2", "S24+", "v2"),
     EVERY keyword variation MUST preserve that qualifier — never drop it.
     Example for "iPhone 17 Pro Max":
       GOOD → ["iphone 17 pro max", "ip17 pro max", "iphone17 pro max", "apple iphone 17 pro max"]
       BAD  → ["iphone17", "ip17", "apple iphone"]  ← these lose "pro max"
  3. Include informal abbreviations ONLY if they still carry the full qualifier.
  4. Broad parent-brand keywords (e.g. just "iphone", "samsung") are acceptable
     as the LAST keyword only, for fallback coverage.

Output ONLY the JSON object — no explanation."""

    try:
        print(f"  LLM structuring topic: '{topic}'...")
        response = llm.invoke(prompt)
        text     = response.content if hasattr(response, "content") else str(response)

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            parsed     = json.loads(text[start:end])
            subreddits = parsed.get("subreddits", [])
            keywords   = parsed.get("keywords",   [topic.lower()])
            print(f"  Subreddits : {subreddits}")
            print(f"  Keywords   : {keywords}")
            return {"subreddits": subreddits, "keywords": keywords}

        print("  Could not parse LLM response — using fallback")

    except Exception as exc:
        print(f"  LLM error: {str(exc)[:150]}")

    return {
        "subreddits": ["iphone", "apple", "technology"],
        "keywords"  : [topic.lower()],
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


def _fetch_submissions(subreddit: str, before_ts: int = None) -> list:
    """
    GET /api/posts/search for one page of posts.

    Applies rule 1 (sleep) and rule 2 (limit=100).
    Uses `before` cursor for rule 3 pagination.

    Returns:
        List of post dicts, or [] on any error.
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
            time.sleep(_SLEEP)

            if r.status_code == 200:
                return r.json().get("data") or []

            if r.status_code >= 500:
                print(f"    Posts API {r.status_code} (attempt {attempt}/{_MAX_RETRIES}) — retrying in {attempt * 8}s...")
                if attempt < _MAX_RETRIES:
                    time.sleep(attempt * 8)
                    continue
            else:
                print(f"    Posts API {r.status_code}: {r.text[:120]}")
            return []

        except Exception as exc:
            exc_msg = str(exc).lower()
            is_transient = "timed out" in exc_msg or "timeout" in exc_msg
            print(f"    Posts request error (attempt {attempt}/{_MAX_RETRIES}): {str(exc)[:120]}")
            if is_transient and attempt < _MAX_RETRIES:
                time.sleep(attempt * 8)
                continue
            time.sleep(_SLEEP)
            return []

    return []


def _fetch_comments_chunk(link_id: str, before_ts: int = None) -> list:
    """
    GET /api/comments/search for one page of comments for a single post.

    The API accepts exactly one base36 link_id per request (e.g. 't3_abc123').
    Applies rule 1 (sleep) and rule 2 (limit=100).

    Returns:
        List of comment dicts, or [] on any error.
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
            time.sleep(_SLEEP)

            if r.status_code == 200:
                return r.json().get("data") or []

            if r.status_code >= 500:
                print(f"    Comments API {r.status_code} (attempt {attempt}/{_MAX_RETRIES}) — retrying in {attempt * 8}s...")
                if attempt < _MAX_RETRIES:
                    time.sleep(attempt * 8)
                    continue
            else:
                print(f"    Comments API {r.status_code}: {r.text[:120]}")
            return []

        except Exception as exc:
            exc_msg = str(exc).lower()
            is_transient = "timed out" in exc_msg or "timeout" in exc_msg
            print(f"    Comments request error (attempt {attempt}/{_MAX_RETRIES}): {str(exc)[:120]}")
            if is_transient and attempt < _MAX_RETRIES:
                time.sleep(attempt * 8)
                continue
            time.sleep(_SLEEP)
            return []

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

    print(f"\n{'='*60}")
    print(f"Arctic Shift API collection started")
    print(f"  Subreddits : {subreddits}")
    print(f"  Keywords   : {keywords}")
    print(f"  Lucene q   : {lucene_q}")
    print(f"  Target     : {min_comments} comments")
    print(f"  Press Ctrl+C at any time to stop and keep collected data.")
    print(f"{'='*60}")

    try:
        for subreddit in subreddits:
            if len(all_comments) >= min_comments:
                break

            print(f"\n--- r/{subreddit} ---")

            # ── STEP 2: Build matching post ID pool ─────────────────────────
            matching_ids  = []
            post_cursor   = start_ts
            weeks_checked = 0

            while len(matching_ids) < _POST_POOL_TARGET and weeks_checked < _MAX_LOOKBACK_WEEKS:
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

                comment_cursor = None
                page_num       = 0

                while True:
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

    if len(all_comments) < min_comments and len(all_comments) > 0:
        print(f"  Collected {len(all_comments)}/{min_comments} comments before stopping.")

    return {
        "subreddit"     : "+".join(subreddits),
        "topic"         : "+".join(keywords),
        "posts"         : [],
        "comments"      : all_comments,
        "posts_count"   : 0,
        "comments_count": len(all_comments),
        "target_reached": len(all_comments) >= min_comments,
        "scraped_at"    : datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK A — YOUTUBE API SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_with_youtube(topic: str, keywords: list, min_comments: int = _MIN_COMMENTS) -> dict:
    """
    Fallback data source using the YouTube Data API v3.

    Differences from Reddit / Arctic Shift:
    - Searches by topic directly (no subreddit routing needed).
    - Pagination uses pageToken, not a timestamp cursor.
    - Comments are all viewers of matching videos — no secondary keyword
      filter required since the search API already returns relevant videos.
    - Score field = likeCount (int) instead of Reddit karma.

    Requires YOUTUBE_API_KEY in the environment (.env).
    Raises EnvironmentError if the key is absent.
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "YOUTUBE_API_KEY not set — YouTube fallback unavailable. "
            "Add it to your .env file."
        )

    _YT_SEARCH   = "https://www.googleapis.com/youtube/v3/search"
    _YT_COMMENTS = "https://www.googleapis.com/youtube/v3/commentThreads"

    print(f"\n{'='*60}")
    print(f"YouTube API fallback started")
    print(f"  Topic    : {topic}")
    print(f"  Keywords : {keywords}")
    print(f"  Target   : {min_comments} comments")
    print(f"{'='*60}")

    # ── Phase 1: Search for relevant videos ──────────────────────────────────
    # Use the topic + first two keywords as the search query for better relevance.
    search_query = topic
    if keywords:
        extra = " ".join(keywords[:2])
        if extra.lower() not in topic.lower():
            search_query = f"{topic} {extra}"

    video_ids  = []
    page_token = None

    for page in range(1, 5):  # max 4 pages × 50 = 200 candidate videos
        params = {
            "key"              : api_key,
            "q"                : search_query,
            "type"             : "video",
            "part"             : "snippet",
            "maxResults"       : 50,
            "relevanceLanguage": "en",
            "order"            : "relevance",
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            r = requests.get(_YT_SEARCH, params=params, timeout=30)
            time.sleep(_SLEEP)

            if r.status_code != 200:
                print(f"  YouTube search API {r.status_code}: {r.text[:120]}")
                break

            data  = r.json()
            items = data.get("items", [])
            for item in items:
                vid_id = item.get("id", {}).get("videoId")
                if vid_id:
                    video_ids.append(vid_id)

            print(f"  [YT search] page {page}: {len(items)} videos | pool={len(video_ids)}")
            page_token = data.get("nextPageToken")
            if not page_token or len(video_ids) >= 150:
                break

        except Exception as e:
            print(f"  YouTube search error: {str(e)[:100]}")
            break

    if not video_ids:
        raise RuntimeError("YouTube search returned no videos for this topic")

    # ── Phase 2: Fetch comments from matching videos ──────────────────────────
    all_comments = []

    for vid_num, video_id in enumerate(video_ids, 1):
        if len(all_comments) >= min_comments:
            break

        page_token = None
        page_num   = 0

        while True:
            page_num += 1
            params = {
                "key"       : api_key,
                "videoId"   : video_id,
                "part"      : "snippet",
                "maxResults": 100,
                "order"     : "relevance",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                r = requests.get(_YT_COMMENTS, params=params, timeout=30)
                time.sleep(_SLEEP)

                if r.status_code == 403:
                    print(f"  [YT] video {vid_num}: comments disabled — skipping")
                    break

                if r.status_code != 200:
                    print(f"  [YT comments] {r.status_code}: {r.text[:80]}")
                    break

                data  = r.json()
                items = data.get("items", [])
                added = 0

                for item in items:
                    snip = (
                        item.get("snippet", {})
                            .get("topLevelComment", {})
                            .get("snippet", {})
                    )
                    text = snip.get("textDisplay", "").strip()
                    if not text:
                        continue

                    published = snip.get("publishedAt", "")
                    try:
                        ts = int(
                            datetime.strptime(published[:19], "%Y-%m-%dT%H:%M:%S")
                            .timestamp()
                        )
                    except Exception:
                        ts = int(datetime.now().timestamp())

                    all_comments.append({
                        "text"      : text,
                        "author"    : snip.get("authorDisplayName", ""),
                        "score"     : int(snip.get("likeCount", 0)),
                        "created_at": str(ts),
                        "post_id"   : video_id,
                    })
                    added += 1

                print(
                    f"  [YT] video {vid_num}/{len(video_ids)} ({video_id}), "
                    f"page {page_num}: {added} comments | total={len(all_comments)}"
                )

                if len(all_comments) >= min_comments:
                    print(f"  Target reached: {len(all_comments)} comments")
                    break

                page_token = data.get("nextPageToken")
                if not page_token or not items:
                    break

            except Exception as e:
                print(f"  [YT] video {vid_num} comment error: {str(e)[:100]}")
                break

    print(f"\n  YouTube complete: {len(all_comments)} comments from {len(video_ids)} videos")

    return {
        "subreddit"     : "YouTube",
        "topic"         : topic,
        "posts"         : [],
        "comments"      : all_comments,
        "posts_count"   : 0,
        "comments_count": len(all_comments),
        "target_reached": len(all_comments) >= min_comments,
        "scraped_at"    : datetime.now().isoformat(),
        "source"        : "youtube",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK B — LOCAL DATASET LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_dataset(topic: str, data_dir: str = "./data/filtered_data") -> Optional[str]:
    """
    Find the most recently modified filtered JSON file for a topic.

    Matches any file whose name starts with combined_<safe_topic>_filtered
    (including versioned copies like _filtered_1.json, _filtered_2.json …).
    Returns the full path of the newest match, or None if nothing found.
    """
    if not os.path.exists(data_dir):
        return None

    safe   = topic.replace(" ", "_").replace("/", "_")[:60]
    prefix = f"combined_{safe}_filtered".lower()

    matches = [
        fname for fname in os.listdir(data_dir)
        if fname.lower().startswith(prefix) and fname.endswith(".json")
    ]

    if not matches:
        return None

    matches.sort(
        key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
        reverse=True,
    )
    return os.path.join(data_dir, matches[0])


# ═══════════════════════════════════════════════════════════════════════════════
# FILE VERSIONING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def versioned_path(directory: str, base_name: str, ext: str) -> str:
    """
    Return a non-conflicting filepath in directory.

    If <directory>/<base_name><ext> already exists, tries
    <base_name>_1<ext>, <base_name>_2<ext>, … until a free slot is found.
    """
    candidate = os.path.join(directory, f"{base_name}{ext}")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = os.path.join(directory, f"{base_name}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


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
