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

    try:
        r = requests.get(_POSTS_ENDPOINT, params=params, timeout=20)
        time.sleep(_SLEEP)  # rule 1 — always throttle

        if r.status_code == 200:
            return r.json().get("data") or []

        print(f"    Posts API {r.status_code}: {r.text[:120]}")
        return []

    except Exception as exc:
        time.sleep(_SLEEP)
        print(f"    Posts request error: {str(exc)[:120]}")
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

    try:
        r = requests.get(_COMMENTS_ENDPOINT, params=params, timeout=20)
        time.sleep(_SLEEP)  # rule 1

        if r.status_code == 200:
            return r.json().get("data") or []

        print(f"    Comments API {r.status_code}: {r.text[:120]}")
        return []

    except Exception as exc:
        time.sleep(_SLEEP)
        print(f"    Comments request error: {str(exc)[:120]}")
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
