"""Root Cause Analysis tools for timeline anomaly investigation."""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(
    detailed_sentiments: List[Dict],
    window_days: int = 7,
    z_threshold: float = 2.5,
) -> List[Dict]:
    """
    Rolling Z-score anomaly detection on daily negative sentiment percentage.

    Returns a list of dicts for days where negative% is unusually high:
        [{date, neg_count, total, neg_pct, z_score, mean_baseline}, ...]
    Requires at least window_days + 1 days of data.
    """
    try:
        import pandas as pd
        import numpy as np

        records = []
        for s in detailed_sentiments:
            ts = s.get("created_at")
            if not ts:
                continue
            try:
                dt = datetime.fromtimestamp(float(str(ts).strip()))
            except Exception:
                continue
            records.append({"date": dt.date(), "label": s.get("label", "neutral")})

        if not records:
            return []

        df = pd.DataFrame(records)
        daily = df.groupby(["date", "label"]).size().unstack(fill_value=0)
        for col in ["positive", "neutral", "negative"]:
            if col not in daily.columns:
                daily[col] = 0

        daily["total"]   = daily[["positive", "neutral", "negative"]].sum(axis=1)
        daily["neg_pct"] = (daily["negative"] / daily["total"].replace(0, 1) * 100).round(2)

        if len(daily) < window_days + 1:
            return []

        rolling_mean = daily["neg_pct"].rolling(window=window_days, min_periods=window_days).mean()
        rolling_std  = daily["neg_pct"].rolling(window=window_days, min_periods=window_days).std()

        anomalies = []
        for date, row in daily.iterrows():
            mean = rolling_mean.get(date)
            std  = rolling_std.get(date)
            if mean is None or std is None or std == 0:
                continue
            z = (row["neg_pct"] - mean) / std
            if z >= z_threshold:
                anomalies.append({
                    "date"         : str(date),
                    "neg_count"    : int(row["negative"]),
                    "total"        : int(row["total"]),
                    "neg_pct"      : round(float(row["neg_pct"]), 1),
                    "z_score"      : round(float(z), 2),
                    "mean_baseline": round(float(mean), 1),
                })

        return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)

    except Exception as e:
        print(f"detect_anomalies error: {str(e)[:100]}")
        return []


# ── Spike Aspect Extraction ───────────────────────────────────────────────────

def get_spike_aspects(
    aspect_analysis: Dict,
    spike_date: str,
    detailed_sentiments: List[Dict],
    top_n: int = 3,
) -> List[str]:
    """
    Return the top_n aspects most associated with negativity on spike_date.
    Falls back to globally worst-negative aspects when date-level data is sparse.
    """
    try:
        from datetime import datetime

        spike_dt = datetime.strptime(spike_date, "%Y-%m-%d").date()

        # Texts from the spike day
        spike_texts = set()
        for s in detailed_sentiments:
            ts = s.get("created_at")
            if not ts:
                continue
            try:
                dt = datetime.fromtimestamp(float(str(ts).strip())).date()
            except Exception:
                continue
            if dt == spike_dt and s.get("label") == "negative":
                spike_texts.add(s.get("text", "").lower())

        if spike_texts and aspect_analysis:
            scored = []
            for aspect, data in aspect_analysis.items():
                if aspect.lower() in ("others", "other/unspecified"):
                    continue
                mentions_on_day = sum(1 for t in spike_texts if aspect.lower() in t)
                neg_pct = data.get("negative", {}).get("percentage", 0)
                scored.append((aspect, mentions_on_day * neg_pct))
            scored.sort(key=lambda x: x[1], reverse=True)
            result = [a for a, _ in scored[:top_n] if scored[0][1] > 0]
            if result:
                return result

        # Fallback: global worst-negative aspects
        if aspect_analysis:
            by_neg = sorted(
                [(a, d.get("negative", {}).get("percentage", 0))
                 for a, d in aspect_analysis.items()
                 if a.lower() not in ("others", "other/unspecified")],
                key=lambda x: x[1], reverse=True,
            )
            return [a for a, _ in by_neg[:top_n]]

        return []

    except Exception as e:
        print(f"get_spike_aspects error: {str(e)[:80]}")
        return []


# ── Three-Tier Web Search ─────────────────────────────────────────────────────

def search_with_fallback(query: str, spike_date: str) -> Tuple[List[str], str]:
    """
    Try Tavily → Serper → Google CSE in order.
    Silently skips any tier whose API key is absent.
    Returns (list_of_snippets, source_label).
    """
    # ── Tier 1: Tavily ────────────────────────────────────────────────────────
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key and not tavily_key.startswith("your_"):
        try:
            from tavily import TavilyClient
            client   = TavilyClient(api_key=tavily_key)
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_answer=False,
            )
            snippets = [r.get("content", "") for r in response.get("results", []) if r.get("content")]
            if snippets:
                return snippets[:5], "Tavily"
        except Exception as e:
            print(f"Tavily search failed: {str(e)[:80]}")

    # ── Tier 2: Serper ────────────────────────────────────────────────────────
    serper_key = os.getenv("SERPER_API_KEY", "")
    if serper_key and not serper_key.startswith("your_"):
        try:
            import requests as _req
            resp = _req.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
                timeout=15,
            )
            if resp.status_code == 200:
                items    = resp.json().get("organic", [])
                snippets = [i.get("snippet", "") for i in items if i.get("snippet")]
                if snippets:
                    return snippets[:5], "Serper"
        except Exception as e:
            print(f"Serper search failed: {str(e)[:80]}")

    # ── Tier 3: Google Custom Search ─────────────────────────────────────────
    google_key = os.getenv("GOOGLE_API_KEY", "")
    cse_id     = os.getenv("GOOGLE_CSE_ID", "")
    if (google_key and not google_key.startswith("your_") and
            cse_id and not cse_id.startswith("your_")):
        try:
            import requests as _req
            resp = _req.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": google_key, "cx": cse_id, "q": query, "num": 5},
                timeout=15,
            )
            if resp.status_code == 200:
                items    = resp.json().get("items", [])
                snippets = [i.get("snippet", "") for i in items if i.get("snippet")]
                if snippets:
                    return snippets[:5], "Google CSE"
        except Exception as e:
            print(f"Google CSE search failed: {str(e)[:80]}")

    return [], "none"


# ── LLM Root Cause Evaluation ─────────────────────────────────────────────────

def analyze_root_cause(
    snippets: List[str],
    aspects: List[str],
    topic: str,
    spike_date: str,
    llm,
) -> Dict:
    """
    Ask the LLM whether the web snippets explain the negative spike.
    Returns dict with keys: status, reasoning, root_cause_summary.
    """
    if not snippets:
        return {
            "status"            : "NO_DATA",
            "reasoning"         : "No web search results were available.",
            "root_cause_summary": "Could not determine root cause — no external data found.",
        }

    context = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
    aspects_str = ", ".join(aspects) if aspects else "general sentiment"

    prompt = (
        f"On {spike_date}, Reddit discussions about '{topic}' showed an unusually high "
        f"spike in negative sentiment, particularly around these aspects: {aspects_str}.\n\n"
        f"Here are web search snippets from around that period:\n{context}\n\n"
        f"Based on these snippets, evaluate whether there is a plausible external cause "
        f"for the Reddit negativity spike.\n\n"
        f"Reply in EXACTLY 3 lines, no extra text:\n"
        f"STATUS: <one of: MATCH | MISMATCH | UNCERTAIN>\n"
        f"REASONING: <one sentence explaining your verdict>\n"
        f"ROOT_CAUSE_SUMMARY: <one sentence suitable for display to a general reader>\n"
    )

    try:
        r    = llm.invoke(prompt)
        text = r.content if hasattr(r, "content") else str(r)

        result = {"status": "UNCERTAIN", "reasoning": "", "root_cause_summary": ""}
        for line in text.strip().splitlines():
            if line.startswith("STATUS:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ("MATCH", "MISMATCH", "UNCERTAIN"):
                    result["status"] = val
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("ROOT_CAUSE_SUMMARY:"):
                result["root_cause_summary"] = line.split(":", 1)[1].strip()

        if not result["root_cause_summary"]:
            result["root_cause_summary"] = text[:200]

        return result

    except Exception as e:
        return {
            "status"            : "ERROR",
            "reasoning"         : str(e)[:120],
            "root_cause_summary": "LLM evaluation failed.",
        }


# ── Full RCA Orchestrator ─────────────────────────────────────────────────────

def run_rca(
    topic: str,
    spike_date: str,
    aspect_analysis: Dict,
    detailed_sentiments: List[Dict],
    llm,
) -> Dict:
    """
    Full RCA pipeline for one anomaly date.
    Returns a result dict suitable for caching in shared_state / session_state.
    """
    aspects = get_spike_aspects(aspect_analysis, spike_date, detailed_sentiments)

    aspect_terms = " ".join(aspects[:2]) if aspects else ""
    query = f"{topic} {aspect_terms} issue problem {spike_date[:7]}".strip()

    snippets, source = search_with_fallback(query, spike_date)

    verdict = analyze_root_cause(snippets, aspects, topic, spike_date, llm)

    return {
        "spike_date"        : spike_date,
        "aspects_analyzed"  : aspects,
        "search_query"      : query,
        "search_source"     : source,
        "snippets_found"    : len(snippets),
        "status"            : verdict["status"],
        "reasoning"         : verdict["reasoning"],
        "root_cause_summary": verdict["root_cause_summary"],
    }
