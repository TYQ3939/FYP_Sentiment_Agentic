"""Root Cause Analysis tools: anomaly detection, multi-tier web search, LLM evaluation."""

import os
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ── 1. Anomaly Detection (Rolling Z-Score) ────────────────────────────────────

def detect_anomalies(
    detailed_sentiments: List[Dict],
    window_days: int = 7,
    z_threshold: float = 2.5,
) -> List[Dict]:
    """
    Compute daily negative-sentiment counts and flag days that exceed
    z_threshold standard deviations above the rolling window mean.

    Requires at least 14 days of data to produce meaningful results.
    Returns list of dicts sorted ascending by date:
      {date, neg_count, total, neg_pct, z_score, mean_baseline}
    """
    if not detailed_sentiments:
        return []

    try:
        import pandas as pd

        records = []
        for s in detailed_sentiments:
            ts = s.get("created_at")
            if not ts:
                continue
            try:
                dt = datetime.fromtimestamp(float(str(ts).strip()))
            except Exception:
                continue
            records.append({"dt": dt, "label": s.get("label", "neutral")})

        if len(records) < 14:
            return []

        df = pd.DataFrame(records)
        df["date"] = df["dt"].dt.date

        daily = df.groupby(["date", "label"]).size().unstack(fill_value=0)
        for col in ["positive", "neutral", "negative"]:
            if col not in daily.columns:
                daily[col] = 0

        daily["total"] = daily["positive"] + daily["neutral"] + daily["negative"]
        daily["neg_pct"] = (daily["negative"] / daily["total"].replace(0, 1)) * 100
        daily = daily.sort_index()

        anomalies = []
        dates = daily.index.tolist()

        for i, date in enumerate(dates):
            window_start = max(0, i - window_days)
            window_vals = daily.iloc[window_start:i]["neg_pct"].tolist()

            if len(window_vals) < 3:
                continue

            mean = statistics.mean(window_vals)
            try:
                std = statistics.stdev(window_vals)
            except statistics.StatisticsError:
                continue

            if std == 0:
                continue

            current_pct = float(daily.loc[date, "neg_pct"])
            z = (current_pct - mean) / std

            if z >= z_threshold:
                anomalies.append({
                    "date": str(date),
                    "neg_count": int(daily.loc[date, "negative"]),
                    "total": int(daily.loc[date, "total"]),
                    "neg_pct": round(current_pct, 1),
                    "z_score": round(z, 2),
                    "mean_baseline": round(mean, 1),
                })

        return anomalies

    except Exception as e:
        print(f"RCA detect_anomalies error: {e}")
        return []


# ── 2. Spike Aspect Extraction ────────────────────────────────────────────────

def get_spike_aspects(
    aspect_analysis: Dict,
    spike_date: str,
    detailed_sentiments: List[Dict],
    top_n: int = 3,
) -> List[str]:
    """
    Return the top N aspects most active on spike_date, weighted by neg%.
    Falls back to globally worst-negative aspects if date data is thin.
    """
    def _global_worst():
        return [
            a for a, _ in sorted(
                [(a, d) for a, d in aspect_analysis.items() if a.lower() != "others"],
                key=lambda x: x[1].get("negative", {}).get("percentage", 0),
                reverse=True,
            )[:top_n]
        ]

    if not aspect_analysis:
        return []

    try:
        spike_texts = []
        for s in detailed_sentiments:
            ts = s.get("created_at")
            if not ts:
                continue
            try:
                dt = datetime.fromtimestamp(float(str(ts).strip()))
            except Exception:
                continue
            if str(dt.date()) == spike_date:
                spike_texts.append(s.get("text", "").lower())

        if not spike_texts:
            return _global_worst()

        hits = {}
        for aspect in aspect_analysis:
            if aspect.lower() == "others":
                continue
            count = sum(1 for t in spike_texts if aspect.lower() in t)
            neg_pct = aspect_analysis[aspect].get("negative", {}).get("percentage", 0)
            hits[aspect] = count * neg_pct

        top = [a for a, _ in sorted(hits.items(), key=lambda x: x[1], reverse=True) if hits[a] > 0]
        return top[:top_n] if top else _global_worst()

    except Exception as e:
        print(f"RCA get_spike_aspects error: {e}")
        return _global_worst()


# ── 3. Web Search Tiers ───────────────────────────────────────────────────────

def _date_window(spike_date: str) -> Tuple[str, str]:
    """Return (date_from, date_to) covering spike_date ± 2 days."""
    dt = datetime.strptime(spike_date, "%Y-%m-%d")
    return (
        (dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        (dt + timedelta(days=2)).strftime("%Y-%m-%d"),
    )


def search_tavily(query: str, date_from: str, date_to: str) -> List[str]:
    """Tier 1: Tavily AI search. Raises RuntimeError on failure."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY not set")
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        result = client.search(query=query, search_depth="basic", max_results=5)
        return [r["content"][:400] for r in result.get("results", []) if r.get("content")]
    except Exception as e:
        raise RuntimeError(f"Tavily: {e}")


def search_serper(query: str, date_from: str, date_to: str) -> List[str]:
    """Tier 2: Serper Google search. Raises RuntimeError on failure."""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("SERPER_API_KEY not set")
    import requests as _r
    resp = _r.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": 5, "dateRange": "custom",
              "startDate": date_from, "endDate": date_to},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return [item["snippet"][:400] for item in resp.json().get("organic", []) if item.get("snippet")]


def search_google_cse(query: str, date_from: str, date_to: str) -> List[str]:
    """Tier 3: Google Custom Search API. Raises RuntimeError on failure."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    cse_id  = os.getenv("GOOGLE_CSE_ID", "")
    if not api_key or not cse_id:
        raise EnvironmentError("GOOGLE_API_KEY or GOOGLE_CSE_ID not set")
    import requests as _r
    resp = _r.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cse_id, "q": query,
                "num": 5, "dateRestrict": "d3"},
        timeout=15,
    )
    resp.raise_for_status()
    return [item["snippet"][:400] for item in resp.json().get("items", []) if item.get("snippet")]


def search_with_fallback(query: str, spike_date: str) -> Tuple[List[str], str]:
    """
    Try Tier 1 (Tavily) → 2 (Serper) → 3 (Google CSE).
    Returns (snippets, tier_name).  Both empty if all tiers fail or lack keys.
    """
    date_from, date_to = _date_window(spike_date)
    for fn, name in [
        (search_tavily,     "Tavily"),
        (search_serper,     "Serper"),
        (search_google_cse, "Google CSE"),
    ]:
        try:
            snippets = fn(query, date_from, date_to)
            if snippets:
                print(f"RCA search: {name} returned {len(snippets)} results")
                return snippets, name
        except EnvironmentError:
            print(f"RCA search: {name} skipped (API key not configured)")
            continue
        except Exception as e:
            print(f"RCA search: {name} failed ({str(e)[:80]}), trying next tier")
            continue
    return [], "none"


# ── 4. LLM Root Cause Evaluation ─────────────────────────────────────────────

def analyze_root_cause(
    snippets: List[str],
    aspects: List[str],
    topic: str,
    spike_date: str,
    llm,
) -> Dict:
    """
    Pass web snippets + spike aspects to LLM for structured root cause verdict.
    Returns {status, reasoning, root_cause_summary}.
    """
    if not snippets:
        return {
            "status": "NO_DATA",
            "reasoning": "No web search results were found for this date window.",
            "root_cause_summary": "Unable to identify a root cause — no external news found.",
        }

    aspects_str = ", ".join(aspects) if aspects else "general topic"
    snippets_text = "\n---\n".join(snippets[:5])

    prompt = (
        f'Reddit sentiment about "{topic}" showed an unusually large negative spike on {spike_date}.\n'
        f"The top discussion aspects on that day were: {aspects_str}\n\n"
        f"Web/news snippets from around that date:\n{snippets_text}\n\n"
        f"Evaluate whether the external events above explain the Reddit negative spike for those aspects.\n\n"
        f"Respond in EXACTLY this 3-line format, nothing else:\n"
        f"STATUS: MATCH or MISMATCH or UNCERTAIN\n"
        f"REASONING: [1-2 sentences explaining the connection or lack of it]\n"
        f"ROOT_CAUSE_SUMMARY: [1 sentence — the single most likely cause of the spike]"
    )

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        result = {"status": "UNCERTAIN", "reasoning": "", "root_cause_summary": ""}
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("STATUS:"):
                val = line[7:].strip().upper()
                if val in ("MATCH", "MISMATCH", "UNCERTAIN"):
                    result["status"] = val
            elif line.startswith("REASONING:"):
                result["reasoning"] = line[10:].strip()
            elif line.startswith("ROOT_CAUSE_SUMMARY:"):
                result["root_cause_summary"] = line[19:].strip()
        return result
    except Exception as e:
        return {
            "status": "ERROR",
            "reasoning": f"LLM evaluation failed: {str(e)[:120]}",
            "root_cause_summary": "Root cause analysis could not complete.",
        }


# ── 5. Full Pipeline Orchestrator ─────────────────────────────────────────────

def run_rca(
    topic: str,
    spike_date: str,
    aspect_analysis: Dict,
    detailed_sentiments: List[Dict],
    llm,
) -> Dict:
    """
    Full RCA pipeline for one anomaly date:
      1. Extract worst aspects on that date
      2. Build targeted search query (topic + aspects)
      3. Search web (Tavily → Serper → Google CSE)
      4. LLM evaluates cause match
    Returns complete result dict.
    """
    aspects = get_spike_aspects(aspect_analysis, spike_date, detailed_sentiments)

    if aspects:
        aspect_clause = " OR ".join(f'"{a}"' for a in aspects[:3])
        query = f'"{topic}" AND ({aspect_clause})'
    else:
        query = f'"{topic}" news issues problems'

    snippets, source = search_with_fallback(query, spike_date)
    verdict = analyze_root_cause(snippets, aspects, topic, spike_date, llm)

    return {
        "spike_date": spike_date,
        "aspects_analyzed": aspects,
        "search_query": query,
        "search_source": source,
        "snippets_found": len(snippets),
        **verdict,
    }
