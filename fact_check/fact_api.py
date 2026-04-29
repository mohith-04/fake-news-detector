"""
Fact-checking via:
  • Google Fact Check Tools API
  • NewsAPI (headline search)
  • Fallback: keyword match against known misinformation patterns
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from utils.config import GOOGLE_FACT_CHECK_API_KEY, NEWS_API_KEY, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

GOOGLE_FC_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWS_API_URL  = "https://newsapi.org/v2/everything"


# ── Google Fact Check ─────────────────────────────────────────────────────────

def google_fact_check(query: str) -> list[dict]:
    """Query the Google Fact Check Tools API for a claim."""
    if not GOOGLE_FACT_CHECK_API_KEY:
        return []
    try:
        resp = requests.get(
            GOOGLE_FC_URL,
            params={"key": GOOGLE_FACT_CHECK_API_KEY, "query": query[:200], "pageSize": 5},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("claims", []):
            for review in item.get("claimReview", []):
                results.append(
                    {
                        "claim":    item.get("text", ""),
                        "rating":   review.get("textualRating", ""),
                        "url":      review.get("url", ""),
                        "reviewer": review.get("publisher", {}).get("name", ""),
                    }
                )
        return results
    except Exception as exc:
        logger.warning("Google Fact Check error: %s", exc)
        return []


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def newsapi_search(query: str, max_results: int = 5) -> list[dict]:
    """Search recent headlines related to the claim."""
    if not NEWS_API_KEY:
        return []
    try:
        resp = requests.get(
            NEWS_API_URL,
            params={
                "apiKey":    NEWS_API_KEY,
                "q":         query[:100],
                "pageSize":  max_results,
                "sortBy":    "relevancy",
                "language":  "en",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "url":         a.get("url", ""),
                "source":      a.get("source", {}).get("name", ""),
            }
            for a in articles
        ]
    except Exception as exc:
        logger.warning("NewsAPI error: %s", exc)
        return []


# ── Heuristic keyword fact-check fallback ─────────────────────────────────────

_KNOWN_FALSE_PATTERNS = {
    "microchip": "No credible evidence that vaccines contain microchips.",
    "5g": "5G networks do not spread viruses – this is a debunked claim.",
    "moon landing": "The Apollo moon landings are thoroughly documented and verified.",
    "flat earth": "Earth is an oblate spheroid – confirmed by independent scientific measurement.",
    "fluoride mind control": "Water fluoridation is safe; no evidence of mind control effects.",
    "climate hoax": "Anthropogenic climate change is supported by overwhelming scientific consensus.",
    "cure for cancer": "No single cure exists; oncology research continues in many directions.",
    "deep state": "The 'deep state' as a conspiratorial cabal has no credible evidentiary basis.",
}


def heuristic_fact_check(text: str) -> list[dict]:
    text_lower = text.lower()
    results = []
    for keyword, rebuttal in _KNOWN_FALSE_PATTERNS.items():
        if keyword in text_lower:
            results.append(
                {
                    "claim":    keyword,
                    "rating":   "False / Misleading",
                    "rebuttal": rebuttal,
                    "reviewer": "Internal Knowledge Base",
                }
            )
    return results


# ── Aggregated entry point ────────────────────────────────────────────────────

def check_claims(claims: list[str]) -> list[dict]:
    """
    For each claim, query available fact-check sources.
    Returns a flat list of findings.
    """
    all_findings: list[dict] = []
    for claim in claims:
        # Google
        gfc = google_fact_check(claim)
        all_findings.extend(gfc)
        # NewsAPI
        news = newsapi_search(claim, max_results=3)
        all_findings.extend(news)
        # Heuristic (always)
        heur = heuristic_fact_check(claim)
        all_findings.extend(heur)

    # Deduplicate by rating text
    seen = set()
    unique = []
    for f in all_findings:
        key = (f.get("claim", ""), f.get("rating", ""), f.get("reviewer", ""))
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique[:10]   # cap at 10 findings
