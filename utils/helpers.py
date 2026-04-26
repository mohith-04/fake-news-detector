"""
Shared utility helpers.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests

from utils.config import DB_PATH, LOG_LEVEL, REQUEST_TIMEOUT

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── URL helpers ───────────────────────────────────────────────────────────────

def extract_domain(url: str) -> str:
    """Return bare domain, e.g. 'https://www.bbc.co.uk/news' → 'bbc.co.uk'."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = parsed.netloc.lower().lstrip("www.")
        return domain
    except Exception:
        return ""


def fetch_url_text(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """Download a webpage and return its visible text (very lightweight)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (FakeNewsDetector/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # crude extraction – strip tags
        raw = resp.text
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:10_000]
    except Exception as exc:
        logger.warning("fetch_url_text failed for %s: %s", url, exc)
        return ""


def is_valid_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip()))


# ── Text helpers ──────────────────────────────────────────────────────────────

def truncate(text: str, max_len: int = 512) -> str:
    return text[:max_len] if len(text) > max_len else text


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── SQLite persistence ────────────────────────────────────────────────────────

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT,
                input_hash  TEXT,
                prediction  TEXT,
                confidence  REAL,
                credibility REAL,
                result_json TEXT
            )
            """
        )
        con.commit()
    logger.debug("DB initialised at %s", DB_PATH)


def save_detection(result: dict[str, Any]) -> None:
    try:
        init_db()
        import json

        rid = sha256(str(time.time()))
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT OR IGNORE INTO detections VALUES (?,?,?,?,?,?,?)",
                (
                    rid,
                    datetime.utcnow().isoformat(),
                    sha256(result.get("input_text", "")),
                    result.get("prediction", ""),
                    result.get("confidence", 0.0),
                    result.get("credibility_score", 0.0),
                    json.dumps(result),
                ),
            )
            con.commit()
    except Exception as exc:
        logger.warning("save_detection failed: %s", exc)


def get_recent_detections(limit: int = 20) -> list[dict]:
    try:
        import json

        init_db()
        with sqlite3.connect(DB_PATH) as con:
            rows = con.execute(
                "SELECT result_json FROM detections ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]
    except Exception as exc:
        logger.warning("get_recent_detections failed: %s", exc)
        return []
