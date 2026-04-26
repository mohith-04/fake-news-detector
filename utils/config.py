"""
Central configuration for Fake News Detector.
"""
import os
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
NEWS_API_KEY               = os.getenv("NEWS_API_KEY", "")
WHOIS_API_KEY              = os.getenv("WHOIS_API_KEY", "")

# ── Model settings ────────────────────────────────────────────────────────────
TFIDF_MODEL_PATH   = "models/saved_model/tfidf_vectorizer.pkl"
LR_MODEL_PATH      = "models/saved_model/lr_classifier.pkl"
BERT_MODEL_NAME    = "distilbert-base-uncased"          # swap for roberta-base
SENTENCE_MODEL     = "all-MiniLM-L6-v2"

# ── Thresholds ────────────────────────────────────────────────────────────────
FAKE_THRESHOLD      = 0.55     # probability above → "FAKE"
SIMILARITY_THRESHOLD = 0.60    # cosine sim for claim matching
MAX_CLAIMS          = 5        # max claims extracted per article

# ── Source scoring weights ────────────────────────────────────────────────────
WEIGHT_HTTPS        = 0.15
WEIGHT_DOMAIN_AGE   = 0.20
WEIGHT_KNOWN_FAKE   = 0.40
WEIGHT_KNOWN_TRUST  = 0.25

# ── Known trusted & fake domains ─────────────────────────────────────────────
TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "theguardian.com", "npr.org", "washingtonpost.com",
    "economist.com", "ft.com", "bloomberg.com", "wsj.com",
    "snopes.com", "factcheck.org", "politifact.com",
}

FAKE_DOMAINS = {
    "beforeitsnews.com", "naturalnews.com", "infowars.com",
    "worldnewsdailyreport.com", "empirenews.net", "thelastlineofdefense.org",
    "abcnews.com.co", "huzlers.com", "newslo.com", "nationalreport.net",
    "theonion.com",  # satire
}

# ── Misc ──────────────────────────────────────────────────────────────────────
MAX_TEXT_LENGTH  = 10_000
REQUEST_TIMEOUT  = 10   # seconds for external HTTP calls
DB_PATH          = "data/detections.sqlite"
LOG_LEVEL        = "INFO"
