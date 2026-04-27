"""
FastAPI backend for the Fake News Detector.

Endpoints
---------
  POST /analyze          – analyze text or URL
  GET  /history          – recent detections
  GET  /health           – service health check
"""
from __future__ import annotations

import logging
import sys
import os
import time
from typing import Optional

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from credibility.scorer import score_url, score_text_only
from explainability.explainer import explain
from fact_check.fact_api import check_claims
from fact_check.similarity import compute_similarity
from models.predict import predict
from processing.claim_extractor import extract_claims
from processing.feature_engineering import extract_features
from processing.preprocess import clean_text
from utils.config import MAX_TEXT_LENGTH
from utils.helpers import fetch_url_text, is_valid_url, save_detection, get_recent_detections

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake News Detector API",
    description="Detect misinformation using NLP, fact-checking, and source credibility analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: Optional[str]  = Field(None,  description="Raw news article text")
    url:  Optional[str]  = Field(None,  description="URL of the news article")
    use_bert: bool        = Field(False, description="Use BERT model (slower but more accurate)")


class AnalyzeResponse(BaseModel):
    prediction:        str
    confidence:        float
    credibility_score: float
    credibility_verdict: str
    explanation:       dict
    fact_check_results: list
    similarity_matches: list
    features:          dict
    model_used:        str
    processing_time_s: float
    input_text_preview: str


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(text: str, url: Optional[str], use_bert: bool) -> dict:
    t0 = time.perf_counter()

    # ── 1. Get text ────────────────────────────────────────────────────────
    source_url = url or ""
    if not text:
        if not url or not is_valid_url(url):
            raise HTTPException(status_code=400, detail="Provide 'text' or a valid 'url'.")
        text = fetch_url_text(url)
        if not text:
            raise HTTPException(status_code=422, detail="Could not fetch content from the provided URL.")

    text = text[:MAX_TEXT_LENGTH]

    # ── 2. Prediction ──────────────────────────────────────────────────────
    pred_result = predict(text, use_bert=use_bert)

    # ── 3. Features ────────────────────────────────────────────────────────
    features = extract_features(text)

    # ── 4. Claims ──────────────────────────────────────────────────────────
    claims = extract_claims(text)

    # ── 5. Fact-check ──────────────────────────────────────────────────────
    fc_results = check_claims(claims)

    # ── 6. Similarity ──────────────────────────────────────────────────────
    reference_texts = [
        r.get("description", "") or r.get("rebuttal", "") or r.get("claim", "")
        for r in fc_results
        if r.get("description") or r.get("rebuttal") or r.get("claim")
    ]
    sim_matches = compute_similarity(claims, reference_texts) if reference_texts else []

    # ── 7. Credibility ────────────────────────────────────────────────────
    cred = score_url(source_url) if source_url else score_text_only()

    # ── 8. Explanation ────────────────────────────────────────────────────
    expl = explain(text)

    elapsed = round(time.perf_counter() - t0, 3)

    result = {
        "prediction":           pred_result["prediction"],
        "confidence":           pred_result["confidence"],
        "credibility_score":    cred["score"],
        "credibility_verdict":  cred["verdict"],
        "explanation":          expl,
        "fact_check_results":   fc_results,
        "similarity_matches":   sim_matches,
        "features":             features,
        "model_used":           pred_result["model_used"],
        "processing_time_s":    elapsed,
        "input_text_preview":   clean_text(text)[:300],
        "input_text":           text,
        "source_url":           source_url,
        "credibility_detail":   cred,
    }
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyze news text or URL")
async def analyze(req: AnalyzeRequest):
    result = run_pipeline(req.text, req.url, req.use_bert)
    save_detection(result)
    return AnalyzeResponse(**{k: result[k] for k in AnalyzeResponse.__fields__})


@app.get("/history", summary="Recent analysis history")
async def history(limit: int = 20):
    return get_recent_detections(limit=limit)


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Dev entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
