"""
Inference layer.

Priority chain:
  1. BERT/DistilBERT (if transformers available + GPU/CPU acceptable)
  2. TF-IDF + Logistic Regression (always available after train.py)
  3. Heuristic rule-based fallback (zero-setup)
"""
from __future__ import annotations

import logging
import os
import pickle
from processing.preprocess import preprocess, detect_emotional_tone
from utils.config import (
    BERT_MODEL_NAME,
    FAKE_THRESHOLD,
    LR_MODEL_PATH,
    TFIDF_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# ── BERT (optional) ───────────────────────────────────────────────────────────
_bert_pipeline = None

def _load_bert():
    global _bert_pipeline
    if _bert_pipeline is not None:
        return _bert_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading BERT pipeline (%s) …", BERT_MODEL_NAME)
        _bert_pipeline = hf_pipeline(
            "text-classification",
            model=BERT_MODEL_NAME,
            truncation=True,
            max_length=512,
        )
        logger.info("BERT pipeline ready.")
    except Exception as exc:
        logger.warning("BERT unavailable: %s", exc)
        _bert_pipeline = None
    return _bert_pipeline


# ── TF-IDF + LR ───────────────────────────────────────────────────────────────
_vectorizer = None
_classifier  = None

def _load_lr_model():
    global _vectorizer, _classifier
    if _vectorizer and _classifier:
        return True
    if not (os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(LR_MODEL_PATH)):
        logger.info("Saved LR model not found – running train.py …")
        try:
            from models.train import train
            train()
        except Exception as exc:
            logger.warning("Auto-training failed: %s", exc)
            return False
    try:
        with open(TFIDF_MODEL_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
        with open(LR_MODEL_PATH, "rb") as f:
            _classifier = pickle.load(f)
        logger.info("LR model loaded.")
        return True
    except Exception as exc:
        logger.warning("LR model load error: %s", exc)
        return False


# ── Heuristic fallback ────────────────────────────────────────────────────────
def _heuristic_predict(text: str) -> dict:
    """Very simple rule-based fake scorer when no ML model is available."""
    tone   = detect_emotional_tone(text)
    score  = tone["emotional_score"]
    caps   = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl   = text.count("!") / max(len(text.split()), 1)
    total  = min(score * 0.6 + caps * 0.2 + excl * 10 * 0.2, 1.0)
    label  = "FAKE" if total >= FAKE_THRESHOLD else "REAL"
    return {
        "prediction":  label,
        "confidence":  round(total if label == "FAKE" else 1 - total, 4),
        "model_used":  "heuristic",
        "raw_scores":  {"fake": round(total, 4), "real": round(1 - total, 4)},
    }

# ── Public API ────────────────────────────────────────────────────────────────

def predict(text: str, use_bert: bool = False) -> dict:
    """
    Returns a prediction dict:
      {prediction, confidence, model_used, raw_scores}
    """
    processed = preprocess(text)

    # 1. BERT
    if use_bert:
        pipe = _load_bert()
        if pipe:
            try:
                result = pipe(processed[:512])[0]
                label  = result["label"].upper()          # model-specific
                score  = float(result["score"])
                # normalise label to FAKE/REAL
                if label not in {"FAKE", "REAL"}:
                    label = "FAKE" if score >= FAKE_THRESHOLD else "REAL"
                return {
                    "prediction": label,
                    "confidence": round(score, 4),
                    "model_used": BERT_MODEL_NAME,
                    "raw_scores": {"bert_score": round(score, 4)},
                }
            except Exception as exc:
                logger.warning("BERT inference error: %s", exc)

    # 2. LR
    if _load_lr_model():
        try:
            vec   = _vectorizer.transform([processed])          # type: ignore
            proba = _classifier.predict_proba(vec)[0]           # type: ignore
            # class order: [0=REAL, 1=FAKE]
            classes = list(_classifier.classes_)                # type: ignore
            fake_idx = classes.index(1) if 1 in classes else 1
            real_idx = classes.index(0) if 0 in classes else 0
            fake_prob = float(proba[fake_idx])
            real_prob = float(proba[real_idx])
            label = "FAKE" if fake_prob >= FAKE_THRESHOLD else "REAL"
            return {
                "prediction": label,
                "confidence": round(max(fake_prob, real_prob), 4),
                "model_used": "tfidf_logistic_regression",
                "raw_scores": {
                    "fake": round(fake_prob, 4),
                    "real": round(real_prob, 4),
                },
            }
        except Exception as exc:
            logger.warning("LR inference error: %s", exc)
    # 3. Heuristic
    logger.info("Falling back to heuristic predictor.")
    return _heuristic_predict(text)
