"""
Explanation engine.

Tries (in order):
  1. LIME  – local linear explanation on TF-IDF features
  2. SHAP  – TreeExplainer / LinearExplainer
  3. Rule-based – highlight emotional phrases + top TF-IDF words
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any

from processing.preprocess import preprocess, detect_emotional_tone
from utils.config import LR_MODEL_PATH, TFIDF_MODEL_PATH

logger = logging.getLogger(__name__)


# ── Helper: load saved models ─────────────────────────────────────────────────

def _load_models():
    if not (os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(LR_MODEL_PATH)):
        return None, None
    try:
        with open(TFIDF_MODEL_PATH, "rb") as f:
            vect = pickle.load(f)
        with open(LR_MODEL_PATH, "rb") as f:
            clf  = pickle.load(f)
        return vect, clf
    except Exception as exc:
        logger.warning("Model load error: %s", exc)
        return None, None


# ── LIME explanation ──────────────────────────────────────────────────────────

def _lime_explain(text: str, vectorizer, classifier) -> dict:
    try:
        from lime.lime_text import LimeTextExplainer

        def predict_fn(texts):
            processed = [preprocess(t) for t in texts]
            vecs      = vectorizer.transform(processed)
            return classifier.predict_proba(vecs)

        explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=10,
            num_samples=500,
        )
        # class 1 = FAKE
        lime_list = exp.as_list(label=1)
        top_features = [
            {"word": w, "weight": round(float(wt), 4)}
            for w, wt in lime_list
        ]
        return {
            "method":       "LIME",
            "top_features": top_features,
            "summary":      _features_to_summary(top_features),
        }
    except Exception as exc:
        logger.warning("LIME failed: %s", exc)
        return {}


# ── SHAP explanation ──────────────────────────────────────────────────────────

def _shap_explain(text: str, vectorizer, classifier) -> dict:
    try:
        import shap
        processed = preprocess(text)
        vec       = vectorizer.transform([processed])

        explainer  = shap.LinearExplainer(classifier, vec, feature_perturbation="interventional")
        shap_vals  = explainer.shap_values(vec)

        # shap_vals shape: (n_classes, n_features) or (n_samples, n_features)
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]   # class=FAKE
        else:
            vals = shap_vals[0]

        feature_names = vectorizer.get_feature_names_out()
        pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:10]
        top_features = [
            {"word": w, "weight": round(float(v), 4)}
            for w, v in pairs
        ]
        return {
            "method":       "SHAP",
            "top_features": top_features,
            "summary":      _features_to_summary(top_features),
        }
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)
        return {}


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_explain(text: str, vectorizer=None, classifier=None) -> dict:
    tone = detect_emotional_tone(text)
    top_features = [
        {"word": phrase, "weight": 0.8}
        for phrase in tone["emotional_phrases"][:10]
    ]

    # Add top TF-IDF words if model available
    if vectorizer and classifier:
        try:
            processed = preprocess(text)
            vec       = vectorizer.transform([processed]).toarray()[0]
            names     = vectorizer.get_feature_names_out()
            coefs     = classifier.coef_[0] if hasattr(classifier, "coef_") else []
            if len(coefs):
                scored = sorted(
                    zip(names, vec * coefs),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:8]
                for word, weight in scored:
                    if vec[list(names).index(word)] > 0:
                        top_features.append({"word": word, "weight": round(float(weight), 4)})
        except Exception:
            pass

    return {
        "method":           "rule_based",
        "top_features":     top_features[:10],
        "emotional_score":  tone["emotional_score"],
        "emotional_phrases": tone["emotional_phrases"],
        "summary":          _features_to_summary(top_features),
    }


# ── Summary formatter ─────────────────────────────────────────────────────────

def _features_to_summary(features: list[dict]) -> str:
    positive = [f["word"] for f in features if f.get("weight", 0) > 0][:5]
    negative = [f["word"] for f in features if f.get("weight", 0) < 0][:3]
    parts = []
    if positive:
        parts.append(f"Indicative of FAKE: {', '.join(positive)}.")
    if negative:
        parts.append(f"Indicative of REAL: {', '.join(negative)}.")
    return " ".join(parts) if parts else "No strong indicators found."


# ── Public API ────────────────────────────────────────────────────────────────

def explain(text: str) -> dict:
    """
    Generate an explanation dict for the given text.

    Returns
    -------
    {method, top_features, summary, emotional_phrases, emotional_score}
    """
    vectorizer, classifier = _load_models()

    # Try LIME first, then SHAP, then rule-based
    result: dict[str, Any] = {}

    if vectorizer and classifier:
        result = _lime_explain(text, vectorizer, classifier)
        if not result:
            result = _shap_explain(text, vectorizer, classifier)

    if not result:
        result = _rule_explain(text, vectorizer, classifier)

    # Always attach emotional tone
    tone = detect_emotional_tone(text)
    result.setdefault("emotional_phrases", tone["emotional_phrases"])
    result.setdefault("emotional_score",   tone["emotional_score"])

    return result
