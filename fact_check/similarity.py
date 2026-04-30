"""
Semantic similarity between extracted claims and verified source sentences.
Uses sentence-transformers (all-MiniLM-L6-v2) when available;
falls back to TF-IDF cosine similarity.
"""
from __future__ import annotations

import logging

import numpy as np

from utils.config import SENTENCE_MODEL, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

_st_model = None


def _load_st_model():
    global _st_model
    if _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(SENTENCE_MODEL)
        logger.info("SentenceTransformer loaded: %s", SENTENCE_MODEL)
    except Exception as exc:
        logger.warning("SentenceTransformer unavailable (%s) – using TF-IDF fallback", exc)
        _st_model = None
    return _st_model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tfidf_similarity(claims: list[str], references: list[str]) -> list[dict]:
    """Pure TF-IDF cosine fallback – no external deps."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_texts = claims + references
    if not all_texts:
        return []
    vect = TfidfVectorizer()
    try:
        mat = vect.fit_transform(all_texts).toarray()
    except Exception:
        return []

    claim_vecs = mat[: len(claims)]
    ref_vecs   = mat[len(claims) :]
    results = []
    for i, claim in enumerate(claims):
        best_score = 0.0
        best_ref   = ""
        for j, ref in enumerate(references):
            score = _cosine(claim_vecs[i], ref_vecs[j])
            if score > best_score:
                best_score = score
                best_ref   = ref
        results.append(
            {
                "claim":            claim,
                "matched_reference": best_ref,
                "similarity":       round(best_score, 4),
                "contradicted":     best_score >= SIMILARITY_THRESHOLD,
            }
        )
    return results


def compute_similarity(claims: list[str], references: list[str]) -> list[dict]:
    """
    Compare each claim against a list of reference sentences.

    Parameters
    ----------
    claims     : extracted claim sentences from the input article
    references : verified/fact-check sentences to compare against

    Returns
    -------
    List of dicts with claim, matched_reference, similarity (0-1), contradicted flag.
    """
    if not claims or not references:
        return []

    model = _load_st_model()
    if model is None:
        return _tfidf_similarity(claims, references)

    try:
        claim_embs = model.encode(claims, convert_to_numpy=True)
        ref_embs   = model.encode(references, convert_to_numpy=True)
        results = []
        for i, claim in enumerate(claims):
            sims = [_cosine(claim_embs[i], ref_embs[j]) for j in range(len(references))]
            best_idx   = int(np.argmax(sims))
            best_score = sims[best_idx]
            results.append(
                {
                    "claim":            claim,
                    "matched_reference": references[best_idx],
                    "similarity":       round(best_score, 4),
                    "contradicted":     best_score >= SIMILARITY_THRESHOLD,
                }
            )
        return results
    except Exception as exc:
        logger.warning("ST similarity error: %s – falling back to TF-IDF", exc)
        return _tfidf_similarity(claims, references)
