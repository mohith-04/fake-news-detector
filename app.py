"""
Streamlit frontend for the Fake News Detector.
Calls the FastAPI backend at localhost:8000.

Run:
    # Terminal 1 – start API
    uvicorn api.main:app --reload

    # Terminal 2 – start UI
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
import os

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .main { background: #0d1117; color: #e6edf3; }
    .stTextArea textarea { background: #161b22; color: #e6edf3; border-radius: 8px; }
    .stTextInput input  { background: #161b22; color: #e6edf3; border-radius: 8px; }

    .verdict-fake {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white; border-radius: 12px; padding: 20px 30px;
        font-size: 2rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 24px rgba(255,68,68,0.35);
    }
    .verdict-real {
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
        color: white; border-radius: 12px; padding: 20px 30px;
        font-size: 2rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 24px rgba(0,200,81,0.35);
    }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #e6edf3; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .tag {
        display: inline-block; background: #1f2937; color: #f59e0b;
        border: 1px solid #f59e0b; border-radius: 20px;
        padding: 3px 12px; margin: 3px; font-size: 0.78rem;
    }
    .section-title {
        color: #58a6ff; font-weight: 700; font-size: 1.05rem;
        border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    use_bert = st.toggle("Use BERT model (slower)", value=False)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This tool combines:\n"
        "- **NLP classification** (TF-IDF / BERT)\n"
        "- **Fact-checking** APIs\n"
        "- **Source credibility** scoring\n"
        "- **Explainability** (LIME / SHAP)\n"
    )
    st.markdown("---")
    if st.button("📋 View Recent History"):
        st.session_state["show_history"] = True


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#58a6ff; font-weight:800; font-size:2.5rem; margin-bottom:0;'>"
    "🔍 Fake News Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#8b949e; margin-top:4px;'>"
    "AI-powered misinformation detection with NLP, fact-checking &amp; source credibility analysis</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Input ─────────────────────────────────────────────────────────────────────
tab_text, tab_url = st.tabs(["📝 Paste Text", "🌐 Enter URL"])

with tab_text:
    news_text = st.text_area(
        "Paste news article text:",
        height=220,
        placeholder="Paste any news article, social media post, or claim here…",
    )

with tab_url:
    news_url = st.text_input("Enter article URL:", placeholder="https://example.com/article")

analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)


# ── Helper: call API (or run in-process as fallback) ─────────────────────────

def call_api(text: str | None, url: str | None, use_bert: bool) -> dict:
    """Try FastAPI first, fall back to running pipeline directly."""
    payload = {"text": text or None, "url": url or None, "use_bert": use_bert}
    try:
        resp = requests.post(f"{API_BASE}/analyze", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Run in-process when the API server is not running
        sys.path.insert(0, ".")
        from api.main import run_pipeline
        return run_pipeline(text, url, use_bert)


# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    text_input = news_text.strip() if news_text else None
    url_input  = news_url.strip()  if news_url  else None

    if not text_input and not url_input:
        st.warning("Please paste article text or enter a URL.")
        st.stop()

    with st.spinner("🧠 Analysing… this may take a moment"):
        try:
            result = call_api(text_input, url_input, use_bert)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # ── Verdict banner ────────────────────────────────────────────────────
    pred  = result.get("prediction", "UNKNOWN")
    conf  = result.get("confidence", 0.0)
    cred  = result.get("credibility_score", 0.0)
    model = result.get("model_used", "—")
    t     = result.get("processing_time_s", 0.0)

    cls = "verdict-fake" if pred == "FAKE" else "verdict-real"
    emoji = "🚨" if pred == "FAKE" else "✅"

    st.markdown(
        f"<div class='{cls}'>{emoji} Prediction: {pred} ({conf*100:.1f}% confidence)</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric row ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Confidence</div>"
            f"<div class='metric-value'>{conf*100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        cred_color = "#00c851" if cred >= 70 else "#ff9800" if cred >= 40 else "#ff4444"
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Credibility Score</div>"
            f"<div class='metric-value' style='color:{cred_color}'>{cred:.0f}/100</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Model</div>"
            f"<div class='metric-value' style='font-size:0.95rem'>{model.replace('_', ' ').title()}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Processing Time</div>"
            f"<div class='metric-value'>{t:.2f}s</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column details ────────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        # Explanation
        expl = result.get("explanation", {})
        st.markdown("<div class='section-title'>🧠 Explanation</div>", unsafe_allow_html=True)

        method = expl.get("method", "rule_based")
        st.caption(f"Method: **{method.upper()}**")

        summary = expl.get("summary", "")
        if summary:
            st.info(summary)

        emotional_phrases = expl.get("emotional_phrases", [])
        if emotional_phrases:
            st.markdown("**⚠️ Emotional / clickbait phrases detected:**")
            tags = " ".join(f"<span class='tag'>{p}</span>" for p in emotional_phrases)
            st.markdown(tags, unsafe_allow_html=True)

        top_features = expl.get("top_features", [])
        if top_features:
            st.markdown("**📊 Top indicative words:**")
            for feat in top_features[:8]:
                word   = feat.get("word", "")
                weight = feat.get("weight", 0.0)
                bar_w  = min(int(abs(weight) * 100), 100)
                color  = "#ff4444" if weight > 0 else "#00c851"
                st.markdown(
                    f"<div style='margin:4px 0'>"
                    f"<span style='display:inline-block;width:130px;font-size:0.85rem'>{word}</span>"
                    f"<span style='display:inline-block;width:{bar_w}px;height:12px;"
                    f"background:{color};border-radius:3px;vertical-align:middle'></span>"
                    f"<span style='color:{color};font-size:0.78rem;margin-left:6px'>{weight:+.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Credibility breakdown
        cred_detail = result.get("credibility_detail", {})
        st.markdown("<div class='section-title'>🌐 Source Credibility</div>", unsafe_allow_html=True)
        verdict = cred_detail.get("verdict", "UNKNOWN")
        domain  = cred_detail.get("domain", "unknown")
        https   = "✅ HTTPS" if cred_detail.get("is_https") else "⚠️ HTTP only"
        st.markdown(f"**Domain:** `{domain}` &nbsp; {https}  \n**Verdict:** {verdict}")
        breakdown = cred_detail.get("breakdown", {})
        if breakdown:
            for key, val in breakdown.items():
                label = key.replace("_", " ").title()
                col_v = "#00c851" if val > 0 else "#ff4444" if val < 0 else "#8b949e"
                st.markdown(
                    f"<div style='font-size:0.82rem'><span style='color:#8b949e;width:180px;"
                    f"display:inline-block'>{label}</span>"
                    f"<span style='color:{col_v}'>{val:+.1f} pts</span></div>",
                    unsafe_allow_html=True,
                )

    with right:
        # Fact-check results
        fc = result.get("fact_check_results", [])
        st.markdown("<div class='section-title'>📋 Fact-Check Findings</div>", unsafe_allow_html=True)
        if fc:
            for item in fc[:5]:
                rating   = item.get("rating",   "—")
                reviewer = item.get("reviewer", "—")
                claim    = item.get("claim", item.get("title", ""))[:120]
                rebuttal = item.get("rebuttal", "")
                url_link = item.get("url", "")

                color = "#ff4444" if any(w in rating.lower() for w in ["false", "mislead", "fake"]) else \
                        "#00c851" if any(w in rating.lower() for w in ["true", "correct", "verified"]) else "#ff9800"

                st.markdown(
                    f"<div style='background:#161b22;border-left:3px solid {color};"
                    f"border-radius:6px;padding:10px;margin:6px 0;'>"
                    f"<div style='font-size:0.78rem;color:#8b949e'>{reviewer}</div>"
                    f"<div style='font-size:0.85rem;margin:4px 0'>{claim}</div>"
                    f"<div style='color:{color};font-weight:600;font-size:0.82rem'>{rating}</div>"
                    + (f"<div style='font-size:0.78rem;color:#8b949e;margin-top:4px'>{rebuttal}</div>" if rebuttal else "")
                    + (f"<a href='{url_link}' target='_blank' style='font-size:0.75rem;color:#58a6ff'>🔗 Source</a>" if url_link else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No specific fact-check results found for this article.")

        # Similarity matches
        sim = result.get("similarity_matches", [])
        st.markdown("<div class='section-title'>🔗 Semantic Similarity Matches</div>", unsafe_allow_html=True)
        if sim:
            for m in sim[:4]:
                score_v = m.get("similarity", 0)
                contra  = m.get("contradicted", False)
                bar_pct = int(score_v * 100)
                color   = "#58a6ff"
                st.markdown(
                    f"<div style='background:#161b22;border-radius:6px;padding:10px;margin:6px 0'>"
                    f"<div style='font-size:0.78rem;color:#8b949e;margin-bottom:4px'>Claim similarity: "
                    f"<span style='color:{color};font-weight:700'>{score_v:.2%}</span></div>"
                    f"<div style='height:6px;background:#21262d;border-radius:3px'>"
                    f"<div style='height:6px;width:{bar_pct}%;background:{color};border-radius:3px'></div></div>"
                    f"<div style='font-size:0.8rem;margin-top:6px;color:#e6edf3'>"
                    f"{m.get('claim','')[:100]}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No semantic similarity matches computed.")

    # ── Text features ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📊 Text Analysis Features</div>", unsafe_allow_html=True)
    feats = result.get("features", {})
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Word count",        feats.get("word_count", "—"))
    fc2.metric("Caps ratio",        f"{feats.get('caps_ratio', 0):.1%}")
    fc3.metric("Exclamations",      feats.get("exclamation_count", "—"))
    fc4.metric("Lexical diversity", f"{feats.get('lexical_diversity', 0):.2f}")

    # ── Raw JSON expander ─────────────────────────────────────────────────
    with st.expander("🗂 Raw JSON Result"):
        st.json(result)


# ── History view ──────────────────────────────────────────────────────────────
if st.session_state.get("show_history"):
    st.markdown("---")
    st.subheader("📋 Recent Analysis History")
    try:
        hist = requests.get(f"{API_BASE}/history", timeout=10).json()
    except Exception:
        try:
            sys.path.insert(0, ".")
            from utils.helpers import get_recent_detections
            hist = get_recent_detections(20)
        except Exception:
            hist = []

    if hist:
        for item in hist:
            pred   = item.get("prediction", "?")
            conf   = item.get("confidence", 0)
            prev   = item.get("input_text_preview", "")[:80]
            color  = "#ff4444" if pred == "FAKE" else "#00c851"
            st.markdown(
                f"<div style='background:#161b22;border-left:3px solid {color};"
                f"border-radius:6px;padding:8px 14px;margin:4px 0'>"
                f"<span style='color:{color};font-weight:700'>{pred}</span> "
                f"({conf*100:.0f}%) — {prev}…</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No history yet.")

    if st.button("Close History"):
        st.session_state["show_history"] = False
        st.rerun()
