# Fake News Detector

An advanced AI-powered misinformation detection system using NLP, fact-checking APIs, source credibility scoring, and explainable AI.

---

##  Architecture

```
fake-news-detector/
├── app.py                        # Streamlit UI
├── requirements.txt
│
├── api/
│   └── main.py                  # FastAPI backend (POST /analyze, GET /history)
│
├── models/
│   ├── train.py                 # TF-IDF + Logistic Regression training
│   ├── predict.py               # Inference (BERT → LR → heuristic chain)
│   └── saved_model/             # Persisted model files
│
├── processing/
│   ├── preprocess.py            # clean → tokenise → stopword removal → lemmatise
│   ├── feature_engineering.py   # caps ratio, exclamations, lexical diversity…
│   └── claim_extractor.py       # spaCy NER + regex claim extraction
│
├── fact_check/
│   ├── fact_api.py              # Google Fact Check + NewsAPI + heuristic KB
│   └── similarity.py            # Sentence-transformer / TF-IDF cosine similarity
│
├── credibility/
│   └── scorer.py                # Domain trust scoring (HTTPS, age, lists, TLD)
│
├── explainability/
│   └── explainer.py             # LIME → SHAP → rule-based fallback
│
├── utils/
│   ├── helpers.py               # URL fetch, SQLite helpers
│   └── config.py                # All tuneable parameters & API keys
│
└── data/
    ├── raw/                     # Drop your CSV datasets here
    └── processed/
```

---

##  Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. (Optional) Set API keys

```bash
export GOOGLE_FACT_CHECK_API_KEY="your_key"
export NEWS_API_KEY="your_key"
```

The system works without API keys using internal heuristics.

### 3. Train the baseline model

```bash
python models/train.py
```

Drop your own CSV files (columns: `text`, `label`) into `data/raw/` before training.
The system ships with a synthetic 20-sample dataset for demonstration.

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs → http://localhost:8000/docs

### 5. Start the UI (separate terminal)

```bash
streamlit run app.py
```

---

## 🔌 API Usage

### Analyze text

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "SHOCKING: Scientists discover miracle cure hidden by Big Pharma!"}'
```

### Analyze URL

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://apnews.com/..."}'
```

### Response structure

```json
{
  "prediction":          "FAKE",
  "confidence":          0.87,
  "credibility_score":   42.0,
  "credibility_verdict": "MEDIUM",
  "model_used":          "tfidf_logistic_regression",
  "processing_time_s":   0.23,
  "explanation": {
    "method":            "LIME",
    "top_features":      [...],
    "summary":           "Indicative of FAKE: shocking, miracle, hidden.",
    "emotional_phrases": ["shocking", "hidden"],
    "emotional_score":   0.4
  },
  "fact_check_results":  [...],
  "similarity_matches":  [...],
  "features": {
    "word_count": 12,
    "caps_ratio": 0.08,
    ...
  }
}
```

---

## Model Chain

| Priority | Model | When used |
|----------|-------|-----------|
| 1 | **BERT/DistilBERT** | `use_bert=true` + transformers installed |
| 2 | **TF-IDF + Logistic Regression** | Default; after `python models/train.py` |
| 3 | **Heuristic rule-based** | Zero-setup fallback |

---

##  Training Your Own Data

Compatible public datasets:
- [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- [ISOT Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

Place CSVs (columns `text`, `label` where 1=fake, 0=real) in `data/raw/` and re-run `python models/train.py`.

---

## Configuration

All knobs are in `utils/config.py`:
- `FAKE_THRESHOLD` – probability cutoff for FAKE label
- `SIMILARITY_THRESHOLD` – cosine similarity to flag a claim as contradicted
- `TRUSTED_DOMAINS` / `FAKE_DOMAINS` – domain lists
- Model names for BERT and SentenceTransformer
