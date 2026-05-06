# Employee Attrition Prediction

> XGBoost + FastAPI system that predicts employee attrition probability, explains each prediction with SHAP and retrieval-augmented historical cases, and generates HR briefings via a local LLM — fully containerized.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Live Demo

| Low Risk Profile | Medium Risk Profile |
|:---:|:---:|
| ![Low Risk](screenshots/Screenshot%202026-05-06%20at%2004.02.40.png) | ![Medium Risk](screenshots/Screenshot%202026-05-06%20at%2004.02.54.png) |

Each prediction returns a risk tier (HIGH / MEDIUM / LOW), SHAP feature attribution bars, three historically similar employees with their actual outcomes, and a two-sentence HR briefing from a local LLM.

---

## Architecture

```
IBM HR CSV (1,470 rows)
        │
        ├── Training Pipeline
        │       ├── SMOTE (class imbalance)
        │       ├── Optuna 50-trial TPE search (objective: F1 minority)
        │       ├── XGBoost (136 features after one-hot encoding)
        │       ├── Isotonic calibration (threshold → 0.368)
        │       ├── Fairness audit (fairlearn MetricFrame)
        │       └── MLflow experiment tracking
        │
        ├── RAG Index (built once at startup)
        │       └── sentence-transformers (all-MiniLM-L6-v2)
        │               → ChromaDB (cosine similarity, HNSW)
        │
        └── FastAPI (port 8000)
                ├── POST /predict
                │       ├── XGBoost predict_proba
                │       ├── SHAP TreeExplainer (top-5 features)
                │       ├── ChromaDB → top-3 similar historical cases
                │       └── Ollama llama3.2 → 2-sentence HR briefing
                ├── POST /predict/batch      (vectorized, up to 1,000 rows)
                ├── POST /predict/batch/csv  (upload CSV → download CSV)
                └── nginx frontend (port 3000)
```

---

## Key Engineering Decisions

- **Optimized for F1 (minority class), not AUC.** Attrition is ~16% of the dataset. Optimizing for AUC rewards the model for getting the majority class right; F1 on the minority class directly measures what matters to HR — catching employees likely to leave. Threshold was swept post-training and set to 0.368 (vs. the naive 0.5) to shift the operating point toward higher recall on the attrition class.

- **RAG over pure LLM generation.** The LLM prompt is grounded with three retrieved historical employees (cosine similarity in ChromaDB) who share the same profile and have known real outcomes. This keeps the LLM explanation factual and traceable rather than hallucinated — HR managers can see which actual employees the system is referencing.

- **Local LLM (Ollama llama3.2), not an API.** Employee data is sensitive. Sending it to a third-party API endpoint is a compliance problem in any real HR context. Running llama3.2 locally via Ollama means all inference stays on-premises. The `/predict` endpoint degrades gracefully — if Ollama is unavailable, it returns `null` for `llm_explanation` without erroring.

- **SMOTE applied inside the training split only.** SMOTE is applied after train/test split to prevent the synthetic minority samples from leaking into test evaluation, which would inflate F1. `scale_pos_weight` is also set in XGBoost as a complementary signal, not a replacement. F1 on the minority class improved from 0.35 (no handling) to 0.51 with both.

- **Docker healthcheck gates the frontend.** The `docker-compose.yml` has the nginx frontend container set with `condition: service_healthy` — it only starts after the FastAPI `/health` endpoint has responded successfully three times. This prevents serving a UI that points to an API still loading its 136-feature model and SHAP explainer.

---

## System Capabilities

| Endpoint | Method | What It Does | Notable |
|----------|--------|--------------|---------|
| `/health` | GET | Returns model name, feature count, and evaluation metrics | Good canary for deployment monitoring |
| `/predict` | POST | Single employee → probability, risk tier, SHAP top-5, RAG cases, LLM briefing | Full inference pipeline; LLM call is best-effort (no error if Ollama is down) |
| `/predict/batch` | POST | Up to 1,000 employees → probability + SHAP top-3 each, vectorized | No LLM call; all SHAP computed in one batch pass |
| `/predict/batch/csv` | POST | Upload CSV → download CSV with predictions appended | Columns appended: `attrition_probability`, `risk_level`, `prediction` |
| `/rag/rebuild` | POST | Rebuilds ChromaDB index from the IBM HR CSV | Use after retraining on new data |

---

## ML Pipeline

**Dataset:** IBM HR Analytics, 1,470 employees, 35 raw features → 136 after one-hot encoding of categorical and ordinal columns. Class distribution: ~84% stayed, ~16% left.

**Class imbalance:** SMOTE applied to the training split only (1,029 rows). `scale_pos_weight` set in XGBoost to the inverse class ratio. Combined, these brought minority-class F1 from 0.35 (baseline, no handling) to 0.51.

**Hyperparameter optimization:** Optuna TPE sampler, 50 trials, objective: F1 on attrition class. Search space covered `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`.

**Threshold calibration:** Post-training probability calibration via isotonic regression. Decision threshold swept from 0.1 to 0.9; 0.368 selected as the operating point maximizing F1 on the minority class.

**Final model metrics (serialized in `model_metadata.json`):**

| Metric | Value |
|--------|-------|
| Accuracy | 85.7% |
| ROC-AUC | 0.789 |
| F1 (attrition class) | 0.511 |
| Decision threshold | 0.368 |
| Training samples | 1,029 |
| Features | 136 |

**Fairness audit:** `fairlearn.MetricFrame` evaluated accuracy, F1, precision, and recall across gender, age group (18–30, 31–40, 41–50, 51+), and marital status. Demographic parity difference and equalized odds difference reported per group. Target: |difference| < 0.10 per EEOC guidance.

**Deep learning baseline:** FT-Transformer (Feature Tokenizer + Transformer) implemented in PyTorch via `rtdl` as a comparison point. Attention over tabular features; CPU-safe on Apple Silicon.

**Experiment tracking:** All model runs logged to MLflow with parameters, metrics, and artifacts.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **ML** | XGBoost, scikit-learn, imbalanced-learn (SMOTE), Optuna, shap, fairlearn, PyTorch, rtdl (FT-Transformer), MLflow |
| **API** | FastAPI, uvicorn, Pydantic, joblib |
| **Frontend** | Vanilla JS / CSS — SVG risk gauge, SHAP bars, RAG cases panel, preset profiles, field tooltips |
| **Infra** | Docker, docker-compose, nginx, healthchecks |
| **LLM + RAG** | Ollama llama3.2 (local), sentence-transformers (all-MiniLM-L6-v2), ChromaDB (HNSW cosine) |

---

## Quickstart

**Prerequisites:** Docker and docker-compose installed. For LLM explanations: [Ollama](https://ollama.ai) running locally with `ollama pull llama3.2`.

**Docker (recommended):**

```bash
docker-compose up
```

Frontend at `http://localhost:3000` — API at `http://localhost:8000`. The frontend waits for the API healthcheck before starting.

**Manual:**

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload        # API on port 8000
cd frontend && python3 -m http.server 3000   # Frontend on port 3000
```

**Single prediction (curl):**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":28,"dailyrate":800,"distancefromhome":10,"employeenumber":1,
       "hourlyrate":60,"monthlyincome":3500,"monthlyrate":15000,
       "totalworkingyears":4,"yearsatcompany":2,
       "businesstravel":"Travel_Rarely","department":"Sales",
       "educationfield":"Marketing","gender":"Male","jobrole":"Sales Representative",
       "maritalstatus":"Single","overtime":"Yes",
       "education":3,"environmentsatisfaction":2,"jobinvolvement":2,"joblevel":1,
       "jobsatisfaction":1,"numcompaniesworked":2,"percentsalaryhike":11,
       "performancerating":3,"relationshipsatisfaction":3,"stockoptionlevel":0,
       "trainingtimeslastyear":2,"worklifebalance":2,"yearsincurrentrole":1,
       "yearssincelastpromotion":0,"yearswithcurrmanager":1}'
```

---

## Repository Structure

```
.
├── Employee_Attrition.ipynb          # Full notebook: EDA, modeling, SHAP, fairness, calibration, MLflow, FT-Transformer
├── api/
│   ├── main.py                       # FastAPI app: /predict, /predict/batch, /predict/batch/csv, /rag/rebuild
│   └── rag.py                        # ChromaDB index build + cosine retrieval
├── frontend/
│   └── index.html                    # Single-page dashboard (no build step)
├── model_artifacts/
│   ├── xgb_model.joblib              # Serialized XGBoost model
│   ├── scaler.joblib                 # Fitted StandardScaler
│   └── model_metadata.json          # Metrics, threshold, feature names
├── screenshots/                      # UI screenshots
├── docker-compose.yml
└── ML Project Report - Employee Attrition Prediction.pdf
```

---

## Authors

**Sourik Dutta** — sd5913@nyu.edu — NYU Computer Science (MS)

**Niharika Bhasin** — nb4048@nyu.edu — NYU Computer Science (MS) — [LinkedIn](https://linkedin.com/in/niharika-bhasin)

---

**Dataset:** [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
