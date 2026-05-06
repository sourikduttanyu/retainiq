# Upgrade Plan — Google-Ready AIML Portfolio

## Goal
Address the two critical gaps flagged for Google L3/L4 Applied ML:
1. No scale signal
2. Shallow LLM integration

---

## Phase A — Docker + Production Packaging
**Time: ~30 min | Signal: production instinct**

- [ ] `Dockerfile` for API (`api/main.py`) — multi-stage, non-root user
- [ ] `docker-compose.yml` — services: `api` (port 8000) + `frontend` (nginx, port 3000)
- [ ] `.dockerignore`
- [ ] Health check in compose (`curl /health`)
- [ ] Update README with `docker-compose up` quickstart

---

## Phase B — Batch Prediction Endpoint
**Time: ~45 min | Signal: scale awareness**

- [ ] `POST /predict/batch` — accepts JSON array of employees (up to 1000)
- [ ] Vectorized preprocessing (single `scaler.transform` call on full batch)
- [ ] Vectorized SHAP (`explainer.shap_values(X_batch)`)
- [ ] Returns array of predictions, each with probability + risk level + top-3 SHAP
- [ ] No LLM call on batch (cost/latency justification — shows systems thinking)
- [ ] `GET /predict/batch/csv` — accepts CSV upload via `UploadFile`, returns CSV with predictions appended
- [ ] Update frontend with a "Batch Upload" tab (CSV drag-drop, results table)

---

## Phase C — RAG-Grounded LLM Explanations
**Time: ~2 hours | Signal: actual AI engineering**

### What
Replace the naive `prompt → Ollama` call with a proper RAG pipeline:
- Embed all 1,470 IBM HR employees using a local embedding model
- At prediction time, retrieve 3 most similar employees from the training set
- Include their actual outcomes (left / stayed) as context in the prompt
- LLM now explains the prediction grounded in real historical cases

### Stack
- **ChromaDB** — local vector store (no API key, no cloud)
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, 384-dim
- **Ollama llama3.2** — generation (unchanged)

### Steps
- [ ] `api/rag.py` — build ChromaDB collection from `WA_Fn-UseC_-HR-Employee-Attrition.csv`
  - Embed each employee's feature vector + metadata
  - Store attrition label, key features (overtime, income, role) as metadata
  - Persist collection to `rag_store/` directory
- [ ] `api/rag.py` — `retrieve_similar(X_scaled, k=3)` function
  - Embed input employee
  - Query collection for k nearest neighbors
  - Return list of `{features, actual_outcome, similarity_score}`
- [ ] Update `api/main.py` — `_llm_explain()` to accept retrieved cases
  - Inject cases into prompt: "Similar employees: [case 1 stayed — here's why], [case 2 left — here's why]"
  - Add `retrieved_cases` to `/predict` response
- [ ] `POST /rag/rebuild` endpoint — rebuild index (for demo / CI purposes)

### Prompt upgrade (before → after)
**Before:**
> "Employee has HIGH risk (73%). Overtime: increases risk. Write 2 sentences."

**After:**
> "Employee has HIGH risk (73%). Top factors: overtime (+0.85), single marital status (+0.32), low income (-0.30).
> 3 similar employees from historical data:
> — Case 1 (similarity 0.94): Sales Rep, single, overtime, $2,800/mo → LEFT after 14 months
> — Case 2 (similarity 0.91): Sales Exec, single, overtime, $3,200/mo → LEFT after 8 months
> — Case 3 (similarity 0.89): Sales Rep, married, overtime, $2,900/mo → STAYED (promoted at month 18)
> Write 2 sentences for an HR manager explaining why this employee is at risk and what action prevented attrition in Case 3."

---

## Phase D — Structured LLM Output
**Time: ~30 min | Signal: production LLM engineering**

- [ ] Force Ollama to return JSON via `format: "json"` parameter
- [ ] Schema: `{"summary": str, "primary_risk_factor": str, "recommended_action": str, "confidence": "high"|"medium"|"low"}`
- [ ] Parse + validate with Pydantic in the API response
- [ ] Update frontend to render structured fields separately (not just a text blob)

---

## Phase E — FT-Transformer Full Training (close the gap)
**Time: ~1 hour | Signal: follow-through on research claim**

- [ ] Actually train FT-Transformer 20 epochs (CPU with OMP fix is fine — it'll finish)
- [ ] Log to MLflow alongside XGBoost/CatBoost
- [ ] Add comparison table cell to notebook: FT-Transformer vs XGBoost (accuracy, AUC, F1)
- [ ] Add markdown analysis: why XGBoost wins on 1,470 rows (the answer is the point)

---

## Phase F — F1 Score Improvement
**Time: ~1 hour | Signal: model quality ownership**

Current: F1 minority = 0.35 (poor)

- [ ] Threshold optimization — sweep 0.2–0.5, pick F1-maximizing threshold
- [ ] Cost-sensitive learning — `scale_pos_weight` in XGBoost (currently not set)
- [ ] Calibration-aware threshold — use calibrated probabilities for threshold sweep
- [ ] Target: F1 minority ≥ 0.50
- [ ] Update `model_metadata.json` with new `decision_threshold`

---

## Order of execution

```
A (Docker) → F (F1 fix) → B (Batch) → C (RAG) → D (Structured output) → E (FT-Transformer)
```

A and F are quick wins with high interview impact.
C is the flagship item — 20 min of interview conversation minimum.

---

## Files touched

| File | Phases |
|------|--------|
| `Dockerfile` | A |
| `docker-compose.yml` | A |
| `.dockerignore` | A |
| `api/main.py` | B, C, D |
| `api/rag.py` | C (new) |
| `rag_store/` | C (generated) |
| `frontend/index.html` | B, D |
| `Employee_Attrition.ipynb` | E, F |
| `model_artifacts/` | F |
| `requirements.txt` | C (chromadb, sentence-transformers) |
| `README.md` | all |
