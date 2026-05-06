# Employee Attrition Prediction API
# Run with: uvicorn api.main:app --reload

import json
import os
import pathlib

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the server can be launched from
# any working directory.
# ---------------------------------------------------------------------------
_BASE = pathlib.Path(__file__).parent.parent
_ARTIFACTS = _BASE / "model_artifacts"

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(_ARTIFACTS / "xgb_model.joblib")
scaler = joblib.load(_ARTIFACTS / "scaler.joblib")

with open(_ARTIFACTS / "model_metadata.json") as f:
    metadata = json.load(f)

FEATURE_NAMES: list[str] = metadata["feature_names"]
METRICS: dict = metadata["metrics"]

explainer = shap.TreeExplainer(model)

# ---------------------------------------------------------------------------
# Categorical and ordinal columns (ordinal treated as get_dummies too)
# ---------------------------------------------------------------------------
_CATEGORICAL = [
    "businesstravel", "department", "educationfield",
    "gender", "jobrole", "maritalstatus", "overtime",
]
_ORDINAL = [
    "education", "environmentsatisfaction", "jobinvolvement", "joblevel",
    "jobsatisfaction", "numcompaniesworked", "percentsalaryhike",
    "performancerating", "relationshipsatisfaction", "stockoptionlevel",
    "trainingtimeslastyear", "worklifebalance", "yearsincurrentrole",
    "yearssincelastpromotion", "yearswithcurrmanager",
]
_DUMMIES_COLS = _CATEGORICAL + _ORDINAL

# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------
class EmployeeInput(BaseModel):
    # Numerical
    age: int
    dailyrate: int
    distancefromhome: int
    employeenumber: int
    hourlyrate: int
    monthlyincome: int
    monthlyrate: int
    totalworkingyears: int
    yearsatcompany: int
    # Categorical
    businesstravel: str
    department: str
    educationfield: str
    gender: str
    jobrole: str
    maritalstatus: str
    overtime: str
    # Ordinal
    education: int
    environmentsatisfaction: int
    jobinvolvement: int
    joblevel: int
    jobsatisfaction: int
    numcompaniesworked: int
    percentsalaryhike: int
    performancerating: int
    relationshipsatisfaction: int
    stockoptionlevel: int
    trainingtimeslastyear: int
    worklifebalance: int
    yearsincurrentrole: int
    yearssincelastpromotion: int
    yearswithcurrmanager: int

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(emp: EmployeeInput) -> np.ndarray:
    row = emp.model_dump()
    # Convert ordinal ints to strings so get_dummies produces named columns
    for col in _ORDINAL:
        row[col] = str(row[col])
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=_DUMMIES_COLS)
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
    return scaler.transform(df)

# ---------------------------------------------------------------------------
# LLM explanation (best-effort)
# ---------------------------------------------------------------------------
def _llm_explain(risk_level: str, probability: float, top_factors: list[dict]) -> str | None:
    factors_text = "; ".join(
        f"{f['feature']} ({f['direction'].replace('_', ' ')}, shap={f['shap_value']:.3f})"
        for f in top_factors[:3]
    )
    prompt = (
        f"An employee has a {risk_level} attrition risk ({probability:.0%} probability). "
        f"Top contributing factors: {factors_text}. "
        "Write a 2-sentence explanation for an HR manager explaining what this means "
        "and what action to consider. Be concise and practical."
    )
    try:
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Employee Attrition Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata["model"],
        "n_features": metadata["n_features"],
        "metrics": METRICS,
    }


@app.post("/predict")
def predict(emp: EmployeeInput):
    try:
        X = preprocess(emp)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    proba = float(model.predict_proba(X)[0, 1])

    if proba > 0.6:
        risk_level = "HIGH"
    elif proba >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    prediction = "Attrition" if proba >= metadata["decision_threshold"] else "No Attrition"

    # SHAP on the preprocessed (scaled) array
    shap_values = explainer.shap_values(X)
    # For binary XGB, shap_values may be a list [neg, pos] or a single array
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    top_indices = np.argsort(np.abs(sv))[::-1][:5]
    shap_explanation = [
        {
            "feature": FEATURE_NAMES[i],
            "shap_value": float(sv[i]),
            "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
        }
        for i in top_indices
    ]

    llm_explanation = _llm_explain(risk_level, proba, shap_explanation)

    return {
        "attrition_probability": round(proba, 4),
        "risk_level": risk_level,
        "prediction": prediction,
        "shap_explanation": shap_explanation,
        "llm_explanation": llm_explanation,
    }
