# Employee Attrition Prediction — NYU ML Project

This project uses supervised machine learning to predict employee attrition using structured HR data. Developed as part of NYU's Machine Learning course (Spring 2025). Received **outstanding instructor feedback** for real-world relevance, robustness, and ethical focus. Extended in 2026 with production-grade ML practices: SHAP explainability, fairness auditing, model calibration, and artifact serialization.

## Abstract

Attrition leads to loss of talent, increased recruitment costs, and lowered productivity. This project trains and evaluates ensemble models (XGBoost, CatBoost, Random Forest) on the IBM HR Analytics dataset to predict which employees are likely to leave — enabling proactive, data-driven HR intervention.

---

## Instructor Feedback

> "**Excellent work on this project!** You've gone above and beyond expectations in terms of scope, practical applicability, and rigor.
> Your focus on explainability, fairness, and real-world HR impact is exactly how machine learning should be applied.
> You've clearly understood the purpose of the project, evaluated your model pipeline thoroughly, and thought deeply about interpretation."

---

## Key Highlights

| Component | Details |
|-----------|---------|
| Dataset | IBM HR Analytics (1,470 records, 34 features) |
| Models | Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, CatBoost |
| Class Imbalance | SMOTE pipeline — resamples training set only; F1/AUC comparison with vs. without |
| Top Model Accuracy | 86% (XGBoost) |
| Explainability | SHAP TreeExplainer — beeswarm, waterfall, dependence plots |
| Fairness Audit | `fairlearn` MetricFrame — demographic parity + equalized odds across gender, age, marital status |
| Calibration | Reliability diagrams, isotonic regression, Brier score, threshold optimizer |
| Artifact Management | `joblib` serialization of model + scaler + `model_metadata.json` |
| Business Insight | Overtime, low income, low job satisfaction → top predictors of attrition |

---

## Contents

- `Employee_Attrition.ipynb` — Colab notebook: EDA, modeling, SHAP, fairness audit, calibration, serialization
- `ML Project Report - Employee Attrition Prediction.pdf` — Full project report
- `model_artifacts/` — Serialized XGBoost model, scaler, and metadata JSON
- `README.md` — This file

---

## EDA Insights

- Younger employees (<30) and those with <2 years tenure had the highest attrition rates
- Overtime and low job/environment satisfaction were the strongest behavioral predictors
- Sales department and Lab Technician role faced the highest turnover
- Commute distance and low income had compounding attrition risk effects

---

## Models and Evaluation

Models evaluated on:
- Accuracy
- F1-Score (minority class: Attrition = Yes) — primary metric
- ROC-AUC (CatBoost: 0.737, XGBoost: 0.73–0.91)
- Confusion Matrix
- Brier Score (calibration quality)

Ensemble models (XGBoost, CatBoost) significantly outperformed linear and distance-based models.

---

## Production-Grade Extensions (2026)

### SHAP Explainability
`TreeExplainer` provides exact SHAP values (no approximation). Outputs include:
- Beeswarm summary plot — global feature impact distribution
- Bar chart — mean |SHAP| feature ranking
- Waterfall plot — per-employee prediction explanation
- Dependence plots — feature interaction with top predictors

### Fairness Audit
`fairlearn.MetricFrame` measures per-group accuracy, F1, precision, recall across:
- **Gender** — checks for disparate prediction rates
- **Age Group** (18–30, 31–40, 41–50, 51+) — checks for age-based bias
- **Marital Status** — checks for marital status disparity

Demographic Parity Difference and Equalized Odds Difference reported per group. Threshold: |< 0.10| per EEOC guidance.

### Probability Calibration
Reliability diagrams for XGBoost, CatBoost, Random Forest. Isotonic regression calibration applied to XGBoost with Brier score before/after comparison. Threshold optimization sweep (precision/recall/F1 vs. threshold) with recommended HR operating point (high recall).

### Artifact Management
`joblib` serialization of trained XGBoost model and StandardScaler. `model_metadata.json` captures training date, feature names, and evaluation metrics. Load-and-predict demo cell verifies reproducibility.

---

## Real-World Application

- Early warning system for at-risk employees
- Per-employee prediction explanation (SHAP waterfall) for HR managers
- Fairness-audited decisions — compliant with algorithmic bias guidelines
- Calibrated probabilities for risk-tiered retention planning
- Serialized artifacts ready for FastAPI/Gradio deployment

---

## Tech Stack

- **Language:** Python
- **Core ML:** scikit-learn, XGBoost, CatBoost, imbalanced-learn (SMOTE)
- **Explainability:** shap
- **Fairness:** fairlearn
- **Serialization:** joblib
- **Visualization:** matplotlib, seaborn
- **Notebook:** Google Colab
- **Dataset:** [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## Future Enhancements (Priority 2)

- **LLM-powered explanations** — Ollama llama3.2 + SHAP values → plain-English HR manager briefings
- **Optuna hyperparameter optimization** — replace default XGBoost config with 50-trial Optuna study
- **MLflow experiment tracking** — log all runs, parameters, and artifacts
- **FT-Transformer** — tabular attention model baseline for research depth
- **FastAPI deployment** — `/predict` endpoint returning probability + SHAP + LLM explanation

---

## Authors

**Sourik Dutta**
Graduate Student, Computer Science — NYU
sd5913@nyu.edu

**Niharika Bhasin**
Graduate Student, Computer Science — NYU
nb4048@nyu.edu
[LinkedIn](https://linkedin.com/in/niharika-bhasin)

---

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Applied Sciences (2022), JETIR (2020), ResearchGate papers
- NYU Machine Learning (INT2), Final Project — Spring 2025
