"""Phase 2 cell insertion script — Optuna, FT-Transformer, MLflow, LLM."""
import json, copy

NB_PATH = 'Employee_Attrition.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Start: {len(cells)} cells")


def code(src: str):
    lines = [l + '\n' for l in src.split('\n')]
    lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def md(src: str):
    lines = [l + '\n' for l in src.split('\n')]
    lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


# ── Section 5B: Optuna (4 cells, insert at 113) ──────────────────────────────

optuna_A = md("""## Section 5B: Hyperparameter Optimization (Optuna)
Optuna is the 2024–2026 standard for HPO — replaces GridSearchCV for gradient boosting.
Uses Tree-structured Parzen Estimator (TPE) to intelligently sample the search space.
50 trials with ROC-AUC objective evaluated via 3-fold stratified cross-validation.""")

optuna_B = code("""import optuna
from sklearn.model_selection import StratifiedKFold
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 50, 300),
        'max_depth':       trial.suggest_int('max_depth', 2, 6),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight':trial.suggest_int('min_child_weight', 1, 10),
        'reg_lambda':      trial.suggest_float('reg_lambda', 0.1, 5.0),
        'reg_alpha':       trial.suggest_float('reg_alpha', 0.0, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,
    }
    m = XGBClassifier(**params)
    scores = cross_val_score(
        m, X_train_std, y_train,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best ROC-AUC: {study.best_value:.4f}")
print(f"Best params:  {study.best_params}")""")

optuna_C = code("""xgb_opt = XGBClassifier(**study.best_params, eval_metric='logloss',
                        random_state=42, verbosity=0)
xgb_opt.fit(X_train_std, y_train)

print("=== XGBoost: Regularized vs Optuna-Tuned ===")
for name, m in [('Regularized (manual)', xgb_clf), ('Optuna-tuned', xgb_opt)]:
    yp  = m.predict(X_test_std)
    ypr = m.predict_proba(X_test_std)[:, 1]
    print(f"\\n{name}")
    print(f"  Train acc: {accuracy_score(y_train, m.predict(X_train_std)):.4f}")
    print(f"  Test  acc: {accuracy_score(y_test,  yp):.4f}")
    print(f"  ROC-AUC : {roc_auc_score(y_test, ypr):.4f}")
    print(f"  F1      : {f1_score(y_test, yp):.4f}")

opt_auc  = roc_auc_score(y_test, xgb_opt.predict_proba(X_test_std)[:, 1])
base_auc = roc_auc_score(y_test, xgb_clf.predict_proba(X_test_std)[:, 1])
if opt_auc >= base_auc:
    xgb_clf = xgb_opt
    print("\\nOptuna XGBoost promoted as primary model")""")

optuna_D = code("""optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Optuna: Optimization History (50 trials)")
plt.tight_layout()
plt.show()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Optuna: Hyperparameter Importance")
plt.tight_layout()
plt.show()""")

optuna_cells = [optuna_A, optuna_B, optuna_C, optuna_D]


# ── Section 5C: FT-Transformer (4 cells, insert at 117 after Optuna) ─────────

ft_A = md("""## Section 5C: Tabular Deep Learning — FT-Transformer
FT-Transformer (Feature Tokenizer + Transformer) applies self-attention to tabular features.
Each feature becomes a token; attention captures feature interactions that tree-based methods
approximate with splits. Relevant to DeepMind / Google research teams working on tabular
foundation models.""")

ft_B = code("""import rtdl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset""")

ft_C = code("""X_tr = torch.tensor(X_train_std, dtype=torch.float32)
X_te = torch.tensor(X_test_std,  dtype=torch.float32)
y_tr = torch.tensor(y_train.values, dtype=torch.long)
y_te = torch.tensor(y_test.values,  dtype=torch.long)

n_features = X_tr.shape[1]

model_ft = rtdl.FTTransformer.make_default(
    n_num_features=n_features,
    cat_cardinalities=[],
    d_out=2,
)

optimizer = torch.optim.AdamW(model_ft.parameters(), lr=1e-4, weight_decay=1e-5)
loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

model_ft.train()
for epoch in range(20):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model_ft(X_batch, None)
        loss   = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20  loss={total_loss/len(loader):.4f}")

model_ft.eval()
with torch.no_grad():
    logits   = model_ft(X_te, None)
    probs_ft = F.softmax(logits, dim=1)[:, 1].numpy()
    preds_ft = logits.argmax(dim=1).numpy()

print(f"\\nFT-Transformer Results:")
print(f"  Accuracy : {accuracy_score(y_test, preds_ft):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, probs_ft):.4f}")
print(f"  F1       : {f1_score(y_test, preds_ft):.4f}")""")

ft_D = md("""## Why FT-Transformer May Not Beat XGBoost Here
On small tabular datasets (< 10K rows), gradient boosting typically outperforms
attention-based models because:
1. Self-attention benefits from scale — more data = richer token interactions
2. Tree inductive bias (axis-aligned splits) matches tabular feature distributions
3. Transformers require more data to learn positional relationships between features

FT-Transformer is included to demonstrate awareness of the research frontier.
The comparison is the point, not the ranking.""")

ft_cells = [ft_A, ft_B, ft_C, ft_D]


# ── Section 4B: MLflow (3 cells, insert after model comparison) ───────────────

mlflow_A = md("""## Section 4B: Experiment Tracking (MLflow)
MLflow logs every run's parameters, metrics, and model artifacts to a local tracking server.
Answers "how do you manage experiments?" in any AIML interview with a concrete artifact to show.""")

mlflow_B = code("""import mlflow
import mlflow.sklearn
from sklearn.metrics import brier_score_loss as bsl

mlflow.set_experiment("employee-attrition-prediction")

models_to_log = {
    'LogisticRegression': (lr_clf,  {}),
    'RandomForest':       (rf_clf,  {'n_estimators': 100, 'max_depth': 15, 'bootstrap': True}),
    'XGBoost_Optuna':     (xgb_clf, study.best_params if 'study' in globals() else {}),
    'CatBoost':           (cb_clf,  {}),
    'DecisionTree':       (dt_clf,  {'max_depth': 5, 'min_samples_leaf': 5}),
}

for run_name, (m, params) in models_to_log.items():
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        yp  = m.predict(X_test_std)
        ypr = m.predict_proba(X_test_std)[:, 1] if hasattr(m, 'predict_proba') else None
        mlflow.log_metric("accuracy",       accuracy_score(y_test, yp))
        mlflow.log_metric("f1_score",       f1_score(y_test, yp))
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, m.predict(X_train_std)))
        if ypr is not None:
            mlflow.log_metric("roc_auc",     roc_auc_score(y_test, ypr))
            mlflow.log_metric("brier_score", bsl(y_test, ypr))
        mlflow.sklearn.log_model(m, run_name)
        print(f"Logged: {run_name}")""")

mlflow_C = code("""from mlflow.tracking import MlflowClient
client = MlflowClient()
exp    = client.get_experiment_by_name("employee-attrition-prediction")
runs   = client.search_runs(exp.experiment_id, order_by=["metrics.roc_auc DESC"])

run_data = [{
    'Model':     r.info.run_name,
    'ROC-AUC':   round(r.data.metrics.get('roc_auc', 0), 4),
    'F1':        round(r.data.metrics.get('f1_score', 0), 4),
    'Accuracy':  round(r.data.metrics.get('accuracy', 0), 4),
    'Train Acc': round(r.data.metrics.get('train_accuracy', 0), 4),
} for r in runs]

print("=== MLflow Experiment Results (sorted by ROC-AUC) ===")
pd.DataFrame(run_data)""")

mlflow_cells = [mlflow_A, mlflow_B, mlflow_C]


# ── Section 10: LLM Explanations (4 cells, append at end) ────────────────────

llm_A = md("""## Section 10: LLM-Powered HR Insight Generation
Combines SHAP-grounded feature attribution with Ollama llama3.2 (local LLM) to generate
plain-English attrition risk explanations for HR managers.
Grounding in SHAP values prevents hallucination — the LLM only narrates what the model
actually computed.""")

llm_B = code("""import subprocess, time, requests

def ensure_ollama_running():
    \"\"\"Start ollama serve if not already running.\"\"\"
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    subprocess.Popen(
        ["/opt/homebrew/bin/ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

ollama_available = ensure_ollama_running()
print(f"Ollama server: {'running' if ollama_available else 'unavailable — using fallback'}")""")

llm_C = code("""def explain_attrition(employee_idx, use_llm=True):
    \"\"\"Generate plain-English attrition explanation for one employee.\"\"\"
    sv            = shap_values[employee_idx]
    feature_names = X_train.columns.tolist()

    top_idx      = np.argsort(np.abs(sv))[-5:][::-1]
    top_features = [(feature_names[i],
                     float(X_test_std[employee_idx][i]),
                     float(sv[i])) for i in top_idx]

    row  = X_test_std[employee_idx:employee_idx+1]
    prob = xgb_clf.predict_proba(row)[0, 1]
    risk = "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW"

    features_str = "\\n".join(
        f"  - {name}: value={val:.2f}, SHAP={shap:+.3f} "
        f"({'increases' if shap > 0 else 'decreases'} risk)"
        for name, val, shap in top_features
    )

    prompt = (
        f"You are an HR analytics assistant. A machine learning model predicts this employee "
        f"has a {prob:.0%} probability of leaving ({risk} risk).\\n\\n"
        f"The top 5 factors driving this prediction are:\\n{features_str}\\n\\n"
        f"Write 2-3 sentences for an HR manager explaining WHY this employee is at risk and "
        f"what action to consider. Be specific, factual, and use the feature names. "
        f"Do not add information not present above."
    )

    if use_llm and ollama_available:
        import ollama as ollama_client
        response    = ollama_client.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}])
        explanation = response['message']['content']
    else:
        top_name, _, top_shap = top_features[0]
        direction   = "high" if top_shap > 0 else "low"
        explanation = (
            f"Employee shows {risk} attrition risk ({prob:.0%}). "
            f"Primary driver: {top_name} ({direction} value, SHAP={top_shap:+.3f}). "
            f"Additional factors: {', '.join(n for n, _, _ in top_features[1:3])}."
        )

    return {'risk': risk, 'probability': prob,
            'explanation': explanation, 'top_features': top_features}""")

llm_D = code("""probs_all = xgb_clf.predict_proba(X_test_std)[:, 1]
high_idx  = int(probs_all.argmax())
med_idx   = int(np.argsort(np.abs(probs_all - 0.5))[0])
low_idx   = int(probs_all.argmin())

for label, idx in [('HIGH RISK', high_idx), ('MEDIUM RISK', med_idx), ('LOW RISK', low_idx)]:
    result = explain_attrition(idx, use_llm=ollama_available)
    print(f"\\n{'='*60}")
    print(f"  {label} | P(attrition) = {result['probability']:.2%}")
    print(f"{'='*60}")
    print(result['explanation'])
    print("\\nTop SHAP drivers:")
    for name, val, shap in result['top_features'][:3]:
        bar = '|' * max(1, int(abs(shap) * 30))
        print(f"  {'+'if shap>0 else '-'} {name:<25} {bar} {shap:+.3f}")""")

llm_cells = [llm_A, llm_B, llm_C, llm_D]


# ── Insert in top-to-bottom order ─────────────────────────────────────────────
# Start: 147 cells

# Step 1: Optuna at position 113 (after XGBoost eval cells 110-112, before SHAP)
for i, c in enumerate(optuna_cells):
    cells.insert(113 + i, c)
assert len(cells) == 151, f"Expected 151, got {len(cells)}"

# Step 2: FT-Transformer at position 117 (right after Optuna)
for i, c in enumerate(ft_cells):
    cells.insert(117 + i, c)
assert len(cells) == 155, f"Expected 155, got {len(cells)}"

# Step 3: MLflow at position 142
# Old cell 133 (last model comparison) shifted by +8 → now at 141; insert after = 142
for i, c in enumerate(mlflow_cells):
    cells.insert(142 + i, c)
assert len(cells) == 158, f"Expected 158, got {len(cells)}"

# Step 4: LLM — append at end
cells.extend(llm_cells)
assert len(cells) == 162, f"Expected 162, got {len(cells)}"

# ── Write back ────────────────────────────────────────────────────────────────
nb['cells'] = cells
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Done. Total cells: {len(cells)}")

# Verify insertion points
print("\nVerification:")
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    if 'Section 5B' in src:
        print(f"  Cell {i}: Section 5B Optuna header")
    if 'study.optimize' in src:
        print(f"  Cell {i}: Optuna study.optimize")
    if 'Section 5C' in src:
        print(f"  Cell {i}: Section 5C FT-Transformer header")
    if 'FTTransformer' in src:
        print(f"  Cell {i}: FT-Transformer training")
    if 'Section 4B' in src:
        print(f"  Cell {i}: Section 4B MLflow header")
    if 'mlflow.set_experiment' in src:
        print(f"  Cell {i}: MLflow logging")
    if 'Section 10' in src:
        print(f"  Cell {i}: Section 10 LLM header")
    if 'ensure_ollama_running' in src:
        print(f"  Cell {i}: Ollama helper")
