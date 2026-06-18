# MLflow Experiment Tracker — POC 🚀

A **Proof of Concept** demonstrating MLflow's core capabilities by training and comparing three models on the California Housing dataset.

---

## 📁 Project Structure

```
mlflow-experiment-tracker/
├── train.py            ← Main training + logging script
├── requirements.txt    ← Python dependencies
├── plots/              ← Auto-generated plots (created on run)
└── mlruns/             ← MLflow tracking data (created on run)
```

---

## ⚙️ Setup

### Step 1 — Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the POC

```bash
python train.py
```

**What happens:**
1. Loads the California Housing dataset (20,640 samples, 8 features)
2. Trains **3 models** — Linear Regression, Random Forest, XGBoost
3. Logs **params, metrics, plots** to MLflow for each model
4. Generates a **comparison dashboard** across all 3 models
5. Creates a **Summary Run** with all results side by side

---

## 📊 Launch MLflow UI

```bash
mlflow ui
```

Open your browser → **http://localhost:5000**

---

## 🔍 What's Logged to MLflow

| Component | Details |
|-----------|---------|
| **Parameters** | Model hyperparameters (depth, estimators, lr, etc.) |
| **Metrics** | RMSE, MAE, R² on test set |
| **Tags** | model_type, dataset, framework |
| **Artifacts** | Prediction plots, feature importance charts |
| **Models** | Saved model artifacts in MLflow format |

---

## 📈 Models & Metrics

| Model | Notes |
|-------|-------|
| **Linear Regression** | Baseline; uses scaled features |
| **Random Forest** | Ensemble; 150 trees, max_depth=12 |
| **XGBoost** | Boosting; 300 estimators, lr=0.08 |

**Target:** House price (in $100k units)  
**Metrics:** RMSE (↓ better), MAE (↓ better), R² (↑ better)

---

## 🖼️ Generated Plots

Each model generates:
- **Actual vs Predicted** scatter plot
- **Residual distribution** histogram
- **Feature importance** bar chart (RF & XGBoost)

Plus a final **Model Comparison Dashboard** comparing all 3 models.

---

## 🌐 MLflow UI — What to Explore

1. **Experiments View** — See all 4 runs listed
2. **Compare Runs** — Select runs → click *Compare* to see side-by-side charts
3. **Artifacts Tab** — View the saved prediction plots
4. **Model Registry** — Register the best model and promote to Production
