"""
╔══════════════════════════════════════════════════════════════════╗
║           MLflow Experiment Tracker - POC                        ║
║                                                                  ║
║  Models  : Linear Regression | Random Forest | XGBoost          ║
║  Dataset : California Housing (sklearn built-in)                 ║
║  Tracks  : Params | Metrics | Artifacts | Plots                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import warnings

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (consistent across all plots)
# ─────────────────────────────────────────────
PALETTE = {
    "Linear Regression": "#4C72B0",
    "Random Forest":     "#2ecc71",
    "XGBoost":           "#e74c3c",
}

# ─────────────────────────────────────────────
#  DATA LOADING & SPLITTING
# ─────────────────────────────────────────────

def load_data():
    """Load California Housing dataset, return raw + scaled splits."""
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    return {
        "X_train":    X_train,
        "X_test":     X_test,
        "X_train_sc": X_train_sc,
        "X_test_sc":  X_test_sc,
        "y_train":    y_train,
        "y_test":     y_test,
        "feature_names": housing.feature_names,
    }


# ─────────────────────────────────────────────
#  METRIC HELPER
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred, model_name: str) -> str:
    color = PALETTE[model_name]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Scatter: actual vs predicted
    axes[0].scatter(y_test, y_pred, alpha=0.4, color=color, s=12, edgecolors="none")
    mn = min(float(y_test.min()), float(y_pred.min()))
    mx = max(float(y_test.max()), float(y_pred.max()))
    axes[0].plot([mn, mx], [mn, mx], "w--", lw=1.8, label="Perfect fit")
    axes[0].set_xlabel("Actual Price ($100k)", color="white")
    axes[0].set_ylabel("Predicted Price ($100k)", color="white")
    axes[0].set_title(f"{model_name}\nActual vs Predicted", color="white", fontsize=12, fontweight="bold")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

    # Histogram of residuals
    residuals = np.array(y_test) - np.array(y_pred)
    axes[1].hist(residuals, bins=60, color=color, alpha=0.85, edgecolor="none")
    axes[1].axvline(0, color="white", linestyle="--", lw=1.8)
    axes[1].set_xlabel("Residual", color="white")
    axes[1].set_ylabel("Count", color="white")
    axes[1].set_title(f"{model_name}\nResidual Distribution", color="white", fontsize=12, fontweight="bold")

    plt.tight_layout(pad=2.0)
    fname = os.path.join(PLOTS_DIR, f"{model_name.replace(' ', '_')}_predictions.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return fname


def plot_feature_importance(importances, feature_names, model_name: str) -> str:
    color = PALETTE[model_name]
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    bars = ax.barh(df["feature"], df["importance"], color=color, edgecolor="none", alpha=0.9)
    ax.set_xlabel("Importance Score", color="white")
    ax.set_title(f"{model_name} — Feature Importance", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelcolor="white")

    for bar in bars:
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}", va="center", color="white", fontsize=9
        )

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"{model_name.replace(' ', '_')}_feature_importance.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return fname


def plot_model_comparison(results: dict) -> str:
    models  = list(results.keys())
    metrics = ["rmse", "mae", "r2"]
    labels  = ["RMSE (↓ better)", "MAE (↓ better)", "R² (↑ better)"]
    colors  = [PALETTE[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#1a1a2e")

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        values = [results[m][metric] for m in models]
        bars   = ax.bar(models, values, color=colors, edgecolor="none", width=0.5)

        ax.set_title(label, color="white", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], color="white", fontsize=9)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}",
                ha="center", va="bottom", color="white", fontsize=10, fontweight="bold"
            )

    fig.suptitle("📊  Model Comparison Dashboard — California Housing",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return fname


# ─────────────────────────────────────────────
#  MODEL TRAINERS
# ─────────────────────────────────────────────

def run_linear_regression(data: dict) -> dict:
    model_name = "Linear Regression"
    print(f"\n{'─'*55}")
    print(f"  Training ➜  {model_name}")
    print(f"{'─'*55}")

    params = {"fit_intercept": True}

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_type",  "linear")
        mlflow.set_tag("dataset",     "California Housing")
        mlflow.set_tag("framework",   "scikit-learn")
        mlflow.log_params(params)

        model = LinearRegression(**params)
        model.fit(data["X_train_sc"], data["y_train"])
        y_pred = model.predict(data["X_test_sc"])

        metrics = compute_metrics(data["y_test"], y_pred)
        mlflow.log_metrics(metrics)

        # Artifacts
        pred_plot = plot_actual_vs_predicted(data["y_test"], y_pred, model_name)
        mlflow.log_artifact(pred_plot, artifact_path="plots")

        mlflow.sklearn.log_model(model, artifact_path="model")

    _print_metrics(model_name, metrics)
    return metrics


def run_random_forest(data: dict) -> dict:
    model_name = "Random Forest"
    print(f"\n{'─'*55}")
    print(f"  Training ➜  {model_name}")
    print(f"{'─'*55}")

    params = {
        "n_estimators":    150,
        "max_depth":       12,
        "min_samples_split": 5,
        "min_samples_leaf":  2,
        "max_features":   "sqrt",
        "random_state":    42,
        "n_jobs":          -1,
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_type",  "ensemble")
        mlflow.set_tag("dataset",     "California Housing")
        mlflow.set_tag("framework",   "scikit-learn")
        mlflow.log_params(params)

        model = RandomForestRegressor(**params)
        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_test"])

        metrics = compute_metrics(data["y_test"], y_pred)
        mlflow.log_metrics(metrics)

        # Artifacts
        pred_plot = plot_actual_vs_predicted(data["y_test"], y_pred, model_name)
        mlflow.log_artifact(pred_plot, artifact_path="plots")

        fi_plot = plot_feature_importance(
            model.feature_importances_, data["feature_names"], model_name
        )
        mlflow.log_artifact(fi_plot, artifact_path="plots")

        mlflow.sklearn.log_model(model, artifact_path="model")

    _print_metrics(model_name, metrics)
    return metrics


def run_xgboost(data: dict) -> dict:
    model_name = "XGBoost"
    print(f"\n{'─'*55}")
    print(f"  Training ➜  {model_name}")
    print(f"{'─'*55}")

    params = {
        "n_estimators":     300,
        "max_depth":          6,
        "learning_rate":    0.08,
        "subsample":        0.85,
        "colsample_bytree": 0.80,
        "reg_alpha":        0.10,
        "reg_lambda":       1.00,
        "random_state":      42,
        "n_jobs":            -1,
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_type",  "boosting")
        mlflow.set_tag("dataset",     "California Housing")
        mlflow.set_tag("framework",   "xgboost")
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(
            data["X_train"], data["y_train"],
            eval_set=[(data["X_test"], data["y_test"])],
            verbose=False,
        )
        y_pred = model.predict(data["X_test"])

        metrics = compute_metrics(data["y_test"], y_pred)
        mlflow.log_metrics(metrics)

        # Artifacts
        pred_plot = plot_actual_vs_predicted(data["y_test"], y_pred, model_name)
        mlflow.log_artifact(pred_plot, artifact_path="plots")

        fi_plot = plot_feature_importance(
            model.feature_importances_, data["feature_names"], model_name
        )
        mlflow.log_artifact(fi_plot, artifact_path="plots")

        mlflow.xgboost.log_model(model, artifact_path="model")

    _print_metrics(model_name, metrics)
    return metrics


# ─────────────────────────────────────────────
#  SUMMARY HELPERS
# ─────────────────────────────────────────────

def _print_metrics(model_name: str, metrics: dict):
    print(f"  ✅  RMSE : {metrics['rmse']:.4f}")
    print(f"  ✅  MAE  : {metrics['mae']:.4f}")
    print(f"  ✅  R²   : {metrics['r2']:.4f}")


def log_summary_run(results: dict):
    """Log a comparison-only run with the final dashboard chart."""
    with mlflow.start_run(run_name="📊 Model Comparison Summary"):
        mlflow.set_tag("run_type", "summary")

        comparison_plot = plot_model_comparison(results)
        mlflow.log_artifact(comparison_plot, artifact_path="plots")

        # Flatten all metrics into the summary run for easy querying
        for model_name, metrics in results.items():
            prefix = model_name.replace(" ", "_")
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v)

        best_by_rmse = min(results, key=lambda m: results[m]["rmse"])
        best_by_r2   = max(results, key=lambda m: results[m]["r2"])
        mlflow.set_tag("best_model_rmse", best_by_rmse)
        mlflow.set_tag("best_model_r2",   best_by_r2)


def print_final_summary(results: dict):
    print("\n" + "═" * 60)
    print("  📊  FINAL RESULTS SUMMARY")
    print("═" * 60)
    print(f"  {'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8}")
    for model, m in results.items():
        print(f"  {model:<22} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}")

    best = min(results, key=lambda x: results[x]["rmse"])
    print(f"\n  🏆  Best Model : {best}  (lowest RMSE)")
    print("═" * 60)
    print("\n  ✅  All runs logged to MLflow!")
    print("  👉  Launch the UI  →  mlflow ui")
    print("  🌐  Open browser   →  http://localhost:5000")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  🚀  MLflow Experiment Tracker POC  🚀")
    print("═" * 60)
    print("  Dataset  :  California Housing (sklearn)")
    print("  Models   :  Linear Regression | Random Forest | XGBoost")
    print("  Tracking :  Params | Metrics | Artifacts | Plots")
    print("═" * 60)

    # ── Experiment setup ──────────────────────────────────────────
    mlflow.set_experiment("MLflow_Experiment_Tracker_POC")

    # ── Load data ─────────────────────────────────────────────────
    print("\n  📦  Loading California Housing dataset …")
    data = load_data()
    print(f"  Train samples : {data['X_train'].shape[0]}")
    print(f"  Test  samples : {data['X_test'].shape[0]}")
    print(f"  Features      : {data['X_train'].shape[1]}")

    # ── Train all three models ────────────────────────────────────
    results = {}
    results["Linear Regression"] = run_linear_regression(data)
    results["Random Forest"]     = run_random_forest(data)
    results["XGBoost"]           = run_xgboost(data)

    # ── Summary run ───────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Logging ➜  Comparison Summary Run")
    print(f"{'─'*55}")
    log_summary_run(results)

    # ── Print final table ─────────────────────────────────────────
    print_final_summary(results)
