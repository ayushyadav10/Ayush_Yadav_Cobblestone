"""
feature_importance_analysis.py
──────────────────────────────
Input:  data/processed/feature_matrix.csv
Output: outputs/feature_importance_analysis/feature_importances.csv
        figures/feature_importance_analysis/feature_importance.png

Computes cross-validated LightGBM feature importances and exports report artifacts.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# pyrefly: ignore [missing-import]
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# ── LOGGING SYSTEM SETUP ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── PATHS SETUP ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR   = ROOT / "outputs" / "feature_importance_analysis"
FIGURES_DIR   = ROOT / "figures" / "feature_importance_analysis"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def run_importance_analysis():
    src = PROCESSED_DIR / "feature_matrix.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}. Run feature_engineering.py first.")

    df = pd.read_csv(src, index_col="datetime", parse_dates=True)
    
    X = df.drop(columns=["price_eur_mwh"])
    y = df["price_eur_mwh"]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_importances = np.zeros(X.shape[1])

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        fold_importances += model.feature_importances_ / 5

    # ══════════════════════════════════════════════════════════
    # Export Importance CSV Data
    # ══════════════════════════════════════════════════════════
    importance_df = pd.DataFrame({"feature": X.columns, "importance": fold_importances}).sort_values(by="importance", ascending=False).reset_index(drop=True)
    csv_out = OUTPUTS_DIR / "feature_importances.csv"
    importance_df.to_csv(csv_out, index=False)
    log.info(f"Saved feature importances ──> {csv_out}")
    
    # ══════════════════════════════════════════════════════════
    # Generate & Export Importance Plot
    # ══════════════════════════════════════════════════════════
    plt.figure(figsize=(12, 10))
    sns.barplot(data=importance_df.head(30), x="importance", y="feature", palette="viridis")
    plt.title("Candidate Space Feature Audit (LightGBM)", fontweight="bold")
    plt.tight_layout()
    
    img_out = FIGURES_DIR / "feature_importance.png"
    plt.savefig(img_out, dpi=300)
    plt.close()
    log.info(f"Saved feature importance plot ──> {img_out}")


if __name__ == "__main__":
    run_importance_analysis()
