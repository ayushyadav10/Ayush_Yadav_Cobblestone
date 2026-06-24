"""
models.py
─────────
Input:  data/processed/feature_matrix.csv

Output: outputs/model/production_model_comparison.csv
        outputs/model/model_summary.csv
        outputs/model/prompt_reference.txt
        outputs/model/predictions_with_actuals.csv
        predictions.csv (stored in root)
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
# pyrefly: ignore [missing-import]
import lightgbm as lgb


# ── LOGGING SYSTEM SETUP ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── PATHS SETUP ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR   = ROOT / "outputs" / "model"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def run_production_models():
    src = PROCESSED_DIR / "feature_matrix.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}. Run feature_engineering.py first.")
    df = pd.read_csv(src, index_col="datetime", parse_dates=True)

    # ══════════════════════════════════════════════════════════
    # CORE DATA-DRIVEN FEATURES (LEAK-SAFE)
    # ══════════════════════════════════════════════════════════
    MODEL_FEATURES = [
        "hour", "weekday", "month", "week_of_year", "is_weekend", "is_sunday", "is_holiday", "is_peak",
        "is_evening_peak", "is_solar_crater", "is_weekend_solar_peak",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
        "price_lag_1h", "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_delta_24h",
        "price_roll_mean_24h", "price_roll_std_24h", "price_roll_mean_168h", "price_roll_std_168h", "ttf_roll_mean_7d",
        "load_mwh_lag24h", "wind_total_mwh_lag24h", "solar_mwh_lag24h", "residual_load_lag24h", 
        "ttf_eur_mwh_lag24h", "temperature_c_lag24h", "gas_gen_mwh_lag24h", "lignite_mwh_lag24h", 
        "renewable_ratio_lag24h", "gas_share_thermal_lag24h", "wind_speed_ms_lag24h",
        "residual_load_lag24h_squared", "renewable_penetration_lag24h"
    ]

    # Isolate components
    X_full = df[MODEL_FEATURES]
    y_full = df["price_eur_mwh"]

    # ══════════════════════════════════════════════════════════
    # ── SPLIT STRICT HOLDOUT (Latest 30 Days = 720 Hours) ──────
    # ══════════════════════════════════════════════════════════
    holdout_hours = 720
    X_train_cv_pool = X_full.iloc[:-holdout_hours]
    y_train_cv_pool = y_full.iloc[:-holdout_hours]
    
    X_test_holdout = X_full.iloc[-holdout_hours:]
    y_test_holdout = y_full.iloc[-holdout_hours:]

    # ══════════════════════════════════════════════════════════
    # ── FREEZE TRUE LEAKAGE-FREE PROMPT REFERENCE (Pre-Holdout Last 7 Days) ──
    # ══════════════════════════════════════════════════════════
    frozen_prompt_val = float(y_train_cv_pool.tail(24 * 7).mean())
    
    with open(OUTPUTS_DIR / "prompt_reference.txt", "w") as f:
        f.write(f"{frozen_prompt_val:.4f}")
    log.info(f"Frozen Pre-Holdout Prompt Reference stored: {frozen_prompt_val:.2f} EUR/MWh")

    log.info(f"Historical CV Pool  : {X_train_cv_pool.shape[0]:,} records")
    log.info(f"Strict Test Holdout : {X_test_holdout.shape[0]:,} records (Last 30 Days)")

    # ══════════════════════════════════════════════════════════
    # ── VALIDATION CORE (5-Fold Walk-Forward on CV Pool) ──────
    # ══════════════════════════════════════════════════════════
    tscv = TimeSeriesSplit(n_splits=5)
    fold_logs = []

    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=15.0))])
    lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_cv_pool)):
        X_train, X_val = X_train_cv_pool.iloc[train_idx], X_train_cv_pool.iloc[val_idx]
        y_train, y_val = y_train_cv_pool.iloc[train_idx], y_train_cv_pool.iloc[val_idx]

        # 1. Naive Baseline (OOS Validation Only)
        val_preds_naive = X_val["price_lag_168h"]

        # 2. Ridge Regression
        ridge_pipe.fit(X_train, y_train)
        t_preds_ridge = ridge_pipe.predict(X_train)
        v_preds_ridge = ridge_pipe.predict(X_val)

        # 3. LightGBM Main Model
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(40, verbose=False)])
        t_preds_lgb = lgb_model.predict(X_train)
        v_preds_lgb = lgb_model.predict(X_val)

        # Log Scores Matrix
        fold_logs.append({"Fold": fold+1, "Model": "Baseline_1_Naive_D7", "Train_RMSE": np.nan, "Val_RMSE": np.sqrt(mean_squared_error(y_val, val_preds_naive)), "Train_MAE": np.nan, "Val_MAE": mean_absolute_error(y_val, val_preds_naive)})
        fold_logs.append({"Fold": fold+1, "Model": "Baseline_2_Ridge_Regression", "Train_RMSE": np.sqrt(mean_squared_error(y_train, t_preds_ridge)), "Val_RMSE": np.sqrt(mean_squared_error(y_val, v_preds_ridge)), "Train_MAE": mean_absolute_error(y_train, t_preds_ridge), "Val_MAE": mean_absolute_error(y_val, v_preds_ridge)})
        fold_logs.append({"Fold": fold+1, "Model": "Main_Model_LightGBM", "Train_RMSE": np.sqrt(mean_squared_error(y_train, t_preds_lgb)), "Val_RMSE": np.sqrt(mean_squared_error(y_val, v_preds_lgb)), "Train_MAE": mean_absolute_error(y_train, t_preds_lgb), "Val_MAE": mean_absolute_error(y_val, v_preds_lgb)})

    # ══════════════════════════════════════════════════════════
    # Compile Summary Cross Validation Scoreboard
    # ══════════════════════════════════════════════════════════
    df_scores = pd.DataFrame(fold_logs)
    summary_matrix = df_scores.groupby("Model").agg({"Train_RMSE": "mean", "Val_RMSE": "mean", "Train_MAE": "mean", "Val_MAE": "mean"}).reset_index().round(3)
    summary_matrix.to_csv(OUTPUTS_DIR / "production_model_comparison.csv", index=False)
    
    # ══════════════════════════════════════════════════════════
    # FINAL TEST HOLDOUT INFERENCE & EXPORT
    # ══════════════════════════════════════════════════════════
    log.info("── Training Final Champion Model on full historical pool for Holdout Target ──")
    lgb_model.fit(X_train_cv_pool, y_train_cv_pool)
    
    # Generate holdout predictions
    final_predictions = lgb_model.predict(X_test_holdout)
     
    # Holdout evaluation metric 
    holdout_rmse = np.sqrt(mean_squared_error(y_test_holdout, final_predictions))
    log.info(f" Final Independent Test Holdout (Last 30 Days) RMSE: {holdout_rmse:.2f} EUR/MWh")


    # Cache model summary metrics (Needed by llm_commentary.py)
    cv_rmse_mean = np.mean(summary_matrix.loc[summary_matrix["Model"] == "Main_Model_LightGBM", "Val_RMSE"].values)
    
    model_summary_df = pd.DataFrame({
        "cv_rmse": [cv_rmse_mean],
        "holdout_rmse": [holdout_rmse]
    })
    summary_out_path = OUTPUTS_DIR / "model_summary.csv"
    model_summary_df.to_csv(summary_out_path, index=False)
    log.info(f"Model Execution Summary Matrix cached ──> {summary_out_path}")

    # ══════════════════════════════════════════════════════════
    # INTERNAL EVALUATION FILE (Used by trading_views.py)
    # ══════════════════════════════════════════════════════════
    evaluation_df = pd.DataFrame({
        "actual_price": y_test_holdout.values,
        "predicted_price": np.round(final_predictions, 2)
    }, index=X_test_holdout.index)
    evaluation_df.index.name = "datetime"
    eval_out_path = OUTPUTS_DIR / "predictions_with_actuals.csv"
    evaluation_df.to_csv(eval_out_path)
    log.info(f"Evaluation file saved ──> {eval_out_path}")

    # ══════════════════════════════════════════════════════════
    # COBBLESTONE SUBMISSION FILE (Root predictions.csv)
    # ══════════════════════════════════════════════════════════
    submission_df = pd.DataFrame({
        "datetime": X_test_holdout.index,
        "y_pred": np.round(final_predictions, 2)
    })
    
    # ISO 8601 format formatting
    submission_df["datetime"] = submission_df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    submission_df.to_csv(ROOT / "predictions.csv", index=False)
    log.info(f"Submission file saved ──> {ROOT / 'predictions.csv'}")

if __name__ == "__main__":
    run_production_models()