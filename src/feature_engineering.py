"""
feature_engineering.py
──────────────────────
Input:  data/processed/processed_dataset.csv
Output: data/processed/feature_matrix.csv

Feature groups:
  1. Calendar          — hour, weekday, month, week_of_year
  2. Custom blocks     — is_evening_peak, is_solar_crater, is_weekend_solar_peak, is_sunday
  3. Cyclical encoding — sin/cos of hour, weekday, month
  4. Lag features      — price lag 1h, 24h, 48h, 168h
  5. Rolling features  — mean/std over 24h and 168h windows
  6. Fundamental lags  — wind, solar, load, residual_load, ttf all lagged 24h
  7. Interaction       — residual_load_squared, renewable_penetration
  8. Holiday flag      — German public holidays via `holidays` library
"""

import logging
from pathlib import Path
# pyrefly: ignore [missing-import]
import holidays
import numpy as np
import pandas as pd

# ── LOGGING SYSTEM SETUP ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── PATHS SETUP ───────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# German public holidays — covers all 16 states (federal level)
DE_HOLIDAYS = holidays.Germany()


def run_feature_engineering() -> pd.DataFrame:
    # ── Load ──────────────────────────────────────────────────
    src = PROCESSED_DIR / "processed_dataset.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}\nRun qa_preprocessing.py first.")
    
    df = pd.read_csv(src, index_col="datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # ══════════════════════════════════════════════════════════
    # 1. CALENDAR FEATURES
    # ══════════════════════════════════════════════════════════

    df["hour"]         = df.index.hour
    df["weekday"]      = df.index.dayofweek        # 0=Mon … 6=Sun
    df["month"]        = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
    df["is_sunday"]    = (df["weekday"] == 6).astype(int)

    df["is_holiday"]   = df.index.normalize().map(lambda d: int(d in DE_HOLIDAYS))

    # Standard peak: 08:00–20:00 on working days
    df["is_peak"] = (
        (df["hour"] >= 8) & (df["hour"] < 20) & (df["is_weekend"] == 0) & (df["is_holiday"] == 0)
    ).astype(int)

    # ══════════════════════════════════════════════════════════
    # 2. CUSTOM EDA-DRIVEN BLOCK FEATURES
    # ══════════════════════════════════════════════════════════

    df["is_evening_peak"] = ((df["hour"] >= 18) & (df["hour"] <= 21)).astype(int)

    df["is_solar_crater"] = (
        df["month"].isin([4, 5, 6, 7, 8, 9]) & (df["hour"] >= 10) & (df["hour"] <= 15)
    ).astype(int)

    df["is_weekend_solar_peak"] = df["is_weekend"] * df["is_solar_crater"]

    # ══════════════════════════════════════════════════════════
    # 3. CYCLICAL ENCODING
    # ══════════════════════════════════════════════════════════

    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]    / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]    / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] /  7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] /  7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]   / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]   / 12)

    # ══════════════════════════════════════════════════════════
    # 4. PRICE LAG FEATURES
    # ══════════════════════════════════════════════════════════

    for lag in [1, 24, 48, 168]:
        df[f"price_lag_{lag}h"] = df["price_eur_mwh"].shift(lag)
    df["price_delta_24h"] = df["price_lag_1h"] - df["price_lag_24h"]

    # ══════════════════════════════════════════════════════════
    # 5. ROLLING FEATURES (No leakage)
    # ══════════════════════════════════════════════════════════

    price_shifted = df["price_eur_mwh"].shift(1)

    df["price_roll_mean_24h"]  = price_shifted.rolling(24,  min_periods=12).mean()
    df["price_roll_std_24h"]   = price_shifted.rolling(24,  min_periods=12).std()
    df["price_roll_mean_168h"] = price_shifted.rolling(168, min_periods=84).mean()
    df["price_roll_std_168h"]  = price_shifted.rolling(168, min_periods=84).std()

    # ══════════════════════════════════════════════════════════
    # 6. FUNDAMENTAL LAG FEATURES (24h lag)
    # ══════════════════════════════════════════════════════════

    fundamental_cols = [
        "wind_total_mwh", "solar_mwh", "load_mwh", "residual_load",
        "ttf_eur_mwh", "temperature_c", "gas_gen_mwh", "lignite_mwh",
        "renewable_ratio", "gas_share_thermal", "wind_speed_ms",
    ]
    for col in fundamental_cols:
        if col in df.columns:
            df[f"{col}_lag24h"] = df[col].shift(24)

    # ══════════════════════════════════════════════════════════
    # 7. INTERACTION & NONLINEAR FEATURES
    # ══════════════════════════════════════════════════════════

    df["residual_load_lag24h_squared"] = (df["residual_load_lag24h"] / 10000.0) ** 2
    df["renewable_penetration_lag24h"] = (
        ((df["wind_total_mwh_lag24h"] + df["solar_mwh_lag24h"]) / df["load_mwh_lag24h"].replace(0, np.nan)).round(4)
    )
    df["ttf_roll_mean_7d"] = df["ttf_eur_mwh"].rolling(168, min_periods=24).mean()
    
    # ══════════════════════════════════════════════════════════
    # 8. DROP WARMUP ROWS
    # ══════════════════════════════════════════════════════════
    df = df.dropna(subset=[
        "price_lag_168h",       # longest lag — determines warmup period
        "price_roll_mean_168h", # longest rolling window
    ])
    
    # ══════════════════════════════════════════════════════════
    # 9. SAVE EXPORTS
    # ══════════════════════════════════════════════════════════
    out = PROCESSED_DIR / "feature_matrix.csv"
    df.to_csv(out)
    log.info(f"Feature matrix saved successfully ──> {out} ({df.shape[0]:,} rows × {df.shape[1]} columns)")

    return df


if __name__ == "__main__":
    run_feature_engineering()
