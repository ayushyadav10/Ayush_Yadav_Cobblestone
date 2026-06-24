"""
trading_views.py
────────────────
Input:  outputs/predictions.csv
        outputs/model/prompt_reference.txt

Output: outputs/trading_views/day_ahead_trading_view.csv
        outputs/trading_views/hourly_pricing_anomalies.csv
        outputs/trading_views/trading_view_summary.csv

Generates systematic trading signals, daily positioning recommendations, 
and confidence matrices based on predictions and pre-holdout references.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

# ── LOGGING SYSTEM SETUP ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── PATHS SETUP ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "trading_views"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_curve_trading_view():
    log.info("=" * 65)
    log.info("ENERGY TRADING VIEW — Fair Value & Pure Positioning Logic")
    log.info("=" * 65)

    preds_src = ROOT / "outputs" / "model" / "predictions_with_actuals.csv"
    ref_src   = ROOT / "outputs" / "model" / "prompt_reference.txt"

    if not (preds_src.exists() and ref_src.exists()):
        raise FileNotFoundError("Missing inputs. Ensure models.py ran successfully first.")

    # ══════════════════════════════════════════════════════════
    # 1. LOAD PRE-HOLDOUT BASELINE REFERENCE (ZERO-LEAKAGE)
    # ══════════════════════════════════════════════════════════
    with open(ref_src, "r") as f:
        frozen_prompt_reference = float(f.read().strip())
    log.info(f"Loaded Isolated Pre-Holdout Prompt Baseline: {frozen_prompt_reference:.2f} EUR/MWh")

    df = pd.read_csv(preds_src, index_col="datetime", parse_dates=True)

    # ══════════════════════════════════════════════════════════
    # 2. HOURLY INTRADAY PRICING ANOMALIES DETECTION
    # ══════════════════════════════════════════════════════════
    # Calculate hourly spreads relative to the pre-holdout prompt reference    
    df["hourly_spread"] = df["predicted_price"] - frozen_prompt_reference
    
    # Classify hourly prices into systematic buy/sell/fair states
    df["pricing_state"] = "FAIR_VALUE"
    df.loc[df["hourly_spread"] > 8.0, "pricing_state"] = "UNDERPRICED_BUY"
    df.loc[df["hourly_spread"] < -8.0, "pricing_state"] = "OVERPRICED_SELL"

    df_hourly_out = df[["predicted_price", "hourly_spread", "pricing_state"]]
    df_hourly_out.to_csv(OUTPUTS_DIR / "hourly_pricing_anomalies.csv")


    # ══════════════════════════════════════════════════════════
    # 3. DAILY AGGREGATION & POSITIONING SIGNAL SYSTEM
    # ══════════════════════════════════════════════════════════
    # Resample to daily baseload mean values
    df_daily = df.resample("D").agg({"predicted_price": "mean"}).rename(columns={"predicted_price": "forecast_fair_value"})
    df_daily["prompt_reference"] = frozen_prompt_reference
    df_daily["baseload_spread"] = df_daily["forecast_fair_value"] - df_daily["prompt_reference"]

    # Compute macro percentage spread relative to prompt reference
    df_daily["spread_pct"] = (df_daily["baseload_spread"] / df_daily["prompt_reference"]) * 100

    # Determine curve bias (LONG/SHORT/NEUTRAL) using a 5.0 EUR/MWh trigger boundary    
    df_daily["curve_positioning"] = "NEUTRAL"
    df_daily.loc[df_daily["baseload_spread"] > 5.0, "curve_positioning"] = "LONG"
    df_daily.loc[df_daily["baseload_spread"] < -5.0, "curve_positioning"] = "SHORT"

    # Confidence Tiers
    df_daily["confidence_level"] = "LOW"
    df_daily.loc[df_daily["baseload_spread"].abs() > 8.0, "confidence_level"] = "MEDIUM"
    df_daily.loc[df_daily["baseload_spread"].abs() > 15.0, "confidence_level"] = "HIGH"

    df_daily.to_csv(OUTPUTS_DIR / "day_ahead_trading_view.csv")
    
    # ══════════════════════════════════════════════════════════
    # 4. EXPORT COMPACT REPORT SUMMARY FOR COMMENTARY ENGINE
    # ══════════════════════════════════════════════════════════
    # Save the most recent trading day's metrics to feed downstream analytics/LLM note    
    df_daily.tail(1).to_csv(OUTPUTS_DIR / "trading_view_summary.csv")

if __name__ == "__main__":
    generate_curve_trading_view()