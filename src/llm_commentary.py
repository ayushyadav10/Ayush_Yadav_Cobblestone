"""
llm_commentary.py
─────────────────
Input:  outputs/model/production_model_comparison.csv
        outputs/model/model_summary.csv
        outputs/trading_views/trading_view_summary.csv

Output: outputs/llm_commentary/market_commentary.txt
        ai_logs/prompt.txt
        ai_logs/response.txt

LLM Component:
  Uses Groq API to generate a daily market morning note.
  The LLM reads structured model metrics + trading signals and writes
  a concise analyst commentary.

Fallback behaviour:
  If GROQ_API_KEY is not set or fails, the script falls back deterministically
  to the production-compliant pre-formatted template using the parsed data arrays
  and adaptive signal context mapping to guarantee alignment.
"""

import logging
import os
from pathlib import Path
import pandas as pd
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from groq import Groq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = ROOT / "outputs" / "llm_commentary"
AI_LOGS_DIR  = ROOT / "ai_logs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
AI_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from project root
load_dotenv(ROOT / ".env")


def run_llm_commentary_engine():
    log.info("=" * 65)
    log.info("LLM COMMENTARY — German DA Power Market Morning Note")
    log.info("=" * 65)

    # ── Load upstream outputs ──────────────────────────────────
    comp_src    = ROOT / "outputs" / "model" / "production_model_comparison.csv"
    summary_src = ROOT / "outputs" / "model" / "model_summary.csv"
    trade_src   = ROOT / "outputs" / "trading_views" / "trading_view_summary.csv"

    for p in [comp_src, summary_src, trade_src]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun models.py and trading_views.py first."
            )

    df_metrics = pd.read_csv(comp_src)
    df_summary = pd.read_csv(summary_src)
    df_trade   = pd.read_csv(trade_src)

    latest      = df_trade.iloc[0]
    naive_rmse  = float(df_metrics.loc[df_metrics["Model"] == "Baseline_1_Naive_D7", "Val_RMSE"].values[0])
    lgb_cv_rmse = float(df_summary["cv_rmse"].iloc[0])
    holdout_rmse= float(df_summary["holdout_rmse"].iloc[0])

    date_str       = str(latest["datetime"]).split()[0]
    fair_value     = float(latest["forecast_fair_value"])
    prompt_ref     = float(latest["prompt_reference"])
    spread_nominal = float(latest["baseload_spread"])
    spread_pct     = float(latest["spread_pct"])
    signal         = str(latest["curve_positioning"])
    confidence     = str(latest["confidence_level"])
    improvement    = ((naive_rmse - lgb_cv_rmse) / naive_rmse) * 100

    # ══════════════════════════════════════════════════════════
    # BUILD PROMPT
    # ══════════════════════════════════════════════════════════
    prompt = f"""You are a senior quantitative analyst on a European power trading desk.
Write a concise daily market morning note for the German Day-Ahead electricity market.
Use clear, professional financial language. Maximum 220 words.
Structure your response with exactly three sections:
1. CURVE POSITIONING VIEW
2. CORE DRIVERS  
3. INVALIDATION RISKS

Use the following data to write the note:

MODEL PERFORMANCE:
- LightGBM CV RMSE: {lgb_cv_rmse:.2f} EUR/MWh
- LightGBM Holdout RMSE: {holdout_rmse:.2f} EUR/MWh
- Error reduction vs Naive D-7 baseline: {improvement:.1f}%

TRADING VIEW FOR {date_str}:
- Forecast fair value: {fair_value:.2f} EUR/MWh
- Prompt curve reference: {prompt_ref:.2f} EUR/MWh
- Baseload spread: {spread_nominal:+.2f} EUR/MWh ({spread_pct:+.1f}%)
- Position signal: {signal}
- Confidence: {confidence}

KEY MARKET DRIVERS (from feature importance analysis):
- Short-term price momentum (price_lag_1h is strongest predictor)
- Residual load nonlinearity (hockey-stick merit order curve)
- TTF gas price trajectory (strong positive relationship with DA prices)
- Cyclical intraday demand patterns (evening peak 18:00-21:00)

RISK INVALIDATION CONDITIONS:
- Wind generation overshoots day-ahead forecast by more than 20%
- TTF gas breaks below 30-day moving average support
- Sudden demand collapse exceeding 10% intraday"""

    # Save prompt to ai_logs/
    with open(AI_LOGS_DIR / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt.strip())
    log.info(f"  Prompt saved → {AI_LOGS_DIR / 'prompt.txt'}")

    # ══════════════════════════════════════════════════════════
    # GROQ API CALL WITH DETERMINISTIC FALLBACK
    # ══════════════════════════════════════════════════════════
    api_key = os.getenv("GROQ_API_KEY")
    response_text = None
    is_live_call = False

    if api_key:
        log.info("  Calling Groq API (llama-3.3-70b-versatile)...")
        try:
            client = Groq(api_key=api_key)
            chat_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior quantitative analyst on a European power trading desk. "
                            "Write concise, precise, trader-focused market commentary. "
                            "Avoid generic filler. Every sentence must reference specific numbers from the data provided."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,
                max_tokens=400,
            )
            response_text = chat_response.choices[0].message.content.strip()
            is_live_call = True
            log.info("  LLM response received successfully via live API call.")
        except Exception as e:
            log.error(f"  Groq API call failed: {e}")
            log.warning("  Switching to production-compliant deterministic fallback...")

    # Deterministic Local Fallback (Runs if API Key is missing OR if the live call fails)
    if response_text is None:
        if not api_key:
            log.warning("  GROQ_API_KEY not found in environment/.env file.")
        log.info("  Using deterministic local analyst engine for output compilation.")
        
        # Mapping context dynamics based on the active trading allocation signal
        position_text = {
            "LONG": "rendering long exposure statistically favorable",
            "SHORT": "rendering short exposure statistically favorable",
            "NEUTRAL": "suggesting limited directional conviction across current spreads"
        }.get(signal, "suggesting structural alignment check rules requirements")

        response_text = f"""1. CURVE POSITIONING VIEW
The Day-Ahead production champion model (Validation CV RMSE: {lgb_cv_rmse:.2f} EUR/MWh | Holdout Test RMSE: {holdout_rmse:.2f} EUR/MWh) has established a systematic {signal} curve positioning bias for the target delivery window, operating at a {confidence} confidence level threshold. The forecast indicates a daily Baseload Fair Value of {fair_value:.2f} EUR/MWh against a frozen screen prompt curve baseline of {prompt_ref:.2f} EUR/MWh. This variance unlocks a nominal alpha spread opportunity of {spread_nominal:+.2f} EUR/MWh ({spread_pct:+.1f}%), {position_text}.

2. CORE DRIVERS
The underlying macro directional edge is strongly driven by short-term price memory momentum and deterministic temporal cyclical demand phases. Non-linear constraints within the European merit-order curves are magnifying due to acute expansion across our quadratic residual load transformation indicators. This implies severe dispatch pressure on peak marginal gas units, shifting clearing points higher relative to trailing rolling averages.

3. INVALIDATION RISKS
This quantitative fair-value matrix stands completely invalidated if real-time physical system metrics experience structural deviations. Key downside vulnerability anchors include: a positive renewable supply shock where actualized wind generation output overshoots prompt forecast paths by more than 20%, a baseline breakdown of front-month TTF gas spot tokens underneath their 30-day moving averages, or an abrupt structural deterioration in macro systemic consumption grids."""

    # ══════════════════════════════════════════════════════════
    # SAVE OUTPUT DELIVERABLES
    # ══════════════════════════════════════════════════════════
    meta_model = "llama-3.3-70b-versatile (Groq Live)" if is_live_call else "Deterministic Hardened Fallback Engine"
    
    with open(AI_LOGS_DIR / "response.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {meta_model}\n")
        f.write(f"Date generated: {date_str}\n")
        f.write(f"Temperature: 0.3\n")
        f.write("=" * 60 + "\n\n")
        f.write(response_text)
    log.info(f"  Response saved → {AI_LOGS_DIR / 'response.txt'}")

    commentary_header = (
        f"GERMAN DAY-AHEAD POWER MARKET — MORNING NOTE\n"
        f"Date: {date_str} | Generated by: LightGBM + {meta_model}\n"
        f"Model RMSE: {lgb_cv_rmse:.2f} EUR/MWh (CV) | {holdout_rmse:.2f} EUR/MWh (holdout)\n"
        + "=" * 60 + "\n\n"
    )

    with open(OUTPUTS_DIR / "market_commentary.txt", "w", encoding="utf-8") as f:
        f.write(commentary_header + response_text)
    log.info(f"  Commentary saved → {OUTPUTS_DIR / 'market_commentary.txt'}")

    # Print to terminal
    print("\n" + "=" * 65)
    print("MARKET COMMENTARY DISPATCH")
    print("=" * 65)
    print(response_text)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_llm_commentary_engine()