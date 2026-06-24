"""
qa_preprocessing.py
───────────────────
Input:  data/merged/master_dataset.csv
Output: data/processed/processed_dataset.csv
        outputs/qa_preprocessing/missing_summary.csv
        outputs/qa_preprocessing/data_statistics.csv
        outputs/qa_preprocessing/qa_summary.csv

Tasks:
  1. Load master dataset
  2. Run QA checks (missing, physical bounds, frozen data, gaps) → outputs/
  3. Fix missing values (ffill)
  4. Add price_outlier_flag
  5. Add residual_load
  6. Verify index integrity
  7. Save qa_summary.csv
  8. Save processed_dataset.csv

"""

import logging
from pathlib import Path
import pandas as pd

# ── LOGGING SYSTEM SETUP ──────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── PATHS SETUP ───────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
MERGED_DIR    = ROOT / "data" / "merged"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR   = ROOT / "outputs" / "qa_preprocessing"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def run_qa_preprocessing() -> pd.DataFrame:
    
    # ── Load MASTER DATASET ─────────────────────────────────────────────
    src = MERGED_DIR / "master_dataset.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}\nRun prepare_data.py first.")

    master = pd.read_csv(src, index_col="datetime", parse_dates=True)
    master.index = pd.to_datetime(master.index).tz_localize(None)

    # Initialise QA summary dict 
    qa = {}

    # ══════════════════════════════════════════════════════════
    # 1 — RUN QUALITY ASSURANCE AUDITS
    # ══════════════════════════════════════════════════════════
    missing = pd.DataFrame({
        "missing_count": master.isna().sum(),
        "missing_pct":   (master.isna().mean() * 100).round(4),
    })
    missing.to_csv(OUTPUTS_DIR / "missing_summary.csv")
    log.info(f"Saved missing summary ──> {OUTPUTS_DIR / 'missing_summary.csv'}")

    for col, row in missing[missing["missing_count"] > 0].iterrows():
        log.warning(f"  ⚠  {col}: {int(row['missing_count'])} missing values ({row['missing_pct']:.2f}%)")


    # Data stats description
    stats = master.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats.to_csv(OUTPUTS_DIR / "data_statistics.csv")
    log.info(f"Saved data statistics ──> {OUTPUTS_DIR / 'data_statistics.csv'}")

    # Physical Sanity Bounds
    generation_cols = [
        "biomass_mwh", "hydro_mwh", "wind_offshore_mwh",
        "wind_onshore_mwh", "solar_mwh", "gas_gen_mwh",
        "lignite_mwh", "hard_coal_mwh",
    ]
    phys_errors = 0
    for col in generation_cols:
        if col not in master.columns:
            continue
        neg = int((master[col] < 0).sum())
        if neg > 0:
            log.warning(f"  ⚠  {col:30s}: {neg} negative generation values detected")
            phys_errors += neg

    temp_anom = int(((master["temperature_c"] < -30) | (master["temperature_c"] > 45)).sum())
    if temp_anom > 0:
        log.warning(f"  ⚠  temperature_c: {temp_anom} anomalies outside [-30°C, 45°C]")
        phys_errors += temp_anom

    qa["physical_bound_errors"] = phys_errors

    # Frozen / Stuck Data Detection
    freeze_summary = {}
    for col in ["price_eur_mwh", "load_mwh", "wind_total_mwh"]:
        if col not in master.columns:
            continue
        streak = (
            (master[col].diff() == 0)
            .astype(int)
            .groupby(master[col].diff().ne(0).cumsum())
            .cumsum()
        )
        max_streak = int(streak.max())
        freeze_summary[col] = max_streak
        if max_streak > 12:
            log.warning(f"  ⚠  {col}: frozen values detected for up to {max_streak} consecutive hours")


    qa["max_price_flatline_hours"] = freeze_summary.get("price_eur_mwh", 0)
    qa["max_load_flatline_hours"]  = freeze_summary.get("load_mwh", 0)
    qa["max_wind_flatline_hours"]  = freeze_summary.get("wind_total_mwh", 0)

    # ══════════════════════════════════════════════════════════
    # 2 — MISSING VALUES IMPUTATION
    # ══════════════════════════════════════════════════════════
    n_missing_before = int(master.isna().sum().sum())
    master = master.ffill()
    n_missing_after  = int(master.isna().sum().sum())
    if n_missing_before > 0:
        log.info(f"Imputed {n_missing_before} missing cells via forward-fill.")
    qa["missing_before_imputation"] = n_missing_before
    qa["missing_after_imputation"]  = n_missing_after

    # ══════════════════════════════════════════════════════════
    # 3 — CONSTRUCT TARGET PRICE OUTLIER INDEX
    # ══════════════════════════════════════════════════════════
    log.info("── Price Outlier Flag ──")

    # Dynamically extract the exact 99th percentile boundary from EDA insights
    OUTLIER_THRESHOLD = float(master["price_eur_mwh"].quantile(0.99))
    master["price_outlier_flag"] = (master["price_eur_mwh"] > OUTLIER_THRESHOLD).astype(int)
    
    n_outliers = int(master["price_outlier_flag"].sum())
    log.info(f"  Threshold (99th pct): {OUTLIER_THRESHOLD:.2f} EUR/MWh")
    
    qa["price_outlier_hours"] = n_outliers

    # ══════════════════════════════════════════════════════════
    # 4 — CONSTRUCT RESIDUAL LOAD COLUMN
    # ══════════════════════════════════════════════════════════
    master["residual_load"] = (
        master["load_mwh"]
        - master["wind_total_mwh"]
        - master["solar_mwh"]
    )

    # ══════════════════════════════════════════════════════════
    # 5 — VERIFY INDEX INTEGRITY
    # ══════════════════════════════════════════════════════════
    n_dupes      = int(master.index.duplicated().sum())
    is_monotonic = bool(master.index.is_monotonic_increasing)

    # Hourly gap detection
    expected = pd.date_range(
        start=master.index.min(),
        end=master.index.max(),
        freq="h",
    )
    missing_hours = len(expected) - len(master.index)
    if missing_hours > 0:
        log.warning(f"  ⚠  Structural timeline gaps: {missing_hours} missing hours detected.")

    qa["duplicate_timestamps"]  = n_dupes
    qa["is_monotonic"]          = is_monotonic
    qa["structural_gap_hours"]  = missing_hours

    if n_dupes > 0:
        raise ValueError(f"{n_dupes} duplicate timestamps identified in dataset.")
    if not is_monotonic:
        raise ValueError("Index monotonicity constraint violated.")

    # ══════════════════════════════════════════════════════════
    # 6 — SAVE VALIDATION SUMMARY REPORTS
    # ══════════════════════════════════════════════════════════
    qa_summary = {
        "execution_time":               pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows":                   len(master),
        "total_columns":                len(master.columns),
        "start_date":                   str(master.index.min()),
        "end_date":                     str(master.index.max()),
        "duplicate_timestamps":         qa["duplicate_timestamps"],
        "is_monotonic_increasing":      qa["is_monotonic"],
        "structural_gap_hours":         qa["structural_gap_hours"],
        "missing_before_imputation":    qa["missing_before_imputation"],
        "missing_after_imputation":     qa["missing_after_imputation"],
        "physical_bound_errors":        qa["physical_bound_errors"],
        "max_price_flatline_hours":     qa["max_price_flatline_hours"],
        "max_load_flatline_hours":      qa["max_load_flatline_hours"],
        "max_wind_flatline_hours":      qa["max_wind_flatline_hours"],
        "price_outlier_hours":          qa["price_outlier_hours"],
    }

    df_summary = pd.DataFrame.from_dict(qa_summary, orient="index", columns=["value"])
    df_summary.index.name = "metric"
    df_summary.to_csv(OUTPUTS_DIR / "qa_summary.csv")
    log.info(f"Saved QA summary ──> {OUTPUTS_DIR / 'qa_summary.csv'}")

    # ══════════════════════════════════════════════════════════
    # 7 — EXPORT CLEANED PROCESSED DATASET
    # ══════════════════════════════════════════════════════════
    out = PROCESSED_DIR / "processed_dataset.csv"
    master.to_csv(out)
    log.info(f"Processed dataset saved successfully ──> {out} ({master.shape[0]:,} rows × {master.shape[1]} columns)")

    return master


if __name__ == "__main__":
    df = run_qa_preprocessing()