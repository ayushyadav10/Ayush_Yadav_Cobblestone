"""
main.py
───────
System Entrypoint for the German Power Market Day-Ahead Forecasting Pipeline.
Executes the following pipeline stages sequentially:
  1. Data Preparation
  2. QA Preprocessing
  3. Feature Engineering
  4. Feature Importance Audit
  5. Model Training & Validation (Cross-Validation + Holdout predictions)
  6. Day-Ahead Trading View Signal Extraction
  7. Automated Market Commentary Note Generation
"""

import time
import logging
import sys

# ── LOGGING SYSTEM SETUP ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main_pipeline")

from src.prepare_data import run_data_preparation
from src.qa_preprocessing import run_qa_preprocessing
from src.feature_engineering import run_feature_engineering
from src.models import run_production_models
from src.trading_views import generate_curve_trading_view
from src.llm_commentary import run_llm_commentary_engine


def run_pipeline():
    pipeline_start = time.time()
    
    log.info("=" * 65)
    log.info(" GERMAN DAY-AHEAD POWER MARKET FORECASTING PIPELINE STARTED ")
    log.info("=" * 65)

    stages = [
        ("1. Data Preparation", run_data_preparation),
        ("2. QA & Preprocessing", run_qa_preprocessing),
        ("3. Feature Engineering", run_feature_engineering),
        ("4. Model Suite Execution", run_production_models),
        ("5. Trading View Generation", generate_curve_trading_view),
        ("6. LLM Commentary Dispatch", run_llm_commentary_engine),
    ]

    for stage_name, stage_func in stages:
        stage_start = time.time()
        log.info(f"▶ Running Stage: {stage_name}...")
        
        try:
            stage_func()
            elapsed = time.time() - stage_start
            log.info(f"✓ Completed Stage: {stage_name} (in {elapsed:.2f} seconds)")
            log.info("-" * 55)
        except Exception as e:
            log.error(f" Pipeline failed at Stage: {stage_name}")
            log.error(f"Error Message: {e}", exc_info=True)
            sys.exit(1)

    total_time = time.time() - pipeline_start
    log.info("=" * 65)
    log.info(f" PIPELINE SUCCESSFUL — All stages completed in {total_time/60:.2f} minutes")
    log.info("=" * 65)


if __name__ == "__main__":
    run_pipeline()
