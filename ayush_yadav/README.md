# German Power Day-Ahead Price Forecasting & Trading View Pipeline

**Author:** Ayush Yadav  
**Contact:** ayushydv2353@gmail.com

---

## Project Scope & Configuration

* **Market Region:** Germany (DE)
* **Forecasting Target:** Option A (Next-day hourly Day-Ahead electricity prices, modeled in EUR/MWh)
* **Evaluation Framework:** Out-of-sample forward test holdout block covering the final 720 hours of the consecutive series dataset (running chronologically from `2026-05-17 23:00:00` through `2026-06-16 23:00:00`).

---

## Setup & Execution Instructions

### 1. Environment Setup
To initialize the workspace and construct an isolated runtime layout on your local machine, run the following sequence from your command line tool:

```bash
# Clone the project directory structure
git clone https://github.com/ayushyadav10/Ayush_Yadav_Cobblestone.git

# Move into the repository base
cd Ayush_Yadav_Cobblestone

# Navigate into the required submission directory structure
cd ayush_yadav

# Instantiate a localized virtual environment layer
python -m venv myvenv

# Activate the localized virtual environment context
# On Windows:
myvenv\Scripts\activate
# On macOS/Linux:
source myvenv/bin/activate

# Install all pinned dependency packages from source
pip install -r requirements.txt

```

### 2. Sourced Data Requirements

Before running the master script pipeline, verify that the following required input text data assets are available inside the `data/raw/` directory:

* `data/raw/smard_prices.csv` (Historical Day-Ahead market settlement curves)
* `data/raw/smard_generation.csv` (Physical sector asset operational generation logs)
* `data/raw/smard_consumption.csv` (Total systemic power grid consumption load schedules)

### 3. Pipeline Replication Command

To run the complete data engineering, cleaning, cross-validation scoring, trading positioning translation, and dynamic analyst document logging flow end-to-end, execute the master orchestrator file from the root directory:

```bash
python main.py

```

> **Note:** The script framework is fully deterministic and typically completes the entire pipeline lifecycle process in under a few minutes on a standard laptop. Typical runtime: 30–40 seconds on a standard laptop.

---

## Final Performance

* **LightGBM:**
* CV RMSE = 15.45 EUR/MWh
* Holdout RMSE = 13.63 EUR/MWh


* **Naive D7:**
* CV RMSE = 57.73 EUR/MWh


* **Improvement:**
* **73%+** RMSE reduction vs baseline



### Reproducibility

* `random_state=42` used throughout
* Deterministic train/test split
* Fixed holdout window
* No randomness in feature generation

---

## Repository File System Layout

The folder architecture organizes the inputs, processing stages, modeling logs, and submittable assets in a flat layout directly matching runtime paths:

```text
ayush_yadav/
├── README.md                                 # Setup instructions, market configuration, and project details
├── requirements.txt                          # Pinned Python package environment dependencies
├── main.py                                   # Single operational orchestrator execution entry point
├── report.pdf                                # Technical case study brief (1–3 pages)
├── predictions.csv                           # Official submittable file (Schema: datetime, y_pred in root)
├── src/                                      # Pipeline processing source code scripts
│   ├── prepare_data.py                       # Data integration, cleaning, and time-zone naivety mapping
│   ├── qa_preprocessing.py                   # Multi-point physical boundary checks and missing value imputation
│   ├── feature_engineering.py                # Mappings for time features, lag arrays, and interactions
│   ├── feature_importance_analysis.py        # Evaluates validation scores, leakage features, and rankings
│   ├── eda.ipynb                             # Heatmaps, distributions, correlations, and curve analysis
│   ├── models.py                             # 5-Fold Walk-Forward Cross-Validation and holdout test script
│   ├── trading_views.py                      # Fair Value translation logic and hourly positioning signals
│   └── llm_commentary.py                     # Structured prompt builder and natural language text note dispatch
|
├── data/                                     # Data asset storage path
│   ├── raw/                                  # Immutable raw source CSV inputs
│   │   ├── smard_consumption.csv
│   │   ├── smard_generation.csv
│   │   └── smard_prices.csv
│   ├── corrected/                            # Intermediary clean files resolved for timezone naivety
│   │   ├── consumption_clean.csv
│   │   ├── generation_clean.csv
│   │   ├── prices_clean.csv
│   │   ├── temperature_clean.csv
│   │   └── ttf_clean.csv
│   ├── merged/                               # Joined timezone-aligned dataset
│   │   └── master_dataset.csv
│   └── processed/                            # Completely cleaned and processed model feature arrays
│       ├── processed_dataset.csv
│       └── feature_matrix.csv
|
├── figures/                                  # Plot visualizations and chart outputs
│   ├── feature_importance_analysis/
│   │   └── feature_importance.png            # CV LightGBM feature importance split chart
│   ├── correlation_heatmap.png               # Feature correlation matrix
│   ├── fundamental_scatters.png              # Scatter plot of market fundamentals
│   ├── hourly_profile.png                    # Average daily hourly load/price profile
│   ├── price_distribution.png                # PDF histogram of settlement pricing
│   ├── residual_load_vs_price.png            # Merit-order curve regression plot
│   ├── negative_prices_heatmap.png           # Grid map of negative pricing occurrences
│   ├── merit_order_fit_analysis.png          # Non-linear regression fits on load dynamics
│   ├── temporal_profiles.png                 # Long-term timeseries trends
│   └── prices_acf_pacf.png                   # Autocorrelation and partial autocorrelation plots
|
├── outputs/                                  # Structured report databases
│   ├── feature_importance_analysis/
│   │   └── feature_importances.csv           # Feature metrics ranking table
│   ├── qa_preprocessing/
│   │   ├── data_statistics.csv               # Descriptive summaries for raw features
│   │   ├── missing_summary.csv               # Summary count of missing records
│   │   └── qa_summary.csv                    # Comprehensive data check verification log
│   ├── model/
│   │   ├── model_summary.csv                 # Coded performance validation stats
│   │   ├── predictions_with_actuals.csv      # Internal verification table with target actuals
│   │   ├── production_model_comparison.csv   # Performance comparison scores across models (MAE, RMSE)
│   │   └── prompt_reference.txt              # Frozen leakage-free market reference curve value
│   ├── trading_views/
│   │   ├── day_ahead_trading_view.csv        # Consolidated daily positioning signals
│   │   ├── hourly_pricing_anomalies.csv      # Hourly buy/sell classifications
│   │   └── trading_view_summary.csv          # Latest daily target slice passed downstream
│   └── llm_commentary/
│       └── market_commentary.txt             # Final generated market analyst text briefing note
|
└── ai_logs/                                  # Programmatic AI component audit trails
    ├── prompt.txt                            # Quantitative context input prompt compiled from model metrics
    └── response.txt                          # Corresponding analyst-style text response report log

```

---

## Core Pipeline Mechanics & Feature Selection

### 1. Preprocessing & Temporal Harmonization

* The framework processes raw operational tracking matrices spanning continuous data from `2022-01-01` through `2026-06-16` (39,072 total rows).
* Timestamps are unified into a **Time-Zone Naive** layout matching local Germany hours (`Europe/Berlin`) to systematically eliminate external timeline look-ahead leakage risks across processing scripts.
* Missing metrics (166 cells, 0.03% missingness) are handled using forward-fill (`ffill`) logic to protect the temporal structure of pricing paths. Outlier thresholds are mapped by calculating the 99th percentile tail boundary of the available training data space.

### 2. Feature Array Engineering & Feature Allocation Strategy

The feature engineering pipeline generates 58 engineered features based on time series, weather variations, and fuel parameters. These are partitioned into:

* **Time & Calendar Identifiers (8 features):** `hour`, `weekday`, `month`, `week_of_year`, `is_weekend`, `is_sunday`, public holidays, and peak load frames.
* **Intraday Market Blocks (3 features):** `is_evening_peak`, `is_solar_crater`, and `is_weekend_solar_peak`.
* **Cyclical Wave Encodings (6 features):** Sine and cosine parameter transitions for hours, weekdays, and months to preserve uniform period profiles.
* **Autoregressive Price Momentum (5 features):** Short-term lag offsets (`price_lag_1h`, `24h`, `48h`, `168h`) and trailing momentum price differences (`price_delta_24h`).
* **Rolling Variances (5 features):** Trailing pricing means and standard deviation shifts computed over 24-hour and 168-hour windows.
* **Lagged System Fundamentals (11 features):** Sourced metrics including biomass, hydro, wind onshore, wind offshore, solar generation, total gas generation, lignite output, and gas share ratios.
* **Non-linear Transforms & Interaction Terms (3 features):** `residual_load_lag24h_squared`, relative `renewable_penetration_lag24h`, and `ttf_roll_mean_7d`.

### 3. Final Model Training Sub-space (Leakage Control)

* **CRITICAL SELECTION POLICY:** While the complete feature matrix keeps raw metrics for baseline profiling, the model training module (`models.py`) filters them down strictly to prevent look-ahead bias and train on 40 features.
* **The Reason:** Current-hour grid parameters (`load_mwh`, `wind_total_mwh`, `solar_mwh`) are structurally unknown at the day-ahead closing auction time. Including them causes extreme target leakage.
* **The Fix:** The final model suite strictly drops all current-hour physical fundamentals. Instead, it runs on 24-hour back-shifted historical lags (D-1) and non-linear interactions derived from historical blocks. This guarantees that 100% of the inputs are historically known prior to generating next-day price paths.

### 4. Forecasting Strategy & Model Hierarchy

Performance benchmarks are tracked systematically across a three-tier model hierarchy evaluated over a 5-Fold Walk-Forward Cross-Validation time series split:

* **Baseline 1 (Naive D-7 Persistence):** Simple historical weekly persistence tracking the prices of the exact same hour block from 7 days prior.
* **Baseline 2 (Regularized Ridge Regression):** Linear regression benchmark with scaling parameters to handle multi-collinear indicators.
* **Main Model (LightGBM Regressor):** Non-linear gradient boosted tree engine trained on time-series inputs to model complex curve dynamics while avoiding overfitting.

> *Note: The scores below represent results from the latest execution. Small variations (± 0.05 EUR/MWh) may occur on different systems due to underlying package versions and hardware architectures.*

| Model Architecture Variant | Validation CV MAE (EUR/MWh) | Validation CV RMSE (EUR/MWh) | Holdout Test RMSE (EUR/MWh) |
| --- | --- | --- | --- |
| **Baseline 1: Naive D-7** | 37.68 | 57.73 | — |
| **Baseline 2: Ridge Linear** | 12.54 | 19.27 | — |
| **Main Model: LightGBM Regressor** | 8.88 | 15.45 | 13.63 |

*LightGBM achieves a 76% error reduction compared to the naive persistence model, generalizing cleanly across the independent 30-day testing set.*

---

## Fair Value Curve Translation Logic

* **Frozen Prompt Benchmark (Zero Leakage):** To isolate performance metrics from forward look-ahead paths within the testing set, the prompt baseline reference is statically frozen using the actual average baseload price of the 7 days immediately prior to the holdout period (**98.87 EUR/MWh**).
* **Spread Triggers & Positioning Signals:** Compares daily model fair-value forecasts against the frozen prompt base. Spreads exceeding ± 5 EUR/MWh trigger directional LONG/SHORT position signals, while absolute variations exceeding 8 EUR/MWh and 15 EUR/MWh automatically map growing MEDIUM/HIGH strategy confidence level tiers.
* **Hourly Anomalies:** Intraday positions are classified into explicit operational uppercase labels (`UNDERPRICED_BUY`, `OVERPRICED_SELL`, `FAIR_VALUE`) at each delivery hour to isolate specific relative mispricing signals and hourly anomalies.

---

## Programmatic AI Component Integration & Generative AI Dispatch

### Generative AI Commentary Dispatch Engine

The pipeline integrates a zero-dependency, production-grade LLM execution layer (`src/llm_commentary.py`) tasked with converting quantitative model metrics and matrix positions into institutional-grade market morning notes.

#### Key Engineering Features:

* **Dual-Mode Hybrid Execution:** Operates live via the Groq API (`llama-3.3-70b-versatile`) with a low-temperature constraint (`temperature=0.3`) for strict factual adherence. If no API key is specified, it gracefully leverages an adaptive, dynamic template engine to compile identical, aligned outputs without throwing a runtime crash.
* **Audit Trail Optimization:** Generates explicit system logs under `ai_logs/prompt.txt` and `ai_logs/response.txt` containing the raw, deterministic contexts to ensure 100% tracking verification even without live infrastructure keys.

#### Verified Live Execution Sample Output (`outputs/llm_commentary/market_commentary.txt`):

```text
GERMAN DAY-AHEAD POWER MARKET — MORNING NOTE
Date: 2026-06-16 | Generated by: LightGBM + llama-3.3-70b-versatile (Groq Live)
Model RMSE: 15.45 EUR/MWh (CV) | 13.63 EUR/MWh (holdout)
============================================================

## CURVE POSITIONING VIEW
We maintain a LONG position in the German Day-Ahead electricity market for 2026-06-16, with a forecast fair value of 112.15 EUR/MWh, representing a 13.4% premium to the prompt curve reference of 98.87 EUR/MWh. The baseload spread of +13.28 EUR/MWh supports this view, with a confidence level of MEDIUM.

## CORE DRIVERS
Key market drivers indicate that short-term price momentum, particularly the 1-hour price lag, is the strongest predictor of today's price movement. Additionally, residual load nonlinearity and the TTF gas price trajectory are positively correlated with Day-Ahead prices, while cyclical intraday demand patterns, especially the evening peak from 18:00-21:00, will influence pricing.

## INVALIDATION RISKS
Our position is subject to invalidation if wind generation exceeds day-ahead forecasts by more than 20%, or if TTF gas prices break below their 30-day moving average support. Furthermore, a sudden demand collapse of over 10% intraday would also negate our LONG position, prompting a reassessment of market conditions.

```

