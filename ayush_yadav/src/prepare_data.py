"""
prepare_data.py
───────────────
Reads raw SMARD CSVs from data/raw/
Cleans and keeps only needed columns
Fetches Temperature (Open-Meteo) and TTF gas (yfinance)
Merges everything into data/merged/master_dataset.csv
All intermediate files are tz-naive (Europe/Berlin local time, no UTC offset).
All outputs overwrite existing files — safe to run multiple times.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
# ── LOGGING SYSTEM SETUP ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
# ── PATHS SETUP ───────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
CORR_DIR   = ROOT / "data" / "corrected"
MERGED_DIR = ROOT / "data" / "merged"
CORR_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)
START_DATE = "2022-01-01"
END_DATE   = "2026-06-16"
# Full tz-naive hourly index in Berlin local time — single source of truth
HOURLY_IDX = pd.date_range(
    START_DATE,
    pd.to_datetime(END_DATE) + pd.Timedelta(hours=23),
    freq="h",
    name="datetime",
)

# ── HELPERS ───────────────────────────────────────────────────────
def _read_smard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        decimal=".",
        thousands=",",
        na_values=["-", "", " "],
        encoding="utf-8-sig",
    )
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    return df


def _parse_smard_dt(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[\t\xa0\u202f]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    parsed = pd.to_datetime(cleaned, format="%b %d %Y %I:%M %p", errors="coerce")
    # Fallback for mixed datetime formats
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            cleaned[mask], format="mixed", dayfirst=False, errors="coerce"
        )
    parsed = (
        parsed
        .dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_localize(None)
    )
    return parsed


def _align(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="first")].reindex(HOURLY_IDX)

# ══════════════════════════════════════════════════════════
# 1. PRICES CLEANING 
# ══════════════════════════════════════════════════════════
def clean_prices(path: Path) -> pd.DataFrame:
    cache = CORR_DIR / "prices_clean.csv"
    raw = _read_smard(path)
    start_col = next(c for c in raw.columns if "start" in c.lower())
    price_col = next(
        c for c in raw.columns
        if ("germany" in c.lower() or "de/lu" in c.lower()) and "neighbour" not in c.lower()
    )
    df = raw[[start_col, price_col]].copy()
    df.columns = ["dt_raw", "price_eur_mwh"]
    df["datetime"] = _parse_smard_dt(df["dt_raw"])
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    df = (
        df[["datetime", "price_eur_mwh"]]
        .dropna(subset=["datetime"])
        .set_index("datetime")
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(cache)
    log.info(f"Saved clean prices ──> {cache}")
    return df

# ══════════════════════════════════════════════════════════
# 2. GENERATION CLEANING 
# ══════════════════════════════════════════════════════════
def clean_generation(path: Path) -> pd.DataFrame:
    cache = CORR_DIR / "generation_clean.csv"
    raw = _read_smard(path)
    start_col = next(c for c in raw.columns if "start" in c.lower())
    col_map = {}
    for c in raw.columns:
        cl = c.lower()
        if   "wind offshore"  in cl: col_map[c] = "wind_offshore_mwh"
        elif "wind onshore"   in cl: col_map[c] = "wind_onshore_mwh"
        elif "photovoltaic"   in cl: col_map[c] = "solar_mwh"
        elif "fossil gas"     in cl: col_map[c] = "gas_gen_mwh"
        elif "lignite"        in cl: col_map[c] = "lignite_mwh"
        elif "hard coal"      in cl: col_map[c] = "hard_coal_mwh"
        elif "biomass"        in cl: col_map[c] = "biomass_mwh"
        elif "hydropower"     in cl and "pumped" not in cl: col_map[c] = "hydro_mwh"
    df = raw[[start_col] + list(col_map.keys())].copy()
    df["datetime"] = _parse_smard_dt(df[start_col])
    df = (
        df.drop(columns=[start_col])
        .rename(columns=col_map)
        .dropna(subset=["datetime"])
        .set_index("datetime")
        .sort_index()
    )
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[~df.index.duplicated(keep="first")]
    # Derived renewable / thermal columns
    df["wind_total_mwh"] = (
        df.get("wind_offshore_mwh", pd.Series(0, index=df.index)).fillna(0)
        + df.get("wind_onshore_mwh", pd.Series(0, index=df.index)).fillna(0)
    )
    thermal = (
        df.get("gas_gen_mwh",   pd.Series(0, index=df.index)).fillna(0)
        + df.get("lignite_mwh", pd.Series(0, index=df.index)).fillna(0)
        + df.get("hard_coal_mwh", pd.Series(0, index=df.index)).fillna(0)
    )
    renewable = (
        df.get("wind_total_mwh", pd.Series(0, index=df.index)).fillna(0)
        + df.get("solar_mwh",    pd.Series(0, index=df.index)).fillna(0)
        + df.get("hydro_mwh",    pd.Series(0, index=df.index)).fillna(0)
        + df.get("biomass_mwh",  pd.Series(0, index=df.index)).fillna(0)
    )
    total = thermal + renewable
    df["renewable_ratio"]   = (renewable / total.replace(0, np.nan)).round(4)
    df["gas_share_thermal"] = (
        df.get("gas_gen_mwh", pd.Series(0, index=df.index)).fillna(0)
        / thermal.replace(0, np.nan)
    ).round(4)
    df.to_csv(cache)
    log.info(f"Saved clean generation ──> {cache}")
    return df

# ══════════════════════════════════════════════════════════
# 3. CONSUMPTION
# ══════════════════════════════════════════════════════════
def clean_consumption(path: Path) -> pd.DataFrame:
    cache = CORR_DIR / "consumption_clean.csv"
    raw = _read_smard(path)

    start_col = next(c for c in raw.columns if "start" in c.lower())
    load_col  = next(
        c for c in raw.columns
        if "grid load" in c.lower()
        and "pumped"   not in c.lower()
        and "incl"     not in c.lower()
    )

    df = raw[[start_col, load_col]].copy()
    df.columns = ["dt_raw", "load_mwh"]
    df["datetime"] = _parse_smard_dt(df["dt_raw"])
    df["load_mwh"] = pd.to_numeric(df["load_mwh"], errors="coerce")

    df = (
        df[["datetime", "load_mwh"]]
        .dropna(subset=["datetime"])
        .set_index("datetime")
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="first")]

    df.to_csv(cache)       
    log.info(f"Saved clean consumption ──> {cache}")
    return df

# ══════════════════════════════════════════════════════════
# 4. TEMPERATURE ARCHIVE FETCH
# ══════════════════════════════════════════════════════════
def fetch_temperature() -> pd.DataFrame:
    cache = CORR_DIR / "temperature_clean.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col="datetime", parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    cities = {
        "berlin":    (52.52, 13.41),
        "frankfurt": (50.11,  8.68),
        "munich":    (48.14, 11.58),
        "hamburg":   (53.55,  9.99),
    }
    frames = []
    start_fetch = (pd.to_datetime(START_DATE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for city, (lat, lon) in cities.items():
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_fetch}&end_date={END_DATE}"
            "&hourly=temperature_2m,wind_speed_10m"
            "&timezone=UTC"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()["hourly"]
        tmp = pd.DataFrame({
            "datetime":        pd.to_datetime(data["time"]),
            f"temp_{city}":    data["temperature_2m"],
            f"windspd_{city}": data["wind_speed_10m"],
        }).set_index("datetime")
        frames.append(tmp)
        log.info(f"  {city}: {len(tmp):,} rows")

    combined = pd.concat(frames, axis=1)
    temp_cols = [c for c in combined.columns if c.startswith("temp_")]
    wnd_cols  = [c for c in combined.columns if c.startswith("windspd_")]
    combined["temperature_c"] = combined[temp_cols].mean(axis=1).round(2)
    combined["wind_speed_ms"] = combined[wnd_cols].mean(axis=1).round(2)

    result = combined[["temperature_c", "wind_speed_ms"]].copy()
    result.index = (
        pd.to_datetime(result.index)
        .tz_localize("UTC")
        .tz_convert("Europe/Berlin")
        .tz_localize(None)
    )
    result.index.name = "datetime"
    result = result[START_DATE:END_DATE]

    result.to_csv(cache)      
    log.info(f"Saved clean temperature ──> {cache}")
    return result

# ══════════════════════════════════════════════════════════
# 5. TTF GAS PRICE FETCH 
# ══════════════════════════════════════════════════════════
def fetch_ttf() -> pd.DataFrame:
    cache = CORR_DIR / "ttf_clean.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col="datetime", parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    try:
        start_fetch = (pd.to_datetime(START_DATE) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        raw = yf.download(
            "TTF=F",
            start=start_fetch,
            end=END_DATE,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        log.warning(f"yfinance error: {e} — TTF values set to NaN")
        raw = pd.DataFrame()

    if raw.empty:
        df = pd.DataFrame({"ttf_eur_mwh": np.nan}, index=HOURLY_IDX)
        df.to_csv(cache)
        return df

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    s = raw["Close"].squeeze()
    s.name = "ttf_eur_mwh"
    s.index = (
        pd.to_datetime(s.index)
        .tz_localize("UTC")
        .tz_convert("Europe/Berlin")
        .tz_localize(None)
    )

    df = s.reindex(HOURLY_IDX, method="ffill").to_frame()

    df.to_csv(cache)          
    log.info(f"Saved clean TTF natural gas prices ──> {cache}")
    return df

# ══════════════════════════════════════════════════════════
# 6. MERGE CHANNELS 
# ══════════════════════════════════════════════════════════    
def merge_all(prices, generation, consumption, temperature, ttf) -> pd.DataFrame:
    master = (
        _align(prices)
        .join(_align(generation),   how="left")
        .join(_align(consumption),  how="left")
        .join(_align(temperature),  how="left")
        .join(_align(ttf),          how="left")
    )
    out = MERGED_DIR / "master_dataset.csv"
    master.to_csv(out)
    log.info(f"Saved merged dataset ──> {out}")
    return master


# ── MAIN EXECUTION ────────────────────────────────────────────────
def run_data_preparation() -> pd.DataFrame:
    for fname in ["smard_prices.csv", "smard_generation.csv", "smard_consumption.csv"]:
        p = RAW_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing raw SMARD source: {p}. Place download inside data/raw/.")
    prices      = clean_prices     (RAW_DIR / "smard_prices.csv")
    generation  = clean_generation (RAW_DIR / "smard_generation.csv")
    consumption = clean_consumption(RAW_DIR / "smard_consumption.csv")
    temperature = fetch_temperature()
    ttf         = fetch_ttf()
    master      = merge_all(prices, generation, consumption, temperature, ttf)
    log.info(f"Data preparation complete ──> master_dataset.csv ({master.shape[0]:,} rows × {master.shape[1]} columns)")
    return master
if __name__ == "__main__":
    run_data_preparation()