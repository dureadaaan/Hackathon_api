"""
===================================================================
SIX-CROP YIELD PREDICTION ENGINE
===================================================================
Built for the models in six_crop_extratrees_models/, which were
trained on RAW current-season satellite/weather/soil values
(NOT the lagged/multi-year features predict.py uses for the
ex_ante_crop_models/ models).

Usage is the same shape as predict.py's predict_yield():
    from six_crop_predict import predict_yield_six_crop
    result = predict_yield_six_crop(
        crop="Wheat",
        district="Faisalabad",
        sowing_date="11/15/2025",
        fertilizer_protocol="Balanced NPK (15-15-15)"
    )
===================================================================
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from fertilizer_engine import calculate_fertilizer_effect

DATASET_PATH = "punjab_model1_master_with_s1_weather_soil.csv"
MODEL_DIR = Path("six_crop_extratrees_models")
SETTINGS_PATH = MODEL_DIR / "six_crop_settings.pkl"

TON_HA_TO_MAUND_ACRE = 10.12705

# The exact raw numerical columns these models expect (from six_crop_settings.pkl)
NUMERICAL_FEATURES = [
    "year_model",
    "NDVI_mean", "NDVI_max", "EVI_mean", "EVI_max",
    "NDRE_mean", "NDRE_max", "GNDVI_mean", "GNDVI_max",
    "NDWI_mean", "NDWI_max",
    "VV_mean", "VV_max", "VH_mean", "VH_max",
    "VV_VH_mean", "VV_VH_max",
    "rainfall_total_mm", "mean_temperature_C",
    "minimum_temperature_C", "maximum_temperature_C",
    "relative_humidity_mean", "solar_radiation_MJ_m2",
    "clay_pct", "sand_pct", "soil_organic_carbon_g_kg",
    "soil_pH", "soil_water_content_pct"
]
CATEGORICAL_FEATURES = ["district"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

_CACHED_DF = None
_CACHED_SETTINGS = None


def sanitize_name(crop_name: str) -> str:
    """Match the model filename convention used by 6_cropy.py."""
    clean = crop_name.replace("&", "and")
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", "_", clean.strip())
    if crop_name == "Rapeseed & Mustard":
        return "Rapeseed_Mustard"
    if crop_name == "Maize (Autumn)":
        return "Maize_Autumn"
    if crop_name == "Maize (Spring)":
        return "Maize_Spring"
    return clean


def get_dataset():
    global _CACHED_DF
    if _CACHED_DF is None:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        df["year_model"] = df["year"].apply(
            lambda x: int(str(x).split("-")[0]) if pd.notna(x) else np.nan
        )
        _CACHED_DF = df
    return _CACHED_DF


def get_settings():
    global _CACHED_SETTINGS
    if _CACHED_SETTINGS is None and SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "rb") as f:
            _CACHED_SETTINGS = pickle.load(f)
    return _CACHED_SETTINGS


def get_supported_crops():
    """Returns the actual list of crops these models were trained on."""
    settings = get_settings()
    if settings and "crops" in settings:
        return settings["crops"]
    return []


def load_model(crop_name: str):
    sanitized = sanitize_name(crop_name)
    path = MODEL_DIR / f"{sanitized}_ExtraTrees.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No six-crop model found for '{crop_name}'. "
            f"Supported crops: {get_supported_crops()}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_year_from_input(year=None, sowing_date=None) -> int:
    if year is not None:
        return int(year)
    if sowing_date:
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(sowing_date, fmt).year
            except ValueError:
                continue
    return datetime.now().year


def predict_yield_six_crop(
    crop: str,
    district: str,
    year: int = None,
    sowing_date: str = None,
    fertilizer_protocol: str = None,
    cultivar: str = None,
    variety: str = None,
    custom_nutrients: dict = None,
    npk_ratio: dict = None
) -> dict:
    """
    Predicts yield using the six_crop_extratrees_models (raw current-season
    features, no lagging). Only supports the crops in get_supported_crops().
    """
    supported = get_supported_crops()
    if supported and crop not in supported:
        raise ValueError(
            f"'{crop}' is not supported by the six-crop models. "
            f"Supported crops: {supported}"
        )

    target_year = parse_year_from_input(year=year, sowing_date=sowing_date)
    df = get_dataset()

    subset = df[
        (df["district"].str.lower() == district.lower()) &
        (df["crop"].str.lower() == crop.lower())
    ].sort_values("year_model")

    if len(subset) == 0:
        raise ValueError(f"No data found for crop '{crop}' in district '{district}'.")

    # Prefer an exact match for the target year with real satellite data.
    exact_rows = subset[(subset["year_model"] == target_year) & (subset["NDVI_mean"].notna())]

    if len(exact_rows) > 0:
        source_row = exact_rows.iloc[0]
        execution_mode = "DIRECT (Actual Satellite/Weather/Soil Data)"
    else:
        # Fallback: carry forward the most recent available year's raw readings.
        prior = subset[subset["year_model"] < target_year]
        if len(prior) == 0:
            prior = subset
        source_row = prior.iloc[-1]
        execution_mode = f"FALLBACK (Carried Forward from {int(source_row['year_model'])})"

    # Build the exact feature row this model expects
    feature_row = {"district": district, "year_model": target_year}
    for feat in NUMERICAL_FEATURES:
        if feat == "year_model":
            continue
        feature_row[feat] = float(source_row[feat]) if feat in source_row and pd.notna(source_row[feat]) else 0.0

    input_df = pd.DataFrame([feature_row])[ALL_FEATURES]

    # 1. Baseline ML prediction
    pipeline = load_model(crop)
    pred_log = pipeline.predict(input_df)[0]
    baseline_yield_t_ha = max(float(np.expm1(pred_log)), 0.0)

    # 2. Fertilizer uplift (same engine as predict.py)
    delta_yield_t_ha, nutrients_applied, fert_desc = calculate_fertilizer_effect(
        crop=crop,
        fertilizer_protocol=fertilizer_protocol,
        custom_nutrients=custom_nutrients or npk_ratio
    )

    final_yield_t_ha = baseline_yield_t_ha + delta_yield_t_ha

    baseline_maund_acre = baseline_yield_t_ha * TON_HA_TO_MAUND_ACRE
    delta_maund_acre = delta_yield_t_ha * TON_HA_TO_MAUND_ACRE
    final_yield_maund = final_yield_t_ha * TON_HA_TO_MAUND_ACRE

    historical_benchmark = float(subset["yield_t_ha"].mean()) if "yield_t_ha" in subset else final_yield_t_ha
    historical_benchmark_maund = historical_benchmark * TON_HA_TO_MAUND_ACRE

    selected_cultivar = cultivar or variety or "Standard Variety"
    selected_fert = fertilizer_protocol or "No Fertilizer (Control)"

    return {
        "status": "success",
        "crop": crop,
        "district": district,
        "year": target_year,
        "sowing_date": sowing_date or f"{target_year}-06-01",
        "cultivar": selected_cultivar,
        "fertilizer_protocol": selected_fert,
        "fertilizer_description": fert_desc,
        "nutrients_applied_kg_ha": nutrients_applied,
        "execution_mode": execution_mode,
        "baseline_yield_t_ha": round(baseline_yield_t_ha, 4),
        "baseline_yield_maund_acre": round(baseline_maund_acre, 2),
        "fertilizer_effect_t_ha": round(delta_yield_t_ha, 4),
        "fertilizer_effect_maund_acre": round(delta_maund_acre, 2),
        "predicted_yield_t_ha": round(final_yield_t_ha, 4),
        "predicted_yield_maund_acre": round(final_yield_maund, 2),
        "district_benchmark_t_ha": round(historical_benchmark, 4),
        "district_benchmark_maund_acre": round(historical_benchmark_maund, 2),
        "yield_vs_benchmark_pct": round(
            ((final_yield_t_ha - historical_benchmark) / (historical_benchmark + 1e-6)) * 100, 2
        )
    }
