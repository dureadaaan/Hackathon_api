"""
===================================================================
GENERIC CROP YIELD PREDICTION ENGINE (API & CLI READY)
===================================================================

Designed for seamless integration with web apps, mobile apps, and REST APIs.
Accepts dynamic payloads (crop, district, sowing_date / year, fertilizer_protocol, cultivar)
and auto-detects satellite data availability:
  - DIRECT MODE: If live Sentinel-1 & 2 NDVI is available in the database.
  - FALLBACK MODE: If future/pre-season, estimates NDVI from multi-year historical trajectories.
  - FERTILIZER AGRO-RESPONSE: Calculates exact active N, P2O5, K2O, S, Zn doses
    and applies quadratic diminishing return uplift: Y_final = Y_baseline + Delta_Y
===================================================================
"""

import os
import re
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import exact fertilizer engine
from fertilizer_engine import calculate_fertilizer_effect, FERTILIZER_REGIMES

# Base paths
DATASET_PATH = "punjab_model1_master_with_s1_weather_soil.csv"
MODEL_DIR = Path("six_crop_extratrees_models")

# 1 metric ton / ha = 10.12705 maunds / acre
TON_HA_TO_MAUND_ACRE = 10.12705

RS_FEATURES = [
    "NDVI_mean", "NDVI_max", "EVI_mean", "EVI_max",
    "NDRE_mean", "NDRE_max", "GNDVI_mean", "GNDVI_max",
    "NDWI_mean", "NDWI_max", "VV_mean", "VV_max",
    "VH_mean", "VH_max", "VV_VH_mean", "VV_VH_max"
]

WEATHER_FEATURES = [
    "rainfall_total_mm", "mean_temperature_C",
    "minimum_temperature_C", "maximum_temperature_C",
    "relative_humidity_mean", "solar_radiation_MJ_m2"
]

SOIL_FEATURES = [
    "clay_pct", "sand_pct", "soil_organic_carbon_g_kg",
    "soil_pH", "soil_water_content_pct"
]

# Cache the dataset in memory for fast API responses
_CACHED_DF = None


def sanitize_name(crop_name: str) -> str:
    """Normalize crop name to match model filename convention."""
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
    """Load and cache historical dataset."""
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


def load_model(crop_name: str):
    """Load the pickled model pipeline for a given crop."""
    sanitized = sanitize_name(crop_name)
    candidates = [
        MODEL_DIR / f"{sanitized}_ExtraTrees.pkl",
        MODEL_DIR / f"{sanitized}_xgboost.pkl",
        MODEL_DIR / f"{sanitized}.pkl",
        Path("six_crop_extratrees_models") / f"{sanitized}_ExtraTrees.pkl"
    ]
    for cand in candidates:
        if cand.exists():
            with open(cand, "rb") as f:
                return pickle.load(f)

    # Case-insensitive fallback
    for d in [MODEL_DIR, Path("six_crop_extratrees_models"), Path("remaining_crop_extratrees")]:
        if d.exists():
            for file in d.glob("*.pkl"):
                if sanitized.lower() in file.name.lower() and "settings" not in file.name.lower():
                    with open(file, "rb") as f:
                        return pickle.load(f)

    raise FileNotFoundError(f"No model found for crop '{crop_name}'")


def parse_year_from_input(year=None, sowing_date=None) -> int:
    """Extracts integer year from either year parameter or sowing_date string."""
    if year is not None:
        try:
            return int(year)
        except (ValueError, TypeError):
            pass

    if sowing_date:
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
            try:
                dt = datetime.strptime(str(sowing_date).strip(), fmt)
                return dt.year
            except ValueError:
                continue

    # Default to current/next calendar year
    return datetime.now().year


def predict_yield(
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
    Main Generic Prediction function called by the App / API.
    
    Parameters:
      crop: Crop name (e.g. 'Rice', 'Wheat', 'Sugarcane', 'Cotton')
      district: District in Punjab (e.g. 'Gujranwala', 'Multan', 'Faisalabad')
      year: Target year (e.g. 2026)
      sowing_date: Date string from UI calendar (e.g. '06/15/2026')
      fertilizer_protocol: Selected regime from UI (e.g. 'Balanced NPK (15-15-15)')
      cultivar / variety: Cultivar name (e.g. 'Super Basmati', 'Inqilab')
      custom_nutrients: Optional dict with N, P2O5, K2O, S, Zn doses
      
    Returns:
      JSON-compatible dictionary with complete yield estimation, baseline, fertilizer delta, and metadata.
    """
    target_year = parse_year_from_input(year=year, sowing_date=sowing_date)
    df = get_dataset()

    # Filter district & crop history
    subset = df[
        (df["district"].str.lower() == district.lower()) &
        (df["crop"].str.lower() == crop.lower())
    ].sort_values("year_model")

    if len(subset) == 0:
        # Fallback to district average if crop is rare in that district
        subset = df[df["district"].str.lower() == district.lower()].sort_values("year_model")

    if len(subset) == 0:
        raise ValueError(f"District '{district}' not recognized in Punjab database.")

    # Check if actual satellite data is available for requested year
    actual_rows = subset[(subset["year_model"] == target_year) & (subset["NDVI_mean"].notna())]
    is_direct_mode = len(actual_rows) > 0

    prior_data = subset[subset["year_model"] < target_year]
    
    # Live GEE Check
    try:
        from gee_live_fetcher import auto_fetch_and_cache_passed_year
        live_satellite = auto_fetch_and_cache_passed_year(district, crop, target_year)
    except Exception:
        live_satellite = None

    if len(prior_data) == 0:
        prior_data = subset

    latest_prior = prior_data.iloc[-1]
    prev_prior = prior_data.iloc[-2] if len(prior_data) > 1 else latest_prior
    past_3_years = prior_data.tail(3)

    # Build Feature Vector for Baseline ML Model
    feature_row = {
        "district": district,
        "crop": crop,
        "year_model": target_year
    }

    # Soil features
    soil_profile = {}
    for feat in SOIL_FEATURES:
        val = float(latest_prior[feat]) if feat in latest_prior else 0.0
        feature_row[feat] = val
        soil_profile[feat] = round(val, 2)

    # Weather features
    for feat in WEATHER_FEATURES:
        feature_row[f"{feat}_hist_mean"] = float(prior_data[feat].mean()) if feat in prior_data else 0.0
        feature_row[f"{feat}_lag1"] = float(latest_prior[feat]) if feat in latest_prior else 0.0

    # Satellite features (Direct vs Fallback)
    if is_direct_mode:
        actual_row = actual_rows.iloc[0]
        execution_mode = "DIRECT (Actual Satellite NDVI)"
        ndvi_used = float(actual_row["NDVI_mean"])
        feature_row["NDVI_mean"] = ndvi_used
        feature_row["NDVI_max"] = float(actual_row["NDVI_max"])
        for feat in RS_FEATURES:
            feature_row[f"{feat}_lag1"] = float(latest_prior[feat])
            feature_row[f"{feat}_lag2"] = float(prev_prior[feat])
            feature_row[f"{feat}_3yr_mean"] = float(past_3_years[feat].mean())
            feature_row[f"{feat}_delta_1yr"] = float(latest_prior[feat] - prev_prior[feat])
    else:
        execution_mode = "FALLBACK (Estimated NDVI from Trajectory)"
        val_t1 = latest_prior["NDVI_mean"]
        val_t2 = prev_prior["NDVI_mean"]
        roll_3 = past_3_years["NDVI_mean"].mean()
        delta = val_t1 - val_t2
        
        pred_ndvi = (0.60 * val_t1) + (0.40 * roll_3) + (0.50 * delta)
        pred_ndvi = np.clip(pred_ndvi, prior_data["NDVI_mean"].min(), prior_data["NDVI_mean"].max())
        ndvi_used = float(pred_ndvi)
        feature_row["NDVI_mean"] = ndvi_used
        feature_row["NDVI_max"] = float(latest_prior["NDVI_max"])
        
        for feat in RS_FEATURES:
            val_1 = float(latest_prior[feat])
            val_2 = float(prev_prior[feat])
            feature_row[f"{feat}_lag1"] = val_1
            feature_row[f"{feat}_lag2"] = val_2
            feature_row[f"{feat}_3yr_mean"] = float(past_3_years[feat].mean())
            feature_row[f"{feat}_delta_1yr"] = float(val_1 - val_2)

    # 1. Base ML Model Inference (Environmental Baseline Y0)
    pipeline = load_model(crop)
    input_df = pd.DataFrame([feature_row])
    pred_log = pipeline.predict(input_df)[0]
    baseline_yield_t_ha = max(float(np.expm1(pred_log)), 0.0)

    # 2. Agronomic Fertilizer Response Delta (Delta Y)
    delta_yield_t_ha, nutrients_applied, fert_desc = calculate_fertilizer_effect(
        crop=crop,
        fertilizer_protocol=fertilizer_protocol,
        custom_nutrients=custom_nutrients or npk_ratio
    )

    # 3. Final Combined Predicted Yield (Y_final = Y_baseline + Delta_Y)
    final_yield_t_ha = baseline_yield_t_ha + delta_yield_t_ha

    # Unit Conversions (Maunds per Acre)
    baseline_maund_acre = baseline_yield_t_ha * TON_HA_TO_MAUND_ACRE
    delta_maund_acre = delta_yield_t_ha * TON_HA_TO_MAUND_ACRE
    final_yield_maund = final_yield_t_ha * TON_HA_TO_MAUND_ACRE

    # District Historical Benchmark
    historical_benchmark = float(prior_data["yield_t_ha"].mean()) if "yield_t_ha" in prior_data else final_yield_t_ha
    historical_benchmark_maund = historical_benchmark * TON_HA_TO_MAUND_ACRE

    selected_cultivar = cultivar or variety or "Standard Variety"
    selected_fert = fertilizer_protocol or "No Fertilizer (Control)"
    season_group = str(latest_prior.get("season_group", "General Season"))

    return {
        "status": "success",
        "crop": crop,
        "district": district,
        "year": target_year,
        "sowing_date": sowing_date or f"{target_year}-06-01",
        "season_group": season_group,
        "cultivar": selected_cultivar,
        "fertilizer_protocol": selected_fert,
        "fertilizer_description": fert_desc,
        "nutrients_applied_kg_ha": nutrients_applied,
        "execution_mode": execution_mode,
        "ndvi_used": round(ndvi_used, 4),
        "soil_profile": soil_profile,
        "baseline_yield_t_ha": round(baseline_yield_t_ha, 4),
        "baseline_yield_maund_acre": round(baseline_maund_acre, 2),
        "fertilizer_effect_t_ha": round(delta_yield_t_ha, 4),
        "fertilizer_effect_maund_acre": round(delta_maund_acre, 2),
        "predicted_yield_t_ha": round(final_yield_t_ha, 4),
        "predicted_yield_maund_acre": round(final_yield_maund, 2),
        "district_benchmark_t_ha": round(historical_benchmark, 4),
        "district_benchmark_maund_acre": round(historical_benchmark_maund, 2),
        "yield_vs_benchmark_pct": round(((final_yield_t_ha - historical_benchmark) / (historical_benchmark + 1e-6)) * 100, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Generic Crop Yield Prediction Engine with Fertilizer Response")
    parser.add_argument("--crop", type=str, default="Rice", help="Crop name (e.g. Rice, Wheat, Cotton, Sugarcane)")
    parser.add_argument("--district", type=str, default="Gujranwala", help="District (e.g. Gujranwala, Multan, Faisalabad)")
    parser.add_argument("--year", type=int, default=2026, help="Target Year (e.g. 2026)")
    parser.add_argument("--sowing_date", type=str, default="06/15/2026", help="Sowing Date (e.g. 06/15/2026)")
    parser.add_argument("--cultivar", type=str, default="Super Basmati", help="Crop variety (e.g. Super Basmati)")
    parser.add_argument("--fertilizer", type=str, default="Balanced NPK (15-15-15)", help="Fertilizer protocol")
    parser.add_argument("--json", action="store_true", help="Output raw JSON for API integration")

    args = parser.parse_args()

    result = predict_yield(
        crop=args.crop,
        district=args.district,
        year=args.year,
        sowing_date=args.sowing_date,
        cultivar=args.cultivar,
        fertilizer_protocol=args.fertilizer
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        nuts = result["nutrients_applied_kg_ha"]
        nut_str = f"N: {nuts['N']} | P2O5: {nuts['P2O5']} | K2O: {nuts['K2O']} | S: {nuts['S']} | Zn: {nuts['Zn']}"
        print("\n" + "=" * 78)
        print(f"  [FIELD LEDGER SIMULATION RESULT] - {result['crop'].upper()} ({result['cultivar']})")
        print("=" * 78)
        print(f"  Location / District        : {result['district']} ({result['season_group']})")
        print(f"  Planting Calendar          : {result['sowing_date']} (Target Year: {result['year']})")
        print(f"  Execution Mode             : {result['execution_mode']}")
        print(f"  Vegetative Index (NDVI)    : {result['ndvi_used']}")
        print("-" * 78)
        print(f"  Fertilizer Regime          : {result['fertilizer_protocol']}")
        print(f"  Regime Breakdown           : {result['fertilizer_description']}")
        print(f"  Active Nutrients (kg/ha)   : {nut_str}")
        print("-" * 78)
        print(f"  BASELINE YIELD (No Fert)   : {result['baseline_yield_t_ha']} t/ha  ({result['baseline_yield_maund_acre']} Maunds/Acre)")
        print(f"  EXPECTED FERTILIZER EFFECT : +{result['fertilizer_effect_t_ha']} t/ha (+{result['fertilizer_effect_maund_acre']} Maunds/Acre)")
        print(f"  ------------------------------------------------------------------------")
        print(f"  ESTIMATED FINAL YIELD      : {result['predicted_yield_t_ha']} t/ha  ({result['predicted_yield_maund_acre']} Maunds/Acre)")
        print(f"  District Historical Avg    : {result['district_benchmark_maund_acre']} Maunds/Acre ({result['yield_vs_benchmark_pct']:+0.2f}% vs District Avg)")
        print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
