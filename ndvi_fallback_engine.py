"""
===================================================================
DYNAMIC SATELLITE & NDVI FALLBACK ENGINE
===================================================================

Non-hardcoded module that:
1. Detects whether live/actual Sentinel-1 & Sentinel-2 features are provided.
2. If available: Uses the actual ground-truth satellite measurements directly.
3. If unavailable (missing/future/pre-season): Dynamically estimates the
   missing vegetative & radar features from historical multi-year trajectories.
===================================================================
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

HISTORICAL_DATASET_PATH = Path("punjab_model1_master_with_s1_weather_soil.csv")

RS_FEATURES = [
    "NDVI_mean", "NDVI_max",
    "EVI_mean", "EVI_max",
    "NDRE_mean", "NDRE_max",
    "GNDVI_mean", "GNDVI_max",
    "NDWI_mean", "NDWI_max",
    "VV_mean", "VV_max",
    "VH_mean", "VH_max",
    "VV_VH_mean", "VV_VH_max"
]

WEATHER_FEATURES = [
    "rainfall_total_mm",
    "mean_temperature_C",
    "minimum_temperature_C",
    "maximum_temperature_C",
    "relative_humidity_mean",
    "solar_radiation_MJ_m2"
]

SOIL_FEATURES = [
    "clay_pct",
    "sand_pct",
    "soil_organic_carbon_g_kg",
    "soil_pH",
    "soil_water_content_pct"
]

class NDVIFallbackEngine:
    def __init__(self, historical_path: Path = HISTORICAL_DATASET_PATH):
        self.historical_path = historical_path
        self._load_history()
        
    def _load_history(self):
        if not self.historical_path.exists():
            raise FileNotFoundError(f"Historical dataset not found at: {self.historical_path}")
            
        df = pd.read_csv(self.historical_path)
        df["year_model"] = df["year"].apply(self._extract_year)
        df = df.dropna(subset=["year_model", "district", "crop"]).copy()
        df["year_model"] = df["year_model"].astype(int)
        df = df.sort_values(by=["district", "crop", "year_model"]).reset_index(drop=True)
        self.history_df = df
        
    @staticmethod
    def _extract_year(value):
        try:
            return int(str(value).split("-")[0])
        except (ValueError, IndexError):
            return np.nan

    def is_satellite_available(self, row_or_df: pd.DataFrame) -> bool:
        """Check if actual Sentinel features are present and non-NaN."""
        for feat in ["NDVI_mean", "NDVI_max", "EVI_mean", "VV_mean"]:
            if feat not in row_or_df.columns or row_or_df[feat].isna().any():
                return False
        return True

    def estimate_satellite_values(self, district: str, crop: str, target_year: int) -> dict:
        """
        Dynamically estimate Sentinel-1 and Sentinel-2 features for (district, crop, target_year)
        using damped autoregressive trend + rolling baseline bounded by historical limits.
        """
        subset = self.history_df[
            (self.history_df["district"].str.lower() == district.lower()) &
            (self.history_df["crop"].str.lower() == crop.lower())
        ].sort_values("year_model")
        
        # If no crop-specific history, fallback to district level average
        if len(subset) == 0:
            subset = self.history_df[
                self.history_df["district"].str.lower() == district.lower()
            ].sort_values("year_model")
            
        if len(subset) == 0:
            # Fallback to global average if district not found
            subset = self.history_df.sort_values("year_model")
            
        # Check if target_year - 1 has passed in real time but is missing from history
        prior_data = subset[subset["year_model"] < target_year]
        
        # Try live Google Earth Engine fetch if target_year - 1 is missing
        try:
            from gee_live_fetcher import auto_fetch_and_cache_passed_year
            live_sat = auto_fetch_and_cache_passed_year(district, crop, target_year)
            if live_sat:
                return live_sat
        except Exception:
            pass

        if len(prior_data) == 0:
            prior_data = subset
            
        latest_row = prior_data.iloc[-1]
        prev_row = prior_data.iloc[-2] if len(prior_data) > 1 else latest_row
        past_3 = prior_data.tail(3)
        
        estimated = {}
        for feat in RS_FEATURES:
            if feat in prior_data.columns:
                val_t1 = latest_row[feat]
                val_t2 = prev_row[feat]
                rolling_mean = past_3[feat].mean()
                delta = val_t1 - val_t2
                
                # Predictive formula: 60% recent state + 40% rolling baseline + 50% momentum
                pred_val = (0.60 * val_t1) + (0.40 * rolling_mean) + (0.50 * delta)
                
                # Physical bounds clipping based on historical min/max for this district & crop
                min_bound = prior_data[feat].min() * 0.90 if prior_data[feat].min() > 0 else prior_data[feat].min() * 1.10
                max_bound = prior_data[feat].max() * 1.10 if prior_data[feat].max() > 0 else prior_data[feat].max() * 0.90
                
                estimated[feat] = float(np.clip(pred_val, min_bound, max_bound))
            else:
                estimated[feat] = 0.0
                
        # Also provide soil and climatology weather if missing
        for feat in SOIL_FEATURES:
            if feat in latest_row:
                estimated[feat] = float(latest_row[feat])
                
        for feat in WEATHER_FEATURES:
            if feat in prior_data:
                estimated[feat] = float(prior_data[feat].mean())
                
        return estimated

    def enrich_dataframe(self, df: pd.DataFrame, target_year: int = 2025) -> pd.DataFrame:
        """
        Enriches an input dataframe. If satellite values exist, keeps them.
        If missing, uses the dynamic fallback to synthesize them.
        Also populates ex-ante lag & rolling features required by model pipelines.
        Adds a 'satellite_data_source' column indicating 'ACTUAL' or 'ESTIMATED_FALLBACK'.
        """
        enriched_rows = []
        missing_count = 0
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            district = str(row_dict.get("district", ""))
            crop = str(row_dict.get("crop", ""))
            
            if "year_model" in row_dict and pd.notna(row_dict["year_model"]):
                year = int(row_dict["year_model"])
            elif "year" in row_dict and pd.notna(row_dict["year"]):
                year = self._extract_year(row_dict["year"])
                if pd.isna(year):
                    year = target_year
                else:
                    year = int(year)
            else:
                year = target_year
            row_dict["year_model"] = year
            
            # Check if primary satellite features are missing
            has_sat = True
            for feat in ["NDVI_mean", "NDVI_max", "EVI_mean", "VV_mean"]:
                if feat not in row_dict or pd.isna(row_dict.get(feat)):
                    has_sat = False
                    break
                    
            if not has_sat:
                missing_count += 1
                row_dict["satellite_data_source"] = "ESTIMATED_FALLBACK"
            else:
                if "satellite_data_source" not in row_dict:
                    row_dict["satellite_data_source"] = "ACTUAL"
                    
            estimated = self.estimate_satellite_values(district, crop, year)
            for feat, val in estimated.items():
                if feat not in row_dict or pd.isna(row_dict.get(feat)):
                    row_dict[feat] = val
                    
            # Populate lag features
            subset = self.history_df[
                (self.history_df["district"].str.lower() == district.lower()) &
                (self.history_df["crop"].str.lower() == crop.lower())
            ].sort_values("year_model")
            
            if len(subset) == 0:
                subset = self.history_df[
                    self.history_df["district"].str.lower() == district.lower()
                ].sort_values("year_model")
            if len(subset) == 0:
                subset = self.history_df.sort_values("year_model")
                
            prior_data = subset[subset["year_model"] < year]
            if len(prior_data) == 0:
                prior_data = subset
                
            latest_row = prior_data.iloc[-1]
            prev_row = prior_data.iloc[-2] if len(prior_data) > 1 else latest_row
            past_3 = prior_data.tail(3)
            
            for feat in RS_FEATURES:
                val_t1 = latest_row[feat] if feat in latest_row else row_dict[feat]
                val_t2 = prev_row[feat] if feat in prev_row else val_t1
                roll_3 = past_3[feat].mean() if feat in past_3 else val_t1
                
                row_dict.setdefault(f"{feat}_lag1", val_t1)
                row_dict.setdefault(f"{feat}_lag2", val_t2)
                row_dict.setdefault(f"{feat}_3yr_mean", roll_3)
                row_dict.setdefault(f"{feat}_delta_1yr", val_t1 - val_t2)
                
            for feat in WEATHER_FEATURES:
                row_dict.setdefault(f"{feat}_hist_mean", prior_data[feat].mean() if feat in prior_data else 0.0)
                row_dict.setdefault(f"{feat}_lag1", latest_row[feat] if feat in latest_row else 0.0)
                
            for feat in SOIL_FEATURES:
                row_dict.setdefault(feat, latest_row[feat] if feat in latest_row else 0.0)
                
            enriched_rows.append(row_dict)
            
        if missing_count > 0:
            print(f"[*] [Fallback Engine]: Satellite NDVI/S1 not available for {missing_count} rows. Dynamically estimated from historical trajectories.")
            
        return pd.DataFrame(enriched_rows)

if __name__ == "__main__":
    engine = NDVIFallbackEngine()
    test_est = engine.estimate_satellite_values("Multan", "Wheat", 2025)
    print("\n[OK] Sample 2025 estimated satellite features for Multan - Wheat:")
    for k in ["NDVI_mean", "NDVI_max", "EVI_mean", "NDRE_mean", "VV_mean", "VH_mean"]:
        print(f"  - {k:12s}: {test_est[k]:.4f}")
