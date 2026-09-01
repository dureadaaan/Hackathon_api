"""
fertilizer_engine.py - Exact Nutrient Decomposition & Agronomic Response Engine
"""

from typing import Dict, Tuple, Optional

# Fertilizer grade specifications (% nutrient by weight) and standard field rates
FERTILIZER_REGIMES = {
    "No Fertilizer (Control)": {
        "description": "Unfertilized soil baseline (0 kg/ha)",
        "products": {},
        "nutrients_kg_ha": {"N": 0.0, "P2O5": 0.0, "K2O": 0.0, "S": 0.0, "Zn": 0.0}
    },
    "High Nitrogen (Urea + DAP)": {
        "description": "150 kg/ha Urea (46-0-0) + 100 kg/ha DAP (18-46-0)",
        "products": {"Urea": 150.0, "DAP": 100.0},
        "nutrients_kg_ha": {
            "N": (150.0 * 0.46) + (100.0 * 0.18),  # 69 + 18 = 87 kg/ha
            "P2O5": 100.0 * 0.46,                  # 46 kg/ha
            "K2O": 0.0,
            "S": 0.0,
            "Zn": 0.0
        }
    },
    "Balanced NPK (15-15-15)": {
        "description": "200 kg/ha Compound NPK (15-15-15)",
        "products": {"NPK_15_15_15": 200.0},
        "nutrients_kg_ha": {
            "N": 200.0 * 0.15,     # 30 kg/ha
            "P2O5": 200.0 * 0.15,  # 30 kg/ha
            "K2O": 200.0 * 0.15,   # 30 kg/ha
            "S": 0.0,
            "Zn": 0.0
        }
    },
    "Potash Heavy (SOP + Urea)": {
        "description": "100 kg/ha SOP (0-0-50 + 18% S) + 100 kg/ha Urea (46-0-0)",
        "products": {"SOP": 100.0, "Urea": 100.0},
        "nutrients_kg_ha": {
            "N": 100.0 * 0.46,     # 46 kg/ha
            "P2O5": 0.0,
            "K2O": 100.0 * 0.50,   # 50 kg/ha
            "S": 100.0 * 0.18,     # 18 kg/ha
            "Zn": 0.0
        }
    },
    "Zinc Fortified Blend": {
        "description": "200 kg/ha NPK (15-15-15) + 15 kg/ha ZnSO4 (33% Zn, 17% S)",
        "products": {"NPK_15_15_15": 200.0, "Zinc_Sulfate": 15.0},
        "nutrients_kg_ha": {
            "N": 200.0 * 0.15,            # 30 kg/ha
            "P2O5": 200.0 * 0.15,         # 30 kg/ha
            "K2O": 200.0 * 0.15,          # 30 kg/ha
            "S": 15.0 * 0.17,             # 2.55 kg/ha
            "Zn": 15.0 * 0.33             # 4.95 kg/ha
        }
    }
}

# Crop-specific response sensitivities (calibrated for Punjab field trials)
CROP_RESPONSE_COEFFICIENTS = {
    "Rice": {
        "beta": {"N": 0.0055, "P2O5": 0.0040, "K2O": 0.0025, "S": 0.0030, "Zn": 0.0150},
        "gamma": {"N": 0.000020, "P2O5": 0.000025, "K2O": 0.000015},
        "max_delta_t_ha": 0.90
    },
    "Wheat": {
        "beta": {"N": 0.0060, "P2O5": 0.0045, "K2O": 0.0020, "S": 0.0025, "Zn": 0.0120},
        "gamma": {"N": 0.000022, "P2O5": 0.000028, "K2O": 0.000012},
        "max_delta_t_ha": 0.85
    },
    "Cotton": {
        "beta": {"N": 0.0040, "P2O5": 0.0030, "K2O": 0.0050, "S": 0.0020, "Zn": 0.0080},
        "gamma": {"N": 0.000018, "P2O5": 0.000020, "K2O": 0.000025},
        "max_delta_t_ha": 0.65
    },
    "Sugarcane": {
        "beta": {"N": 0.0800, "P2O5": 0.0500, "K2O": 0.0600, "S": 0.0200, "Zn": 0.0400},
        "gamma": {"N": 0.000150, "P2O5": 0.000100, "K2O": 0.000120},
        "max_delta_t_ha": 12.0
    },
    "Maize_Autumn": {
        "beta": {"N": 0.0080, "P2O5": 0.0060, "K2O": 0.0035, "S": 0.0030, "Zn": 0.0180},
        "gamma": {"N": 0.000028, "P2O5": 0.000035, "K2O": 0.000020},
        "max_delta_t_ha": 1.20
    },
    "Maize_Spring": {
        "beta": {"N": 0.0085, "P2O5": 0.0065, "K2O": 0.0035, "S": 0.0030, "Zn": 0.0180},
        "gamma": {"N": 0.000030, "P2O5": 0.000038, "K2O": 0.000020},
        "max_delta_t_ha": 1.30
    }
}


def calculate_fertilizer_effect(
    crop: str,
    fertilizer_protocol: Optional[str] = None,
    custom_nutrients: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float], str]:
    clean_crop = crop.replace(" ", "_")
    
    matched_regime = None
    if fertilizer_protocol:
        proto_lower = fertilizer_protocol.lower()
        for name, data in FERTILIZER_REGIMES.items():
            name_lower = name.lower()
            if proto_lower in name_lower or name_lower in proto_lower:
                matched_regime = data
                break
            if "urea" in proto_lower and "dap" in proto_lower and "high nitrogen" in name_lower:
                matched_regime = data
                break
            if "15-15-15" in proto_lower and "zinc" not in proto_lower and "balanced" in name_lower:
                matched_regime = data
                break
            if "potash" in proto_lower or "sop" in proto_lower:
                if "potash" in name_lower:
                    matched_regime = data
                    break
            if "zinc" in proto_lower:
                if "zinc" in name_lower:
                    matched_regime = data
                    break
            if "control" in proto_lower or "none" in proto_lower or "no fertilizer" in proto_lower:
                if "control" in name_lower:
                    matched_regime = data
                    break

    if custom_nutrients:
        nutrients = {k: float(v) for k, v in custom_nutrients.items()}
        desc = "Custom User Formulation"
    elif matched_regime:
        nutrients = matched_regime["nutrients_kg_ha"]
        desc = matched_regime["description"]
    else:
        nutrients = FERTILIZER_REGIMES["No Fertilizer (Control)"]["nutrients_kg_ha"]
        desc = "Standard Regional Baseline (No explicit fertilizer delta)"

    coeffs = CROP_RESPONSE_COEFFICIENTS.get(
        clean_crop, 
        CROP_RESPONSE_COEFFICIENTS.get(crop, CROP_RESPONSE_COEFFICIENTS["Wheat"])
    )
    beta = coeffs["beta"]
    gamma = coeffs["gamma"]

    n = nutrients.get("N", 0.0)
    p = nutrients.get("P2O5", 0.0)
    k = nutrients.get("K2O", 0.0)
    s = nutrients.get("S", 0.0)
    zn = nutrients.get("Zn", 0.0)

    delta_n = (beta.get("N", 0.0) * n) - (gamma.get("N", 0.0) * (n ** 2))
    delta_p = (beta.get("P2O5", 0.0) * p) - (gamma.get("P2O5", 0.0) * (p ** 2))
    delta_k = (beta.get("K2O", 0.0) * k) - (gamma.get("K2O", 0.0) * (k ** 2))
    delta_s = beta.get("S", 0.0) * s
    delta_zn = beta.get("Zn", 0.0) * zn

    synergy = 0.10 * min(delta_n, delta_p) if (n > 0 and p > 0 and delta_n > 0 and delta_p > 0) else 0.0

    total_delta = max(0.0, delta_n + delta_p + delta_k + delta_s + delta_zn + synergy)
    capped_delta = min(total_delta, coeffs["max_delta_t_ha"])

    return round(capped_delta, 4), {k: round(v, 2) for k, v in nutrients.items()}, desc
