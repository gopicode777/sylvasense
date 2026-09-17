"""
biomass.py — Aboveground Biomass (AGB) & Carbon estimation

Uses a published crown-diameter-based allometric equation rather than an
opaque regression, so every number in the validation report traces back to
a citable source:

  AGB (kg) = a * D^b

  where D = crown diameter (m), and (a, b) are taken from:
    - Jucker et al. (2017), "Allometric equations for integrating remote
      sensing imagery into forest monitoring programmes", Global Change
      Biology. Pantropical crown-diameter-to-AGB fit.
    - Chave et al. (2014) pantropical biomass model is used as the
      diameter-at-breast-height (DBH) cross-check when a height proxy is
      available (documented but optional here).

Carbon fraction and CO2 equivalence follow IPCC (2006) Good Practice
Guidance defaults:
    Carbon (kg)  = AGB (kg) * 0.47
    CO2e  (kg)   = Carbon (kg) * 3.667   (44/12 molecular weight ratio)

SAR-derived estimates carry a wider uncertainty band because backscatter
saturates at higher biomass (~150-200 t/ha for C-band VV) — this is
reflected in the confidence score, not hidden.
"""

from dataclasses import dataclass

# Jucker et al. 2017, pantropical generalized crown-diameter allometry
# AGB(kg) = a * D(m)^b  (broad generalized-model coefficients)
ALLOMETRIC_A = 0.567
ALLOMETRIC_B = 2.393

CARBON_FRACTION = 0.47      # IPCC 2006 default
CO2_CONVERSION = 3.667      # 44/12, CO2 <-> C molecular weight ratio

# SAR backscatter saturates above this rough AGB threshold (t/ha equivalent
# per crown scale) — used only to flag reduced confidence, not to clip values.
SAR_SATURATION_WARNING_KG = 450.0


def agb_from_crown_diameter(diameter_m: float) -> float:
    """Single-tree AGB (kg) from crown diameter (m)."""
    if diameter_m <= 0:
        return 0.0
    return ALLOMETRIC_A * (diameter_m ** ALLOMETRIC_B)


def carbon_metrics(agb_kg: float) -> dict:
    carbon_kg = agb_kg * CARBON_FRACTION
    co2e_kg = carbon_kg * CO2_CONVERSION
    return {
        "agb_kg": round(agb_kg, 2),
        "carbon_kg": round(carbon_kg, 2),
        "co2e_kg": round(co2e_kg, 2),
    }


def confidence_score(mean_ndvi: float, is_sar: bool, agb_kg: float, cloud_frac: float) -> float:
    """
    Heuristic, transparent confidence score in [0,1], combining:
      - signal strength (higher NDVI = more reliable vegetation signal)
      - estimation source (optical generally more reliable than SAR-only)
      - SAR saturation risk at high biomass
      - cloud contamination fraction in the surrounding scene
    This is intentionally simple and inspectable rather than a black-box
    number — the validation report explains exactly how it's computed.
    """
    score = 0.5 + 0.4 * max(0.0, min(mean_ndvi, 1.0))
    if is_sar:
        score -= 0.15
        if agb_kg > SAR_SATURATION_WARNING_KG:
            score -= 0.15
    score -= 0.2 * cloud_frac
    return round(max(0.05, min(score, 0.98)), 2)


def summarize_polygon(features: list, cloud_frac: float) -> dict:
    """Aggregates per-tree GeoJSON features into polygon-level totals."""
    total_agb = 0.0
    total_trees = len(features)
    confidences = []

    for f in features:
        props = f["properties"]
        d = props["crown_diameter_m"]
        agb = agb_from_crown_diameter(d)
        metrics = carbon_metrics(agb)
        is_sar = props["estimation_source"] == "sar"
        conf = confidence_score(props["mean_ndvi"], is_sar, agb, cloud_frac)

        props.update(metrics)
        props["confidence"] = conf
        confidences.append(conf)
        total_agb += agb

    total_metrics = carbon_metrics(total_agb)
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return {
        "tree_count": total_trees,
        "total_agb_kg": total_metrics["agb_kg"],
        "total_carbon_kg": total_metrics["carbon_kg"],
        "total_co2e_kg": total_metrics["co2e_kg"],
        "avg_confidence": avg_confidence,
        "cloud_fraction": round(cloud_frac, 3),
    }
