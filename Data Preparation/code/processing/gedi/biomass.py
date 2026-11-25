"""GEDI rh98 (m) -> above-ground biomass (Mg/ha) for Sabah mangrove labels.

Both equations are from Simard et al. (2019), Nature Geoscience 12, 40-45,
https://doi.org/10.1038/s41561-018-0279-1, Supplementary Table 8.

MAIN   Southeast Asia generic (Myanmar to Hawaii; R2 0.55, RMSE 148 Mg/ha)
           AGB = exp(3.9042 + 0.0858 * H)
       H is a canopy-height-model (top-of-canopy) height, so GEDI rh98 is used directly.
       Not a power law: it gives 49.6 Mg/ha at H = 0 and grows without limit, so H is capped at
       H_CAP (see below).

SENS   Global Hmax power model (R2 0.70, RMSE 180 Mg/ha)
           AGB = 2.572 * Hmax ** 1.5191
       rh98 is used as Hmax (maximum canopy height). Used only for the sensitivity check
       (Project.md 5.6); labels and models use MAIN.

H_CAP: the paper's global maximum AGB is 910.5 Mg/ha. MAIN reaches 910 Mg/ha at
(ln 910.5 - 3.9042) / 0.0858 = 33.9 m, so heights above 33.9 m are treated as implausible and
dropped (set to NaN), for both equations.
"""
import numpy as np

H_CAP = 33.9          # m; MAIN at 33.9 m = 909.5 Mg/ha, just under the paper's maximum (910.5)
CARBON_FRACTION = 0.451


def agb_main(h):
    """Southeast Asia generic: exp(3.9042 + 0.0858 h), Mg/ha. h above H_CAP or below 0 -> NaN."""
    h = np.asarray(h, dtype="float64")
    out = np.exp(3.9042 + 0.0858 * h)
    return np.where((h >= 0) & (h <= H_CAP), out, np.nan)


def agb_sensitivity(h):
    """Global Hmax power model: 2.572 * h**1.5191, Mg/ha. h above H_CAP or below 0 -> NaN."""
    h = np.asarray(h, dtype="float64")
    with np.errstate(invalid="ignore"):
        out = 2.572 * np.power(h, 1.5191)
    return np.where((h >= 0) & (h <= H_CAP), out, np.nan)


def carbon(agb):
    """Above-ground carbon, Mg C/ha (IPCC factor 0.451)."""
    return np.asarray(agb) * CARBON_FRACTION
