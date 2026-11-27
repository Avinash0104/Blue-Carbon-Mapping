---
date: 2026-11-27
tags: [blue-carbon, data-preparation, processing]
---

# Data processing: from raw data to model inputs

Each step in simple terms, with the code responsible. Overview in [[Context]]; chronology in [[Log]].

Paths are inside `Data Preparation/code/` unless they start with `Model Analysis/`. Line numbers are as of 2026-11-27.

> [!important]
> The models see only the **21 satellite channels**. GEDI height is **not** an input: it is turned into the biomass **label** the models learn to predict.

```
Satellite -> 2019-2023 layers -> derived channels -+
                                                    +-> 3x3 weighted average -> normalise -> RF / XGBoost
GEDI -> filter -> biomass -> 25 m cell labels ------+
```

## 0. Mangrove mask (where data is used)
| Step | Simple explanation | Code |
|---|---|---|
| Full mask | Mangrove in GMW 2020 **and** in all six years 2015-2020 | `collecting/mask/mangrove_mask.py:3` |
| Inner mask | Shrunk by 30 m; used to pick label cells | `processing/mask/inner_mask.py:45` |
| Satellite zone | Mangrove plus 200 m; satellite data is kept only here | `collecting/satellite/composites.py:100` |

## 1. Sentinel-2 (optical, 10 channels)
| Step | Simple explanation | Code |
|---|---|---|
| Cloud mask | Remove pixels with a clear-sky score below 0.6 | `collecting/satellite/composites.py:34` |
| Median | Middle value of all clear scenes 2019-2023 | `collecting/satellite/composites.py:50` |
| Resize | The six 20 m bands go to 10 m (bicubic); negatives set to 0 | `collecting/satellite/composites.py:51-53` |

## 2. Sentinel-1 (radar, VV and VH)
| Step | Simple explanation | Code |
|---|---|---|
| Select and median | IW-mode scenes with VV and VH; median of the decibel values | `collecting/satellite/composites.py:59-73` |

The code does not filter by orbit direction; it relies on only descending passes existing over Sabah (code comment).

## 3. PALSAR-2 (radar, HH and HV)
| Step | Simple explanation | Code |
|---|---|---|
| Resize | 25 m to 10 m, bicubic: each new pixel is a smooth weighted mix of the 16 nearest originals. No real detail is added | `collecting/satellite/composites.py:83` |
| Decibels | Raw number (DN) raised to at least 1, then `10 log10(DN^2) - 83` | `collecting/satellite/composites.py:85` |
| Land only | Keep pixels with quality value 255 | `collecting/satellite/composites.py:87` |
| Median | Median of the five yearly layers | `collecting/satellite/composites.py:97` |

## 4. DEM (elevation)
| Step | Simple explanation | Code |
|---|---|---|
| Resize | 30 m to 10 m (bilinear), tiles joined. It is a surface height, so treetops are included | `collecting/satellite/composites.py:114` |

## Export of steps 1-4
`collecting/satellite/export_satellite.py:61-78`: cut to the zone, store as whole numbers (empty = -32768), download as GeoTIFF.
Scales: Sentinel-2 reflectance x 10,000; radar decibels x 100; DEM metres x 10.

## 5. Derived channels (6 used)
| Channels | Simple explanation | Code |
|---|---|---|
| NDVI, NDRE, NDMI, EVI | Vegetation indices from the Sentinel-2 medians | `processing/satellite/derive_channels.py:45-57` |
| VV - VH | Radar ratio (a ratio is a difference in decibels) | `processing/satellite/derive_channels.py:55` |
| HV - HH | PALSAR ratio | `processing/satellite/derive_channels.py:63` |

Slope is also calculated but not used. Total channels: 10 + 2 + 2 + 1 + 6 = **21**.

## 6. GEDI (labels only)
| Step | Simple explanation | Code |
|---|---|---|
| Shot filters | Good quality, not degraded, sensitivity 0.98-1, power beams, night | `collecting/gedi/export_gedi.py:26` |
| Monthly export | Mangrove only, height 0-50 m, one band per month | `collecting/gedi/export_gedi.py:38-71` |
| Join pieces | Downloaded files read as one | `processing/gedi/build_gedi_vrt.py` |
| Select | Cell centre on the inner mask, height up to 33.9 m | `processing/gedi/make_labels.py:53` |
| Biomass | `AGB = exp(3.9042 + 0.0858 * height)` (Mg/ha), shot by shot | `processing/gedi/biomass.py:27` |
| Cell label | Average of the shots in the same 25 m cell (12,638 cells) | `processing/gedi/make_labels.py:63` |

GEDI measures **height** (RH98, about the canopy top, in a footprint about 25 m wide). Biomass is an **estimate** from that height, so model errors are measured against estimated biomass, not field measurements.

## 7. Build the model table
| Step | Simple explanation | Code |
|---|---|---|
| Channel list | Order of the 21 channels and their scales | `processing/features/channels.py` |
| 3x3 weights | A 25 m cell covers parts of 9 pixels; weight = share of the cell on each pixel (sum 1) | `processing/gedi/cells.py:23-39` |
| Read pixels | Empty pixels marked missing, scaling undone | `processing/features/build_cell_features.py:94` |
| Weighted average | 9 pixels become 1 value per channel; missing pixels skipped | `processing/features/build_cell_features.py:38` |
| Split | Random 70/15/15, fixed | `processing/split/make_split.py` |
| Training statistics | Mean and standard deviation from the 8,846 training cells only | `processing/features/build_cell_features.py:124` |

Output: `data/processed/features/cell_features.csv` (one row per cell) and `norm_stats.json`.

**Weighted average, example (made-up NDVI values):**

| Values | | | Weights | | |
|---|---|---|---|---|---|
| 0.80 | 0.78 | 0.60 | 0.16 | 0.16 | 0.08 |
| 0.82 | 0.76 | 0.62 | 0.16 | 0.16 | 0.08 |
| 0.70 | 0.68 | 0.50 | 0.08 | 0.08 | 0.04 |

Multiply pixel by weight and add all nine: **0.73**. The weights come from geometry; the model neither learns nor sees them. A label is never copied onto individual 10 m pixels.

## 8. Normalise (just before the model)
| Step | Simple explanation | Code |
|---|---|---|
| Scale | `(value - mean) / standard deviation`; missing becomes 0 (the training mean) | `processing/features/normalise.py:31` |
| Load for models | Table read and normalised for RF and XGBoost | `Model Analysis/models/train.py:83` |

The planned U-Nets will use the same channels and statistics, but on image tiles: they predict every pixel and compare the weighted average of the predictions with the cell label (not built yet).
