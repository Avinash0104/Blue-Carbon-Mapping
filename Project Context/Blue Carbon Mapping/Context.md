---
date: 2026-11-23
tags: [context, blue-carbon, data-preparation]
---

# Context

Current state of the data-preparation phase: what is done, how we got there, and what is running. The chronology is in [[Log]]. Ideas for better and more trustworthy predictions: [[Improvements]]. Spec: `Project.md`. Technical reference: `Data Preparation/Plan.md`.

## How the model works (read this first)

> [!important]
> **Height is NOT a model input.** The models see only the 21 satellite channels. GEDI height is used once, to make the biomass labels.
> **Biomass is NOT measured.** GEDI measures height (RH98); biomass is estimated from that height with an equation.

```
MAKING THE ANSWERS (labels)
GEDI laser return -> RH98 height (m) -> equation -> biomass (Mg/ha), one value per 25 m cell

TRAINING
21 satellite channels -> model -> predicted biomass  <- compared with the biomass label

MAPPING (later)
21 satellite channels -> trained model -> biomass everywhere
```

- **What GEDI measures:** a laser on the International Space Station; footprint about 25 m. RH98 = the height below which 98% of the returned energy lies, in practice the canopy height.
- **What the equation is for:** the models need examples with known biomass to learn from, and GEDI only gives height. `AGB = exp(3.9042 + 0.0858 * RH98)` (section 4).
- **Why height is not an input:** GEDI covers only scattered tracks (12,638 usable cells), while the satellites cover all mangroves, so a satellite-only model can fill the gaps. The label is calculated from height, so a model given height would just relearn the equation.
- **DEM nuance:** the DEM channel is a surface elevation (treetops included), so it carries a rough hint of canopy height. It is a separate satellite product, not GEDI's measurement.
- **What "error" means:** scores compare predictions with biomass **estimated from GEDI height**, not with field-measured biomass.

**Every processing step and its code:** [[Processing]].

**How a label meets the pixels** (details in section 5): a label is never given to individual 10 m pixels. A 25 m cell overlaps a 3 x 3 block of pixels, each weighted by the share of the cell lying on it (weights sum to 1).
- **RF and XGBoost:** each channel is the weighted average of the 9 pixels, giving one row of 21 inputs and one label per cell. The weights are fixed geometry; the model does not learn or see them.
- **U-Nets (planned):** predict every pixel, then compare the weighted average of the predictions with the cell's label.

## Where things stand (2026-11-23)

| Item | State |
|---|---|
| Spec and docs (`Project.md`, `Plan.md`) | Updated to all decisions below |
| Mangrove mask (v5) | Exported, downloaded, verified: 2,679.5 km2 |
| Inner mask (30 m) | Rebuilt from v5: 1,916 km2 (71.5% of mangrove) |
| GEDI labels (v2) | Exported, downloaded, verified: **12,800 training labels** (go) |
| Height-to-biomass code | Done, tested (see section 4) |
| Label tables (shots, cells) | Built and spot-checked: 12,800 shots, 12,638 cells (see section 5) |
| Satellite input code: static | Done and verified on a test export |
| Satellite input code: 2019-2023 composite | **Done and verified 2026-11-25**: 12 pieces in `data/raw/satellite/composite/`, derived channels written to `data/processed/satellite/` |
| Satellite static (DEM, PALSAR) | Exported, downloaded (4 pieces) and verified: DEM covers 100% of mangrove, HH/HV 98.92% (89 label cells have no PALSAR pixel) |
| Cell feature table (`code/processing/features/`) | **Final** (21 of 21 channels, 12,638 cells), `norm_stats.json` written (write-once), see section 7 |
| Local derived-channel script | Done and verified |
| Four small test exports | Done, all verified |
| RF and XGBoost (`Model Analysis/models/`) | **Run 2026-11-26**: test RMSE 94.5 / 94.2 Mg/ha, R2 0.49 (null 132.6; noise floor about 81); statistically tied. Spatial hold-out RMSE 98.6 both. U-Nets not started |

Earth Engine (project `carbon-mapping-mangroves`, started 2026-11-23, output to Drive folder `BlueCarbon`, one task each):
- Static (DEM, PALSAR HH and HV): task `V2VB3T7GMKT6NEP2T3PNXB4N` -> `sabah_static`: **COMPLETED**; Drive split it into 4 pieces, downloaded to `data/raw/satellite/` and verified.
- Composite (Sentinel-2 and Sentinel-1, 2019-2023 median): task `CH6EQ4J7IFQOD5IIMMCTCB53` -> `sabah_composite_2019_2023`: finished; downloaded and verified 2026-11-25 (see Log.md).

The full composite's runtime is uncertain (about half a day to several days; see the test result below).

Completed and downloaded into `data/raw/`: mask v5 (task `SULB73HEF2ZIXOODMCXCFMP2`) and GEDI v2 (task `5RSIV2OBBSZ5ZOFQ6E53XLTB`). They replaced the older v3/v4 masks, the old inner mask and the old GEDI pieces, which are no longer in `data/`.

The project is in Earth Engine "restricted mode" (noncommercial compute quota exceeded), so tasks wait a long time in READY before starting (the four test tasks took about an hour).

## Ground rules for the data

- Everything is GeoTIFF only (no Earth Engine assets), built in Earth Engine, downloaded from Drive.
- One common grid for every file so pixels line up: **EPSG:32650, 10 m, 43,995 x 36,200, top-left (312050, 816500)**, x 312,050-752,000, y 454,500-816,500. GEDI is the same extent at 25 m (17,598 x 14,480). Pixel edges coincide every 50 m.
- The grid is the bounding box of the Sabah outline, with edges on 50 m multiples. A raster must be a rectangle and mangroves fringe the whole coast, so cropping to the mangrove box would save little; keeping one grid avoids offsets between layers.
- Nodata is not tagged inside the files; mask it when loading: -1 for GEDI, -32768 for static and yearly files. The mask files have none.

## Products and how they were decided

### 1. Mangrove mask
- **What:** binary uint8, 1 = stable mangrove, 0 = everything else, no nodata. Stable mangrove = Global Mangrove Watch v4.0.19 (2020) AND GMW v3 mangrove in all six years 2015-2020. Built by `code/collecting/mask/mangrove_mask.py`.
- **1/0 instead of 1/0/255:** simpler; a separate "ignore" class was parked for later.
- **Sabah outline: FAO GAUL 2025 level 1, not geoBoundaries.** A plot at four coastal sites showed geoBoundaries' rough outline cutting off real mangrove at the coast. GAUL 2025 keeps about 290 km2 more (stable mangrove about 2,675 km2 inside GAUL vs 2,382 km2 inside geoBoundaries on the same 30 m test). The 2015 GAUL is deprecated, so 2025 is used. This also enlarged the grid to the GAUL 2025 bounding box.

![[outline_compare.png]]

- **Inner mask:** the full mask shrunk by 30 m (`code/processing/mask/inner_mask.py`). It selects GEDI training labels; the full mask is used for mapping. Rebuilt from v5 (27 s): it keeps 1,916 of 2,679.5 km2 (71.5%).
- **Files:** `data/raw/mask/sabah_mangrove_mask.tif` (full mask) and `data/raw/mask/sabah_mangrove_inner_mask.tif` (inner mask; the shrink distance, 30 m, is stored inside the file as the tag `INNER_MASK_METRES`).
- **Mask v5 verified:** EPSG:32650, 10 m, 43,995 x 36,200, top-left (312050, 816500), values only 0 and 1, 26,795,143 mangrove pixels = 2,679.5 km2 (267,951 ha, under the 298,000 ha sanity limit; consistent with the ~2,675 km2 GAUL estimate).

### 2. GEDI labels
- **What:** 48 monthly float32 bands (`rh98_2019_04` ... `rh98_2023_03`), GEDI L2A rh98 in metres, 25 m pixels, nodata -1, from `LARSE/GEDI/GEDI02_A_002_MONTHLY`. Built by `code/collecting/gedi/export_gedi.py`.
- **Filters (stricter than before):** `quality_flag` = 1, `degrade_flag` = 0, sensitivity 0.98-1, **power beams only (5, 6, 8, 11)**, **night shots** (`solar_elevation < 0`), rh98 0-50 m, pixel on mask value 1. Power beams and night follow NASA's guidance for the best height accuracy. Beam IDs 5, 6, 8 and 11 are the power beams by convention and are consistent with the data; they are not listed in the Earth Engine catalogue.
- **Raster, not footprint points.** Footprints are about 25 m and geolocation error is of similar size, so snapping to the 25 m cell loses little; the raster fits the GeoTIFF-only rule and is already verified. Footprint points would only matter for the optional shift-resilient loss.
- **No true 10 m labels exist:** a footprint is about 25 m wide. The label stays with its 25 m cell and is linked to the 10 m pixels by area weights (section 5). The centre pixel is used only to select cells on the mask and for reference.
- **Full-state label counts (v2 export, 12 pieces + VRT, 48 bands, grid correct):** from `code/processing/gedi/count_labels.py`; each 25 m cell is placed on the 10 m pixel at its centre.

| Step | Shots | Cells |
|---|---|---|
| Valid in the export (Earth Engine filters, on mask = 1) | 18,336 | 18,093 |
| Centre pixel on the full mask | 17,843 | 17,609 |
| ... and on the inner mask (30 m): training labels | 12,907 | 12,741 |
| ... and rh98 <= 33.9 m | **12,800** | **12,638** |

  12,800 is "several thousand or more", so the go/no-go is **go**. 40 of 48 months have labels (median 130 per month, maximum 1,656); 2.7% of label centres fall on a mask pixel that is not 1. Training labels: rh98 2.9-33.9 m, median 15.4 m. Biomass with the main equation: median 186, mean 224 Mg/ha, mean carbon 101 Mg C/ha, close to the Sabah field mean of 95 ± 52 Mg C/ha (Lee et al.); with the sensitivity equation: median 164, mean 181 Mg/ha, mean carbon 82 Mg C/ha.
- **Earlier pilot counts** (1 degree window around Sandakan, 117.4-118.4 E, 5.4-6.4 N), pixels after each filter:

| Step | Pixels |
|---|---|
| Raw rh98 | 1,561,431 |
| Quality flag | 628,966 |
| Degrade flag | 466,662 |
| Sensitivity >= 0.98 | 97,562 |
| Power beams | 75,499 |
| Night | 37,955 |
| On mangrove | **4,702** |

Sensitivity is the biggest cut. A separate check on a 3 km window with labels gave 203 values in 202 cells from the exported-style stack, matching an independent count (202), with values 10.0-29.5 m.

### 3. Satellite inputs (mangrove + 200 m margin only)
- **Zone:** mangrove plus a **200 m** margin, built on a 100 m grid and used as a mask on the 10 m data. The margin was first 500 m (5,982 km2 against 2,389 km2 of mangrove, too much and it defeats the inner mask); measured options were 200 m = 4,493 km2 and 500 m = 5,982 km2, and 200 m was chosen. Outside the zone the files are nodata.
- **21 channels:** Sentinel-2 B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (10) + NDVI, NDRE, NDMI, EVI (4); Sentinel-1 VV, VH + VV/VH (3); PALSAR-2 HH, HV + HV/HH (3); Copernicus DEM elevation (1).
- **Slope dropped.** Mangroves are flat and the DSM noise dominates; it can come back as an ablation. The script can still write it.
- **Exports (int16, nodata -32768):**
  - Static `sabah_static` (about 0.55 GB): `DEM_x10`, `HH_x100`, `HV_x100`. PALSAR is the median of the 2019-2023 yearly mosaics, which covers 99.4% of mangrove pixels (single years leave gaps, e.g. 2021 misses 44%).
  - Composite `sabah_composite_2019_2023` (about 2.2 GB): the 10 S2 bands (÷10,000) and `VV_x100`, `VH_x100`, as one median over all clear scenes in 2019-2023.
  - **Why one composite, not five yearly files** (decided 2026-11-23; the spec's "2019-2023 median composites" meant this, but the code had been writing yearly files): the labels are effectively static (the mask keeps only mangrove stable 2015-2020, and structure changes little in five years against an equation RMSE of 148 Mg/ha); a five-year median is cleaner and has fewer cloud gaps in cloudy Sabah; and it is about 5x cheaper (one 2.2 GB export instead of about 11 GB). Yearly files would repeat each labelled pixel up to five times with the same height, closer to augmentation than new information. Cost: about 5x more scenes per pixel (727 vs 147 in a 9 km window), so Earth Engine memory must be checked with a small test export (`export_satellite.py testc`). Fallback: export yearly files and take the median across years locally. Single-year exports still work (`full 2020`).
  - int16 was chosen over float32 because the rounding (0.1 m, 0.01 dB, 0.0001 reflectance) is far finer than the accuracy of the sources, and it halves the size.
- **Sentinel-2:** Cloud Score+ `cs_cdf` >= 0.6 cloud mask, calendar-year median. The six 20 m bands are medianed at 20 m, then resampled to 10 m with **bicubic** and clamped at 0. Resampling every scene first ran out of Earth Engine memory.
- **PALSAR:** bicubic from 25 m, applied to DN and clamped at DN >= 1 before the dB conversion (dB = 10 log10(DN^2) - 83).
- **DEM:** bilinear (smooth surface, sharper interpolation adds nothing). Bicubic was chosen for the optical and radar bands because the models need detail at 10 m; the clamp handles bicubic's overshoot at sharp edges.
- **Sentinel-1:** IW mode, **descending only** (the only direction acquired over Sabah; Dong et al. used ascending, which isn't available here). **No focal-mean speckle filters** on S1 or PALSAR: the yearly or multi-year median already averages out speckle and a filter would blur narrow fringes.
- **Derived channels are computed locally** (`code/processing/satellite/derive_channels.py`): NDVI, NDRE, NDMI, EVI, VV-VH (dB) from the composite (or any single-year file); HV-HH (dB) from the static file. They use no Earth Engine quota.

### 4. Height-to-biomass equation (decided 2026-11-23)
Source: Simard et al. (2019), Nature Geoscience 12, 40-45, Supplementary Table 8 (read from the paper's Methods and the pasted table). Code: `code/processing/gedi/biomass.py`; checks: `check_biomass.py` (pass).

| Role | Equation | Fit | Height input |
|---|---|---|---|
| **Main** (labels, all models) | AGB = exp(3.9042 + 0.0858 · H) Mg/ha, Southeast Asia generic (Aslan et al. 2016) | R² 0.55, RMSE 148 Mg/ha | Canopy-height-model height, so GEDI rh98 directly |
| **Sensitivity** (carbon-total check only) | AGB = 2.572 · Hmax^1.5191, global Hmax model | R² 0.70, RMSE 180 Mg/ha | rh98 taken as Hmax |

Why this choice and how we got there:
- The paper fitted new regional models only for the Americas, East Africa and South Asia (Bangladesh). For Southeast Asia it used the published Aslan et al. (2016) model, which is the exponential above. It is the only equation fitted for the region, and regions differ a lot (East Africa and the Americas differ by 2-3x at the same height).
- **Correction made during the work:** the first search results led me to dismiss this exponential as a "2018" equation with a different height. The pasted table showed it is the Southeast Asia model itself. My first quoted fit statistics (R² 0.77, RMSE 57.2) also did not match the paper; the paper gives regional RMSE 54.3-103.4 Mg/ha and plot-level map accuracy 84.2 Mg/ha.
- The Hba equations (global 3.254 · Hba^1.5295, etc.) expect basal-area-weighted height, which is about 0.63 x Hmax (paper: Hba = 1.0754 x SRTM, Hmax = 1.697 x SRTM). GEDI rh98 is a top-of-canopy height, so feeding it to an Hba equation would overstate biomass, and the ratio conversion rests on an unverified assumption. So they were not used.
- The sensitivity equation is a different functional form (power law), so the carbon totals under it show how much the baseline depends on the equation. It uses rh98 as Hmax, again with no conversion. It is applied to the same heights, with no retraining.
- **Height cap 33.9 m:** the exponential is unbounded (49.6 Mg/ha at 0 m, about 3,600 Mg/ha at 50 m). It reaches 909.5 Mg/ha at 33.9 m, just under the paper's global maximum (910.5), so labels above 33.9 m (or below 0) are dropped for both equations. The GEDI export keeps 0-50 m; the cap is applied locally.
- Values (Mg/ha) at rh98 = 5 / 10 / 15 / 20 / 30 m: main 76 / 117 / 180 / 276 / 651; sensitivity 30 / 85 / 157 / 244 / 451. At the labels' typical 10-30 m heights (median about 15 m) both give about 160-180 Mg/ha, against about 211 Mg/ha implied by the Sabah field mean (95 Mg C/ha / 0.451).
- RMSE 148 Mg/ha is large against a mean near 200, so label noise from the equation is substantial. Averaging over pixels reduces noise but not bias, so equation uncertainty is reported separately from the ensemble spread.

### 5. How a label is attached to the images (decided 2026-11-23)
Tables: `data/processed/labels/gedi_shots.csv` (12,800 shots) and `gedi_cells.csv` (12,638 cells = the training labels). Code: `code/processing/gedi/cells.py`, `make_labels.py`; checks `check_cells.py` pass.

- **A label belongs to the 25 m cell, not to a 10 m pixel.** A cell covers 2.5 x 2.5 pixels, so it overlaps a 3 x 3 window of pixels, each weighted by its share of the cell's area (weights sum to 1, e.g. 0.4/0.4/0.2 along an axis).
- **Options considered:**
  - A, centre pixel only: simple, but the shot was already snapped to a 25 m cell, so the exact footprint position is unknown and the pixel can be the wrong one.
  - D, copy the label onto every pixel: rejected, because one measurement would count about 6 times (inflated training set and scores) and it falsely claims each pixel has the footprint's value.
  - **B (chosen for RF and XGBoost):** average the 21 channels over the cell's window and train on one row per cell. Matches the scale GEDI measured and averages out speckle.
  - **C (chosen for the U-Nets):** predict every 10 m pixel and compare the weighted mean of the predictions over the window with the label (footprint loss). The model decides how the biomass is spread inside the cell, so 10 m detail comes from the images and is not copied from the label.
- **All four models are evaluated at the cell level** (cell-mean prediction vs cell label), so the RQ1 comparison uses the same unit. The centre pixel is stored for reference only.
- An experiment comparing A, B and C was planned but cannot run yet: the test windows hold 0 labels (9 km preview) and 8 cell labels (44 km window), and a fair comparison needs the spatial split. The decision was taken on the reasoning above instead.
- **Repeated cells:** 162 cells have two shots (in different months). Biomass is converted shot by shot and then averaged; e.g. for one repeated cell the mean of the biomasses is 481.9 Mg/ha while the biomass of the mean height would be 480.5 (the equation is not linear).
- The inner mask puts every selected cell at least 30 m from non-mangrove, so every cell's 3 x 3 window lies fully on mangrove (`mangrove_frac` = 1.0 for all cells).

### 6. Spatial split: design decision and label coverage
Decision (2026-11-23): **one random 70/15/15 split of the label cells for all four models** (see 6b for how it was reached; the earlier plan of a held-out region plus spatial blocks was dropped). The split file is fixed once written.

Where the 12,638 label cells are (script `code/processing/gedi/label_coverage.py`; table `data/processed/labels/label_coverage.csv`; district polygons `data/raw/boundaries/sabah_districts_gaul2025.geojson`, GAUL 2025 level 2, simplified to 100 m):

![[label_coverage.png]]

| Group | Districts | Label cells | Share |
|---|---|---|---|
| East coast | Kinabatangan 4,391; Tawau 3,150; Beluran 2,091; Sandakan 827; Lahad Datu 601; Semporna 444; Kunak 49 | 11,553 | 91% |
| North | Kudat 328; Pitas 232; Kota Marudu 115 | 675 | 5% |
| West | Kuala Penyu 248; Beaufort 70; Tuaran 45; Kota Belud 40; Sipitang 5; Papar 2 | 410 | 3% |

- The labels sit in GEDI tracks, visible as stripes on the map, so neighbouring cells are strongly alike (see 6b for what this means for a random split).
- Mean biomass (main equation) differs by district: Kota Marudu 375 Mg/ha (115 cells, the high outlier), Lahad Datu 277, Kudat 253, Tawau 223, Kinabatangan 218 (median 155), Beluran 214, Sandakan 204, Beaufort 194.
- Label density per km2 of mangrove varies widely (Tawau 8.7, Lahad Datu 8.1, Kinabatangan 7.6, Beluran 3.2, Sandakan 2.1).
- 9 of the 25 districts have no labels (inland, no mangrove). Two cells fell outside every polygon and were assigned to the nearest district; a few mangrove areas lie outside the generalised polygons (coarse mangrove area 2,699 km2 against 2,679.5 from the mask).

### 6b. The split: how it was decided, and the result
**Result:** `data/processed/labels/gedi_split.csv` (code `code/processing/split/make_split.py`, seed 42, sha256 prefix f3bbd16fea37d927): train 8,846 cells (70.0%), validation 1,896 (15.0%), test 1,896 (15.0%). Mean biomass is the same in all three (223 / 226 / 223 Mg/ha; p10-p90 about 100-397). The file is never changed once written; the script refuses to overwrite it.

**How it was decided**
- First plan: hold out one region and split the rest into spatial blocks. This was dropped before anything was saved.
- Variogram of the cell labels (`code/processing/split/variogram.py`, figure below): the semivariance levels off at **3.4 km (biomass) and 4.5 km (height)**; about 60% of the variance is nugget (GEDI noise plus equation error), so neighbouring cells already differ a lot. Blocks of 10 km and a 5 km buffer were proposed.
- Literature check of the split in the closest papers: Dong et al. (2023, attention U-Net on GEDI) split image patches **at random, 7:2:1**, plus 5-fold cross-validation (the 7:2:1 ratio and counts come from a summary of the unread part of the paper); Lang et al. (2023) hold out **20% of 100 km Sentinel-2 tiles** and test on independent airborne LiDAR; H. Zhou et al. (2025) use **leave-one-out** cross-validation on 46 field plots (random, because of the small sample). Abu Bakar et al. (2024) and the full text of Ploton et al. (2020) could not be read (publisher blocks).
- **User decision: a random split for now, as in the earlier papers** (data are limited; a spatial check on another held-out area, possibly in another state, can be added later).

![[variogram.png]]

**Known limitation (measured):** with a random split, 71% of validation and test cells have a training cell within 100 m, 99% within 1 km (median distance 56 m), against a variogram range of 3-5 km. For comparison, 10 km blocks would have left 1% within 100 m and 24% within 1 km. So the scores will be optimistic compared with predicting an unlabelled area; Ploton et al. (2020) report R² falling from over 0.5 to almost 0 once validation is spatial (range 0.14-0.53 depending on the method). To be stated in the dissertation; also `Project.md` section 3.

**How the U-Nets use it:** a 128 x 128 tile can contain cells of all three sets; the loss and metrics use only the cells of the set in question (the imagery is shared, the labels are not).

### 7. Cell feature table and normalisation statistics (built 2026-11-24)
Code: `code/processing/features/` (`channels.py` = the 21 channels, their order and sources; `build_cell_features.py`; `check_features.py`, all pass). Output: `data/processed/features/`.
- One row per label cell (`gedi_cells.csv` order), joined to the split. Each channel = weighted mean over the cell's 3 x 3 window (option B), nodata to NaN first, then the scale factor. Values stored in physical units; no x_utm/y_utm in the table (`cell_row`/`cell_col` are keys only: models select inputs with `CHANNELS`).
- Statistics (mean, std with ddof 0, per channel) from the TRAIN cells only, shared by all four models.
- **Partial until the composite exists:** with 3 of 21 channels it writes `cell_features_partial.csv` and `norm_stats_partial.json` (overwritable). Only with all 21 does it write `cell_features.csv` and the write-once `norm_stats.json`. A missing or incomplete satellite piece stops the run (it cannot pass as "no data").
- Static result (train): DEM mean 9.98 m, HH -8.93 dB, HV -15.17 dB. **89 of 12,638 cells (0.7%; 61 train, 15 val, 13 test) have no PALSAR pixel**, so HH and HV are NaN there. **Decided 2026-11-24:** missing values are filled with the training mean (0 after normalisation), identically for all four models, via `code/processing/features/normalise.py`; nothing is dropped and the split is untouched; no separate accuracy check for these cells.
- Checked against an independent area calculation (synthetic pieces, incl. a window across four pieces) and against direct reads of 40 real cells (largest difference 4e-15).
- Compared with Dong et al. (2023; full text read): same sensors, resampling, PALSAR dB equation, training-only standardisation and area-weighted RF features. Differences, all deliberate: their lat/lon inputs (banned here), 2019 only with min/max NDVI, focal-mean filters, ascending S1, every footprint pixel labelled (our option D, rejected), canopy-cover-dependent sensitivity plus a geolocation filter (the only idea worth considering: it would change the label set). Their paper has no XGBoost.

### 8. Evaluation module (built 2026-11-24; `Model Analysis/evaluation/`)
Folder layout: `Model Analysis/` holds `evaluation/` (code), `models/` (empty, for RF, XGBoost, U-Nets) and `data/predictions/` + `data/results/`. It only reads labels and the split from `Data Preparation/`. Run from the `Model Analysis` folder: `python evaluation/evaluate.py --models null rf ... --pairs rf:null`; checks: `python evaluation/check_evaluation.py`.
- **Contract:** every model writes `data/predictions/<model>.csv` (cell_row, cell_col, set, pred, optional pred_std = spread of the members' cell-mean predictions). Cell-level prediction is the model's job (U-Nets: weighted 3 x 3 window mean with `cells.py` weights). Labels and coordinates are joined by key from `gedi_cells.csv`, never written by a model; all test cells must be present; `set` must match the split; the sidecar records the split hash and read refuses a different split.
- **Definitions (decided 2026-11-24):** R2 = 1 - SSres/SStot (squared Pearson also reported, named), bias = pred - obs, relative RMSE = 100 RMSE / mean(obs), Mg/ha only. Bins by observed value: 0-100, 100-200, 200-300, 300-400, 400+ (fixed in advance; regression to the mean makes low bins positive and high bins negative for any model, so compare with the null row). Block bootstrap: 4 km blocks (about the variogram range; 284 test blocks), 2,000 resamples, seed 42, 95% percentile intervals, same blocks for every model; paired differences on the same blocks; no interval if fewer than 20 blocks. The bootstrap holds predictions fixed: it captures between-block sampling variation only, not training or seed variance, and not the random-split optimism. Only test cells are scored; train/validation scores come from the out-of-fold files `<model>_oof.csv`. Uncertainty check: 5 equal-count bins of pred_std, Spearman rho as the main evidence.
- **Null model** (train mean, 223.27 Mg/ha) on the real test cells: R2 about 0, bias +0.4, **RMSE 132.6 Mg/ha (95% CI 121.3-143.9)**, relative RMSE 59.5%, MAE 98.9. This is the baseline every model must beat. 72% of test cells have a train cell within 100 m, 99% within 1 km (printed with every run).
- **Aggregated 0.25-1 ha accuracy (decided 2026-11-24: keep both versions).** (a) Test-cell version, built in `evaluation/aggregate.py` and `results_aggregated.csv`: test cells averaged in 50 m / 100 m squares with 2+ test cells; small sample (20 squares at 50 m, 93 at 100 m on the real split), reported as an indication only, no confidence interval when under 20 blocks. Null model: 100 m RMSE 117.9, R2 -0.004. (b) Against independent data (field values, GEDI L4B, other maps): to do once the map exists.
- Code review: no leakage route found; fixed val scoring depending on model order, unclear errors for bad pairs, zero-width intervals on one block, and added a read-time split-hash check, NaN-coordinate check and constant-spread guard (tests added).

- **Robustness evaluations added 2026-11-25** (chosen as the ones that decide whether the results matter and which model wins):
  - *Label-noise floor* (`noise_floor.py`): 162 cells hit by two shots give single-shot noise sigma = 81.6 Mg/ha (95% CI about 61-98, 35% of the mean). Best achievable test RMSE about **81 Mg/ha** and R2 ceiling about **0.62** (null model 132.6). Approximate lower bound: noise shared by both shots is not seen.
  - *Calibration and total bias*: `bias_pct` (bias as % of the mean = bias of a carbon total) and `calib_slope` (y on p; 1 = calibrated) added to every metric table with bootstrap intervals.
  - *Spatial hold-out* (`spatial.py`): 5 folds of 10 km blocks, 5 km buffer, each cell tested once; folds file `data/spatial/spatial_folds.csv` fixed once (2,527-2,528 test cells per fold; train 5,746-7,491). Extra evaluation, `gedi_split.csv` untouched. Null model: RMSE 132.2; by fold 114-152.
  - *Seed variance* (`seed_variance.py`): per-seed scores, mean/sd, and whether a model difference exceeds the seed spread. U-Nets use ensemble members as seeds.
  - Still to do: independent-data check (needs the map), residual clustering, ablations.

## Verification of the test exports

Four small exports (a 9 km preview window for mask, static and 2020; GEDI on a 13 km window) were run through the real batch path and checked locally:

| File | Result |
|---|---|
| Mask | EPSG:32650, 10 m, uint8, values only 0/1, corners on the grid |
| Static | int16, 3 bands in the right order; DEM -4 to 29 m (median 5.8 m); HV median about -15 dB |
| Yearly 2020 | int16, 12 bands in the right order; S2 median NIR about 0.30; VV and VH medians about -8.1 and -14.6 dB; same coverage as static |
| GEDI | 48 bands, 25 m, float32, right grid; the 13 km window had no labels (only about 61 shots passed the filters and none fell on mangrove), so labels were checked separately (see section 2) |
| Derived channels (real files) | NDVI median 0.74, VV-VH 6.5 dB, HV-HH -6.2 dB, slope median 2.8 degrees |

**2019-2023 composite test** (task `2J3VPPQG3JSEQPS35HS7U75H`, 44 km Sandakan window, 4,436 x 4,431 px, 149 MB file; local copies deleted after the checks): COMPLETED with no error, so the five-year median fits in Earth Engine memory in a batch export. It waited 45 min in the queue and ran for 78 min. Checks: EPSG:32650, 10 m, int16, 12 bands in the right order; every band valid on the same 21.2% of pixels (417 km2 zone in a 1,966 km2 window); **no mangrove pixel (1,863,810 in the window) lacks data**; at mangrove pixels the median NIR (B8) is 0.276, VV -8.4 dB, VH -14.6 dB; derived NDVI median 0.79, NDRE 0.57, NDMI 0.40, EVI 0.50, VV-VH 6.3 dB. Against the single-year 2020 test on the 9 km overlap (276,873 shared pixels, same valid pixels), the bands agree closely: correlation 0.97-0.99, mean absolute difference 0.006-0.012 in reflectance and about 0.45 dB in backscatter, with no systematic shift (mean differences within 0.003 in reflectance and 0.12 dB). Runtime scaling to the full state is uncertain: the full zone is about 10.8 times the test zone (4,493 vs 417 km2) and the full grid about 81 times the test window, so the full composite could take anywhere from about half a day to several days.

The full 12-band yearly image fails as an *interactive* download (Earth Engine memory limit, 503), but the same image completed through the batch export, so batch is the route for the full data.

## Code layout

```
code/
  collecting/    Earth Engine side: builds and exports the raw data
    mask/        mangrove_mask.py  export_mask.py  boundary.py  check.html
    gedi/        export_gedi.py  labels.py
    satellite/   composites.py  export_satellite.py  inspect_sat.py
  processing/    local side: builds the processed data from the raw files
    mask/        inner_mask.py
    gedi/        build_gedi_vrt.py  biomass.py  cells.py  count_labels.py  make_labels.py  label_coverage.py
                 check_biomass.py  check_cells.py   (checks for biomass.py and cells.py)
    split/       region.py  variogram.py  make_split.py
    satellite/   derive_channels.py
data/
  raw/         as downloaded from Earth Engine / Drive, never edited
    mask/  gedi/  satellite/  boundaries/
  processed/   built locally from raw by the scripts
    labels/  satellite/
```

Docs: `Plan.md` describes each product and the workflow. `finalize_tifs.py` was removed because masking nodata at load time is simpler than rewriting the files.
