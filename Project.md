# Blue Carbon Mapping of Mangroves in Sabah

Final year project (FYP). Deep learning + GEDI spaceborne LiDAR to map mangrove above-ground biomass (AGB) and carbon across Sabah, Malaysia.

---

# Plan

## 1. Project summary

- **Aim:** quantify and map the blue carbon stored in Sabah's mangroves, establishing a reliable, state-wide carbon baseline that supports conservation and participation in blue carbon markets.
- **Objectives:**
  1. Develop an accurate, validated model for estimating mangrove AGB from freely available satellite imagery.
  2. Establish a reliable, state-wide carbon baseline for Sabah's mangroves.
  3. Provide an open, accessible tool (web map + reproducible workflow) so others can explore, check and update the results.
- **Research questions:**
  - **RQ1:** How accurately do Random Forest, XGBoost, an attention U-Net and a nested U-Net (UNet++) trained on GEDI-derived labels estimate mangrove biomass, and which performs best?
  - **RQ2:** How much above-ground carbon do Sabah's mangroves store, and what is its distribution and uncertainty across districts and protected areas?
- **Core idea:** sparse GEDI footprints (~25 m) provide canopy height, which is converted into biomass labels. Models learn to predict biomass from wall-to-wall 10 m satellite imagery, producing a state-wide biomass map that is then converted into carbon totals.
- **Novelty:** first application of GEDI-label deep learning to Malaysian mangroves; mangrove-specific model (general global models overestimate mangrove height — Lang et al., 2023); results reported with a measured spatial-leakage caveat (Ploton et al., 2020).

## 2. Fixed data decisions (do not change without updating this file)

### 2.1 Study area, grid and period
- **Study area:** Sabah's **mangrove extent only** (plus a **200 m** buffer for model context), not the whole state.
- **Grid:** **EPSG:32650 (UTM zone 50N), 10 m**. Every layer must be on this exact grid and cover the same export region.
- **Period:** **2019–2023**. GEDI: **2019-04-01 to 2023-03-17** (GEDI was in storage 17 Mar 2023 – 27 Apr 2024).
- **Export region / format:** **no GEE assets; GeoTIFF files only** (via Drive). Every export uses the mask's grid (EPSG:32650, 10 m, 43,995 × 36,200, top-left 312050, 816500; x 312,050–752,000, y 454,500–816,500, the GAUL 2025 Sabah bounding box with edges on 50 m multiples); pixels outside "mangroves + 200 m buffer" are no-data. Use this same grid for every export.

### 2.2 Mangrove mask
- **Outline:** GMW v4.0.19, 2020, ~10 m — `projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020` (community catalogue).
- **Stability check:** GMW v3.0 yearly extents 2015–2020 — `projects/sat-io/open-datasets/GMW/extent/gmw_v3_{year}_vec`. Keep only pixels mangrove in **all six years**.
- **Full mask** = v4 (2020) ∩ v3 stable 2015–2020, clipped to Sabah; used for **mapping**.
- **Inner mask** = full mask shrunk by **30 m**; used to **select GEDI training labels**.
- **Encoding:** both masks are binary, **1 = mangrove, 0 = everything else**, no nodata value.
- **Sabah boundary:** `FAO/GAUL/2025/level1` (GAUL0_NAME = "Malaysia", GAUL1_NAME = "Sabah"; the 2015 version is deprecated). Districts: `FAO/GAUL/2025/level2` (check the property names in the catalogue). Chosen over geoBoundaries ADM1 after a check: geoBoundaries' rough outline cut real mangrove off the coast (GAUL 2025 keeps about 290 km² more). Protected areas: `WCMC/WDPA/current/polygons` (check it includes Sabah forest reserves).
- Sanity check: full-mask area should be somewhat below ~298,000 ha.

### 2.3 GEDI labels
- **Dataset:** `LARSE/GEDI/GEDI02_A_002_MONTHLY` (25 m monthly raster). Use footprint-level `LARSE/GEDI/GEDI02_A_002` only if the shift-resilient loss is implemented (needs orbit/beam per shot).
- **Filters:** `quality_flag == 1`, `degrade_flag == 0`, `0.98 <= sensitivity <= 1`, **power beams** only (`beam` 5, 6, 8, 11), **night-time** shots (`solar_elevation < 0`), `0 <= rh98 <= 50` m (min 0 keeps the −1 no-data value unambiguous), inside **inner mask**.
- **Height metric:** RH98.
- **Height-to-biomass conversion** (Simard et al. 2019, Supplementary Table 8; code `code/processing/gedi/biomass.py`):
  - **Main (labels and all models):** Southeast Asia generic equation (originally Aslan et al., 2016), **AGB = exp(3.9042 + 0.0858 · H)** Mg/ha, R² 0.55, RMSE 148 Mg/ha. H is a canopy-height-model (top-of-canopy) height, so GEDI RH98 is used directly with no conversion. It is not a power law (49.6 Mg/ha at H = 0, unbounded growth), so it is only valid up to **H = 33.9 m** (909.5 Mg/ha, just under the paper's global maximum of 910.5 Mg/ha).
  - **Sensitivity (section 5.6 only):** global Hmax power model, **AGB = 2.572 · Hmax^1.5191**, R² 0.70, RMSE 180 Mg/ha, with RH98 used as Hmax. Applied to the same heights to give a second set of carbon totals; no retraining.
  - The Hba-based equations (e.g. global 3.254 · Hba^1.5295) were not used: they expect basal-area-weighted height, which is about 0.63 × Hmax, and GEDI RH98 is a top-of-canopy height.
- Remove implausible biomass values from the loss: labels with RH98 above **33.9 m** (or below 0) are dropped for both equations. The GEDI export keeps 0–50 m; this cap is applied locally.
- **Labels belong to the 25 m cell, not to one 10 m pixel** (decided 2026-11-23; code `code/processing/gedi/cells.py`, `make_labels.py`). A cell's label = the mean biomass of its shots (each shot converted first, then averaged over months). A cell overlaps a 3 × 3 window of 10 m pixels, each weighted by its share of the cell's area (weights sum to 1). A label is never copied onto every pixel (that would count one measurement about 6 times), and the centre pixel alone is only kept for reference (the monthly raster has already snapped each shot to its cell, so the exact footprint position is unknown).
  - **Pixel models (RF, XGBoost):** one row per cell; features = the weighted mean of the 21 channels over the cell's window.
  - **U-Nets:** the network predicts biomass for every 10 m pixel; the loss compares the weighted mean of the predictions over the cell's window with the cell's label.
  - **Evaluation for all four models is at the cell level** (cell-mean prediction vs cell label), so RQ1 compares them on the same unit.
- **Go/no-go:** count usable footprints. Several thousand+ = proceed; ~1,000–3,000 = workable (consider sensitivity 0.95); <1,000 = discuss with supervisor.
- Record footprint counts after every filtering step (table for methods chapter).

### 2.4 Satellite inputs — 21 channels, 2019–2023 median composites
| Source | GEE dataset | Layers (count) |
|---|---|---|
| Sentinel-2 L2A | `COPERNICUS/S2_SR_HARMONIZED` | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (10) + NDVI, NDRE, NDMI, EVI (4) |
| Cloud mask | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | `cs_cdf` ≥ ~0.6 = clear (mask only, not an input) |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | VV, VH (dB), VV/VH ratio (3) |
| ALOS-2 PALSAR-2 | `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` | HH, HV (dB), HV/HH ratio (3) |
| Copernicus DEM | `COPERNICUS/DEM/GLO30_2024_1` (`GLO30` is deprecated and superseded by this) | elevation (1). **Slope dropped for now** (mangroves are flat; DSM noise dominates). May be added later as an ablation. |

- **Temporal composite:** one **median over all clear scenes in 2019–2023** (a single composite, not five yearly files and not a median of yearly medians). Reasons: the labels are effectively static (the mask keeps only mangrove stable 2015–2020 and canopy structure changes little over five years compared with the equation error), a five-year median is cleaner and has fewer cloud gaps in a cloudy region, and it is about 5× cheaper to export. Cost: each pixel's median uses about 5× more scenes (727 vs 147 in a 9 km test window), so Earth Engine memory must be checked with a small test export. Fallback if it fails: export yearly files and take the median across years locally. Single-year exports remain possible (`year_image(..., 2020)`).
- **Sentinel-2:** exclude B1, B9, B10 (atmospheric, 60 m). The six 20 m bands (B5, B6, B7, B8A, B11, B12) are medianed at 20 m, then resampled to 10 m with **bicubic** and clamped at ≥ 0 (bicubic can overshoot near sharp edges). Resampling every scene first ran out of Earth Engine memory. Reflectance scale ÷10,000.
- **Sentinel-1:** IW mode, **one orbit direction**: descending, the only direction acquired over Sabah (Dong et al. used ascending, which is not available here). No focal-mean speckle filter: the yearly median over many scenes already averages out speckle, and a filter would blur narrow fringes. Median.
- **PALSAR-2:** convert DN to gamma-naught (dB): **γ⁰ = 10·log₁₀(DN²) − 83.0**; no focal-mean filter (the median of the 2019–2023 mosaics and the coarse 25 m pixels already limit speckle); median of 2019–2023 mosaics; bicubic resampling from 25 m to 10 m (applied to DN, with DN clamped at ≥ 1 before the dB conversion).
- **Copernicus DEM:** bilinear resampling from 30 m to 10 m (smooth surface, so sharper interpolation adds nothing).
- **Do NOT add latitude/longitude as inputs** (lets models memorise location; breaks spatial validation).
- Normalisation statistics (mean/std per channel) computed from **training data only**.
- **Missing values (decided 2026-11-24):** PALSAR HH/HV cover 98.92% of mangrove, so 89 of 12,638 label cells (0.7%) have no PALSAR pixel in their window (about 1% of mangrove pixels at mapping time). **Fill missing values with the training mean** (0 after normalisation), identically for all four models, using `code/processing/features/normalise.py`. No cell is dropped and the split is unchanged. No separate accuracy check for these cells is planned.
- Store exports compactly (16-bit, scaled: reflectance ×10,000, DEM ×10 m, backscatter dB ×100; nodata −32768). One export per layer; splitting into regional pieces is not needed for now (Drive splits large files automatically).

## 3. Data split — shared by ALL models
- **Design (decided 2026-11-23): one random split of the label cells, 70/15/15 (train / validation / test), fixed seed (42), the same for all four models.** The same approach as Dong et al. (2023), the closest published method (random 7:2:1 split). Written once to `data/processed/labels/gedi_split.csv` (code `code/processing/split/make_split.py`): 8,846 / 1,896 / 1,896 cells.
- **Known limitation:** GEDI cells lie in dense tracks, and cells within about 4 km resemble each other (variogram range 3.4–4.5 km). With a random split 71% of validation/test cells have a training cell within 100 m (99% within 1 km), so the scores will be optimistic compared with predicting a new area (Ploton et al., 2020). State this in the dissertation and do not compare R² with spatially validated studies as if equal.
- **Earlier design (dropped, 2026-11-23):** a held-out region plus 10 km spatial blocks, which would have avoided this leakage. The user chose the random split for now. A spatial check can be added later (a held-out area, possibly in another state); it would be an extra evaluation, not a change to this split.
- **Tiles for the U-Nets:** a 128×128 tile can contain cells from all three sets. The loss and the metrics use only the cells of the set being trained or evaluated (imagery is shared, labels are not).
- Save the split as a file; **never change it once fixed**. Normalisation statistics come from the training cells only.

## 4. Models

| Model | Input | Notes |
|---|---|---|
| Random Forest | Cell table (21 features, weighted mean over each labelled cell's 3 × 3 pixel window) | Malaysian benchmark (H. Zhou et al., 2025) |
| XGBoost | Same cell table | Malaysian benchmark (Abu Bakar et al., 2024) |
| Attention U-Net | 128×128 tiles | ResNet-34 encoder + attention gates (Oktay et al., 2018; Dong et al., 2023) |
| Nested U-Net (UNet++) | 128×128 tiles | `smp.UnetPlusPlus`, same encoder as attention U-Net (Z. Zhou et al., 2018) |

- Optional extra baseline: **null model** (predict training mean) — every model must beat it.
- Fairness: both U-Nets use identical inputs, labels, split, loss, augmentation and ensemble size.

### Training setup (both deep models)
- **Loss:** masked Huber (labelled cells only, on the weighted mean of the predicted pixels in each cell's window) **+ λ·masked cross-entropy** on biomass classes (λ ≈ 0.5; ~6–8 quantile-based bins incl. a top "dense stand" bin). Final prediction = regression head.
- **Optional:** shift-resilient loss (±1 pixel, one shared shift per GEDI track, tracks ≥10 shots) — only if footprint-level GEDI is used.
- **Train from scratch** (ImageNet weights hurt with Sentinel inputs — Pauls et al., 2024).
- Augmentation: flips, 90° rotations, transposes only (no small-angle rotation/scaling — smears sparse labels).
- AdamW (lr ~1e-3, wd ~1e-4), reduce-on-plateau or cosine, **early stopping on validation RMSE**, mixed precision.
- **Ensemble:** 5 models per architecture; mean = prediction, spread = per-pixel uncertainty. Drop to 3 for both if compute-limited.
- **Fold ensemble (decided 2026-11-25, all four models), two phases on the pooled train + validation cells (85%, 10,742) of `gedi_split.csv`; test cells (15%) never used and identical to the split file:** (1) **Tuning** (RF/XGBoost only, no ensemble, once): the pool is cut into 5 random folds (`Model Analysis/data/cv/tuning_folds.csv`, write-once, seed 42); each setting is fitted on 4 folds (68% of all cells) and scored on the tuning fold (17%, fold 0); the winner is fixed. (2) **Performance** (all four models): the same pool is shuffled again and cut into 5 new random folds (`Model Analysis/data/cv/cv_folds.csv`, write-once, seed 43, never changed). Member k (its own seed) is fitted once on the 4 folds outside fold k (68%) and scored on fold k; cross-validation score = out-of-fold predictions (`<model>_oof.csv`) averaged over all pooled cells (the tuning cells are spread over the new folds, so this score is slightly optimistic; the test score is the clean headline). Test prediction = mean of the 5 members, uncertainty = their spread. U-Nets use the performance cut, each member early-stops on its own fold. Seed spread therefore also includes training-subset variation.
- Mapping: overlapping tiles (50%) with blended overlaps; mask output to the full mangrove mask.

## 5. Evaluation and validation
1. **Internal (vs held-out GEDI, at the cell level):** R², RMSE, relative RMSE, bias, MAE on the test cells; **errors by biomass range**; **block-bootstrap** confidence intervals (blocks of cells, so spatial dependence is not ignored); aggregated accuracy in two forms (decided 2026-11-24): (a) **test-cell version**, test cells averaged in 50 m (0.25 ha) and 100 m (1 ha) squares that hold 2+ test cells (`evaluation/aggregate.py`; small sample, about 20 and 93 squares, so an indication only); (b) **against independent data** (items 4–5 below), averaging the final map over 0.25–1 ha.
2. **Benchmarking:** RF, XGBoost, null model vs both U-Nets (RQ1).
   - **Robustness checks (decided 2026-11-25):** label-noise floor from repeat GEDI shots (best achievable RMSE), calibration slope and bias as % of the mean, seed-to-seed variance (a model difference must exceed it to name a winner), and a spatial hold-out check (5 folds of 10 km blocks, 5 km buffer; an extra evaluation, the main split is unchanged) to test whether scores and the model ranking survive in unseen areas.
3. **Uncertainty check:** bin test pixels by predicted uncertainty; mean error must rise with uncertainty.
4. **Independent field values:** Lee et al. (2024) Sabah mean AGC ≈ 95 ± 52 Mg C/ha (5 studies); H. Zhou et al. (2025) Matang.
5. **Product intercomparison:** Simard et al. (2019), GEDI L4B (1 km), ESA CCI Biomass, Lang et al. (2023) height.
6. **Sensitivity:** carbon totals under a second published height–biomass equation (not testable via GEDI test labels).
7. Plausibility: value ranges, spatial patterns, no tile/cloud artefacts.

## 6. Carbon conversion and outputs
- Carbon = AGB × **0.451**; CO₂e = carbon × **44/12**.
- Totals with uncertainty (ensemble spread of totals) by **district** and **protection status** (RQ2).
- Outputs: 10 m AGB map, uncertainty map, carbon/CO₂e maps, summary tables, interactive web app.

## 7. Interactive web app — "Sabah Mangrove Carbon Explorer"
- **Purpose:** delivers Objective 3 — lets agencies, researchers, carbon project developers and examiners explore the results without specialist software, and see how reliable they are.
- **What to build:** an interactive web map application that displays the final results. It loads **pre-computed** maps and district/protected-area totals, so it stays fast; no models run inside the app.
- **Map layers (toggleable, with legends):**
  - Above-ground biomass (Mg/ha), 10 m
  - Per-pixel uncertainty (ensemble spread)
  - Carbon (Mg C/ha) and CO₂e (Mg CO₂e/ha)
  - Mangrove extent (full mask)
  - District boundaries and protected areas
- **Interaction:**
  - Clicking a pixel shows the biomass, carbon, CO₂e and uncertainty at that point.
  - Selecting a district or protected area shows its total carbon and CO₂e with an uncertainty range, its mangrove area and its mean carbon density.
  - Summary panel/chart comparing districts and protected vs unprotected mangroves.
- **About/methods panel:** data sources, period (2019–2023), model used, accuracy on independent test data, and key caveats (GEDI-derived labels, saturation in dense stands, uncertainty meaning). Link to the code repository for reproducibility.
- **Design:** simple and readable — clear legends with units, sensible colour ramps (sequential for biomass/carbon, distinct for uncertainty), Sabah-wide default view.

## 8. Findings to report
The dissertation, web app and presentation should communicate:
1. **Model comparison (RQ1):** metrics table for RF, XGBoost, attention U-Net and nested U-Net (plus null model), on the test cells, with confidence intervals; errors by biomass range; which model performs best and by how much; whether spatial context (U-Nets) beats pixel-based models.
2. **Reliability:** whether uncertainty tracks real error; how estimates compare with published field values and existing biomass maps; where the model struggles (e.g. dense stands, narrow fringes).
3. **Carbon baseline (RQ2):** Sabah's total mangrove above-ground carbon and CO₂e with uncertainty; mean carbon density; mangrove area mapped.
4. **Distribution:** totals and densities by district and by protection status; spatial patterns (where high-carbon mangroves are, how much lies outside protected areas).
5. **Sensitivity:** how much totals change with a different height–biomass equation.
6. **Implications and limitations:** relevance for conservation priorities and blue carbon credit schemes; limits of GEDI-derived labels, saturation, period (2019–2023) and above-ground-only carbon.

# Tools

## 9. Rules for Claude when working on this project
- Follow the fixed decisions above; if a change seems needed, **propose it and explain why** before changing anything, and update this file.
- **Never** add lat/lon inputs, change the split once fixed, or compute normalisation from non-training data.
- Keep every model's inputs, labels and split identical so RQ1 comparisons stay fair.
- Prefer simple, reproducible code; set and record random seeds; log dataset IDs, dates and parameters for each export/run.
- When unsure of a GEE dataset path or band name, tell the user to check the catalogue rather than guessing.
- Writing: APA 7th edition; plain, concise academic English; disambiguate "H. Zhou et al. (2025)" vs "Z. Zhou et al. (2018)". Proposal limit was 1,500 words (excluding title and references).
- The user prefers short, simple explanations and step-by-step guidance; explain the "why" behind each step.

## 10. Agents
Project agents live in `.claude/agents/`. Use them proactively, without waiting to be asked:

| Agent | Use it when | Can edit files? |
|---|---|---|
| `planner` | Before any non-trivial change to the data pipeline, models or evaluation | No |
| `code-reviewer` | After modifying any script, before calling the task done | No |
| `debugger` | On a traceback, a failed GEE task or a failing test | Yes |
| `test-runner` | After any implementation or fix, as the final check | No |

Rules for all agents:
- They must follow the fixed decisions in section 2 and the rules in section 9. If an agent suggests breaking one, stop and ask me first.
- The `debugger` fixes only the bug it was given. It does not refactor or change data decisions.
- Summarise each agent's result for me briefly before moving on.

## 11. Reports and notes
Vault: `Project Context/Blue Carbon Mapping/` (relative to this file). It is the project's persistent notebook.
- At the start of a session, read `Context.md` and the latest entries in `Log.md`. If either is missing, create it.
- Keep `Context.md` current: status of each phase, decisions, open questions.
- When I say "log this", a phase wraps up, or a session wraps up, append a dated entry to `Log.md`: specify what has been done and how. Take screenshots or place results here.
- Use [[wikilinks]] and YAML frontmatter (date, tags).

## 12. Core references
- Abu Bakar et al. (2024) — RF/XGBoost mangrove AGB, Peninsular Malaysia.
- Bunting et al. (2022) — Global Mangrove Watch v3.0.
- Dong et al. (2023) — attention U-Net + GEDI biomass, Guangdong (closest method; GEE workflow followed).
- Dubayah et al. (2020) — GEDI mission.
- Lang et al. (2023) — global 10 m canopy height, sparse GEDI supervision, ensembles.
- Lee et al. (2024) — Malaysian blue carbon review; Sabah understudied; mapping a national priority.
- Nascetti et al. (2023) — BioMassters benchmark (UNet++ among top solutions).
- Oktay et al. (2018) — attention gates.
- Pauls et al. (2024) — shift-resilient loss; ImageNet pretraining hurts.
- Ploton et al. (2020) — spatial validation.
- Simard et al. (2019) — mangrove canopy height; height–biomass equations.
- H. Zhou et al. (2025) — Matang mangrove AGB (S2 + ALOS-2, RF).
- Z. Zhou et al. (2018) — UNet++.
