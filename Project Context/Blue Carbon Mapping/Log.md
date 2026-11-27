---
date: 2026-11-23
tags: [log, blue-carbon, data-preparation]
---

# Log

Chronological record of what was done and how. Current state and reasons are in [[Context]].

## 2026-11-22 to 2026-11-23: data-preparation decisions, code changes and test exports

### Inputs and slope
- Added the remaining Sentinel-2 bands (B5, B6, B7, B8A, B11, B12) and the radar bands to the satellite stack. Assessed the slope channel: cheap to compute but not useful on flat mangrove, so it was dropped (21 channels). `Project.md` 2.4 updated.

### Study area, mask encoding and margin
- Clarified that the satellite data covers the mangrove zone only, while the mask and GEDI labels share the full-Sabah grid. Kept one common grid, with nodata outside the zone.
- Mask changed to binary 1/0 with no nodata (was 1/0/255).
- Measured the zone area for different margins by dilating the mangrove mask on a 100 m grid: **200 m = 4,493 km2, 500 m = 5,982 km2**, against 2,389 km2 of mangrove. Margin set to **200 m** (500 m was too wide and defeated the purpose of the inner mask).
- Removed `finalize_tifs.py`. Nodata is masked when loading instead.

### Resampling and filtering decisions
- Resampling: bicubic for the six Sentinel-2 20 m bands and for PALSAR 25 m (both clamped, because bicubic overshoots at sharp edges), bilinear for the 30 m DEM.
- First attempt at per-scene bicubic S2 resampling ran out of Earth Engine memory ("User memory limit exceeded") on a 9 km window. Replaced by: median at 20 m, `.reproject("EPSG:32650", None, 20)`, then `.resample("bicubic").max(0)`. Verified alone and inside the combined S2 composite.
- Sentinel-1 descending only (only direction over Sabah); no focal-mean filters on S1 or PALSAR (median already averages speckle).
- Spec 2.2, 2.3 and 2.4 updated to match; GEDI filter wording changed to `0.98 <= sensitivity <= 1`.

### Sabah outline: geoBoundaries to GAUL 2025
- Compared the two outlines at four coastal sites (plot `outline_compare.png`, in the vault). geoBoundaries cut off real mangrove; GAUL 2025 keeps about 290 km2 more (stable mangrove 2,675 vs 2,382 km2 in a 30 m test).
- Switched every script to `FAO/GAUL/2025/level1` (GAUL0_NAME "Malaysia", GAUL1_NAME "Sabah"): `composites.py`, `export_gedi.py`, `labels.py`, `mask/boundary.py`, `inspect_sat.py`, `export_mask.py`.
- Enlarged the grid to cover the GAUL 2025 bounding box: 43,995 x 36,200 at 10 m, top-left (312050, 816500); GEDI 17,598 x 14,480 at 25 m. Spec and docs updated; no export was started at this stage.

### Storage format and derived channels
- Static file changed to **int16** (DEM x10, dB x100, nodata -32768). Precision loss is far below the accuracy of the sources.
- Wrote `code/processing/satellite/derive_channels.py`: computes NDVI, NDRE, NDMI, EVI, VV-VH locally from the yearly files and HV-HH (and optional slope) from the static file. Tested on synthetic data and real previews.

### GEDI labels
- Filters tightened: sensitivity 0.98-1, power beams 5, 6, 8, 11, night shots (`solar_elevation < 0`). Beam IDs are consistent with the data but not listed in the catalogue.
- Fixed a bug where `label_image` returned an Element (`copyProperties`), by wrapping it in `ee.Image(...)`.
- Decided to keep the monthly raster (rather than footprint points) and rasterise labels at the 10 m pixel at the cell centre in a later local step.
- Counted labels per filter step over a 1 degree Sandakan window: 1,561,431 -> 628,966 -> 466,662 -> 97,562 -> 75,499 -> 37,955 -> **4,702** on mangrove (earlier check of the same window: 4,704).

### Test exports (small batch tasks, Drive folder `BlueCarbon_test`)
- Started four tasks: `t2_mask`, `t2_static`, `t2_2020` (9 km preview window) and `t2_gedi` (13 km Sandakan Bay window). They stayed in READY for about an hour (restricted mode), then completed without errors.
- Downloaded to `Data Preparation/test_data/` and checked: grids, dtypes, band order, value ranges, valid fractions (35% = mangrove + 200 m zone). All passed.
- `t2_gedi` came back with no labels: only about 61 shots passed all filters in that window and none fell on mangrove. The GEDI stack was then checked at a densest-label 3 km window by interactive download: 203 values in 202 cells, matching an independent count, values 10.0-29.5 m. Note: Earth Engine `.sample()` on these images returned nothing even where labels exist, so `reduceRegion` was used instead.
- Ran `derive_channels.py` on the real test files: NDVI median 0.74, NDRE 0.53, NDMI 0.32, EVI 0.51, VV-VH 6.5 dB, HV-HH -6.2 dB, slope median 2.8 degrees.

### Full exports started
- `export_mask.py`: added `.toUint8()` so the output matches the verified test file; started task `SULB73HEF2ZIXOODMCXCFMP2` -> `sabah_mangrove_mask_v5.tif` (Drive `BlueCarbon`).
- `export_gedi.py`: started task `5RSIV2OBBSZ5ZOFQ6E53XLTB` -> `sabah_gedi_rh98_monthly_v2.tif` (Drive `BlueCarbon`).
- Static and yearly satellite exports are not started. An earlier GEDI export was started without approval and cancelled at the user's request; exports are only started on an explicit go-ahead.

### Height-to-biomass equation
- Searched for the Simard et al. (2019) equations. The ORNL DAAC page failed on a certificate error and the Nature, CORE and MDPI copies were blocked for the fetch agent, so the paper was read through the user's connected Chrome session (Methods text) and the user pasted Supplementary Table 8.
- Findings: the paper's biomass equations are `AGB = a * Hx^b` on Hba (or Hmax); regional models were fitted for the Americas, East Africa and South Asia only; for Southeast Asia it used the published Aslan et al. (2016) model `exp(3.9042 + 0.0858 * H_CHM)` (R² 0.55, RMSE 148 Mg/ha). Earlier quoted fit statistics (R² 0.77, RMSE 57.2) came from a search summary and did not match the paper; the exponential was also misjudged as a different 2018 equation. Both were corrected.
- Decision (user approved): **main** = Southeast Asia exponential on rh98; **sensitivity** = global Hmax power model `2.572 * Hmax^1.5191`. Hba models rejected (they would need an unverified ~0.63 Hba/Hmax conversion).
- Cap: the exponential reaches the paper's maximum AGB (910.5 Mg/ha) at 33.9 m; labels above 33.9 m or below 0 are dropped for both. A first cap of 34 m was set, found to give 917 Mg/ha, and changed to 33.9 m.
- Wrote `code/processing/gedi/biomass.py` (`agb_main`, `agb_sensitivity`, `carbon`, `H_CAP = 33.9`) and `check_biomass.py` (hand-computed values at 5-30 m, cap and range, carbon factor); all tests pass.
- Updated `Project.md` 2.3 (replaced the TODO) and `Plan.md`; added section 4 to [[Context]].

### Satellite inputs: single 2019-2023 composite instead of yearly files
- Spotted a mismatch: the spec said "2019-2023 median composites" but the code and `Plan.md` produced five yearly files. Compared the two options (cost, cloud gaps, label timing, flexibility) and the user chose the single composite.
- `composites.py`: added `PERIOD = (2019, 2023)` and `date_range()`; `s2_collection`, `s2_composite` and `s1_composite` now take a year or a (first, last) range. `export_satellite.py`: `year_image(geom, zone, years=PERIOD)`, new modes `testc` (small-window Drive test) and `full composite` (writes `sabah_composite_2019_2023`); single-year modes kept as a fallback. `derive_channels.py` needs no logic change (same 12 bands); its docs now point at the composite.
- Checked without exporting: the image builds with the same 12 bands in the same order; the 9 km test window holds 727 Sentinel-2 scenes for 2019-2023 against 147 for 2020, so each pixel's median uses about 5x more scenes. Whether the full-size composite fits Earth Engine memory is untested.
- Updated `Project.md` 2.4 (new "Temporal composite" bullet), `Plan.md` (status, composite section, workflow, sizes) and [[Context]]. Nothing was exported.

### Mask v5 and GEDI v2 exports: results
- Both tasks finished (COMPLETED, no errors) and the files were downloaded to `data/` by the user; the old v3/v4 masks, old inner mask and old GEDI pieces are no longer there.
- Started the composite test export `test_composite` (task `2J3VPPQG3JSEQPS35HS7U75H`, 44 km window). First attempt failed with `ModuleNotFoundError` because `python -I` does not put the script folder on the path; reran without `-I`. Still queued (READY) at the last check.
- **Mask v5:** grid, dtype and values correct; 26,795,143 mangrove pixels = 2,679.5 km2 (267,951 ha).
- **Inner mask:** `inner_mask.py 30` on v5: 1,916 km2, 71.5% kept (27 s).
- **GEDI v2:** `build_gedi_vrt.py` stitched 12 pieces, 48 bands. Wrote `code/processing/gedi/count_labels.py` (4 min for the full state) to count labels after each local step: 18,336 valid shots -> 17,843 on the full mask -> 12,907 on the inner mask -> **12,800** with rh98 <= 33.9 m (12,638 cells). Go/no-go: go. Biomass with the main equation gives mean carbon 101 Mg C/ha, close to the Sabah field mean of 95 ± 52.
- Updated `Plan.md` (status, mask, inner mask and GEDI sections) and [[Context]].

### Label tables and how labels attach to pixels
- User asked whether placing a label on every pixel of the footprint is acceptable and challenged the centre-pixel option as equally inaccurate. Explained: the centre pixel (A) and copying onto all pixels (D) share the unknown footprint position, but D also counts one measurement about 6 times; the monthly raster has already snapped each shot to a 25 m cell, so the exact position is unknown anyway. Explained B (average the 21 channels over the cell's pixels, for RF/XGBoost) and C (U-Net loss on the weighted mean of its predictions over the cell, for the U-Nets). User chose B and C.
- The proposed A/B/C experiment could not run: the 9 km preview window holds 0 labels and the 44 km composite-test window 8 cells (4 on the inner mask), and a fair comparison needs the spatial split. Decision taken on the reasoning instead; the comparison can still be run later because the table stores the centre pixel too.
- Wrote `code/processing/gedi/cells.py` (a 25 m cell always overlaps a 3 x 3 window of 10 m pixels; weights from the overlap area; `cell_mean` skips NaNs) and `check_cells.py` (weights sum to 1, hand-computed patterns 0.4/0.4/0.2 and 0.2/0.4/0.4, window inside the grid at the last cell, centre pixel inside the window, NaN handling); all pass.
- Wrote `code/processing/gedi/make_labels.py` (3 min): `data/processed/labels/gedi_shots.csv` (12,800 rows) and `gedi_cells.csv` (12,638 rows; biomass averaged over the cell's shots after conversion; 162 cells have two shots). Counts match `count_labels.py`. Spot check: 25 random shots against the raw VRT, 0 mismatches. `mangrove_frac` is 1.0 for every cell (the inner mask puts selected cells at least 30 m from non-mangrove).
- Updated `Project.md` 2.3 (rasterise line replaced by the cell-label rule), section 4 (RF and XGBoost table rows, loss), section 5 (cell-level evaluation); `Plan.md` and [[Context]].

### Spatial split: design decision and label coverage
- Explained why a whole region is held out (stricter test than blocks; coasts differ; district totals depend on regional bias) and compared a single fixed split with a held-out region plus 5-fold block cross-validation. User chose the simple fixed split and kept block cross-validation as an optional extra (likely not used). Added this to `Project.md` section 3.
- Label coverage: wrote `code/processing/gedi/label_coverage.py`. It fetched the 25 Sabah districts from `FAO/GAUL/2025/level2` (property `GAUL2_NAME`, simplified to 100 m, saved as GeoJSON), assigned each label cell to a district (one point on a shared border matched twice and was de-duplicated; 2 points outside every polygon were snapped to the nearest), estimated mangrove area per district from the mask, and drew `label_coverage.png` (in the vault: map of cells coloured by biomass plus bars per district; the first version had the colour-bar label colliding with the district names and stray axis labels, fixed and re-checked by eye). Outputs: `data/processed/labels/label_coverage.csv` and the figure.
- Result: 91% of labels are on the east coast (Kinabatangan 35%, Tawau 25%, Beluran 17%), 5% in the north, 3% in the west. 9 of 25 districts have no labels. Details in [[Context]] section 6. Nothing was saved as a split.

### Composite test export: result
- Task `2J3VPPQG3JSEQPS35HS7U75H` finished COMPLETED (queued 15:57-16:41, ran 16:41-17:59). The user moved `test_composite.tif` (149 MB) into `test_data/`.
- Verified: grid, int16, 12 bands, same valid pixels in every band (21.2% of the window), 0 mangrove pixels without data, plausible values at mangrove pixels, `derive_channels.py` runs on it (NDVI median 0.79, VV-VH 6.3 dB).
- Compared with the single-year 2020 test on the 9 km overlap. My first comparison gave correlations near 0.05, which was a bug in my own check (the row offset had the wrong sign); with the offset fixed the correlation is 0.97-0.99 and the mean absolute difference 0.006-0.012 reflectance and about 0.45 dB.
- Runtime for the full state is the open question: zone 10.8x and grid 81x the test, so about half a day to several days. See [[Context]].

### Full satellite exports started
- On the user's go-ahead (one composite task, no regional pieces): `export_satellite.py full static` -> task `V2VB3T7GMKT6NEP2T3PNXB4N` (`sabah_static`) and `export_satellite.py full composite` -> task `CH6EQ4J7IFQOD5IIMMCTCB53` (`sabah_composite_2019_2023`), both to Drive folder `BlueCarbon`. A Drive search found no existing `sabah_static` file, so no stale copy.

### Removed the local test data
- Deleted the `test_data/` folder (243 MB: the four small test exports, the composite test and their derived-channel files). Nothing reads it, and the results are recorded here and in [[Context]]. Kept the biomass and cell checks and the `test`/`testc` modes of `export_satellite.py`.

### Data folder split into raw and processed
- `data/` now has `raw/` (as downloaded from Earth Engine or fetched once, never edited: `mask/` with the full and inner masks, `gedi/`, `satellite/`, `boundaries/`) and `processed/` (built locally from raw: `labels/` and `satellite/` for derived channels). The GEDI `.vrt` stays with its pieces in `raw/gedi/`.
- Updated the paths in `count_labels.py`, `make_labels.py`, `inner_mask.py`, `label_coverage.py`, `build_gedi_vrt.py`, `export_gedi.py` and `derive_channels.py`. Two behaviour changes: `inner_mask.py` creates its output folder, and `derive_channels.py` now writes to `data/processed/satellite/` and no longer next to its input.
- Checked after the move: unit tests pass; `inner_mask.py` gives the same 1,916 km2 (71.5%); `label_coverage.py` and `make_labels.py` give the same 12,638 cells and 12,800 shots; `derive_channels.py` tested on a small synthetic file (output landed in `processed/satellite/`).

### Cleanup
- Deleted scratch files in `test_data/`, the `__pycache__` folders, and the vault's `Welcome.md` (user request); removed the "Never edit `Welcome.md`" line from `Project.md`.
- Old data in `data/` (v3/v4 masks, inner mask, 12 GEDI pieces) kept until the new exports are downloaded and verified.

### Spatial split: variogram, literature check and change to a random split
- Wrote `code/processing/split/region.py` (district of each label cell) and `variogram.py`. Variogram range 3.4 km (biomass) and 4.5 km (height), nugget about 60% of the variance; figure `variogram.png`. A code review found nothing critical; added a check that every cell has a district.
- Measured the leakage of a random split: 71% of test cells had a training cell within 100 m (99% within 1 km). With 10 km blocks: 1% within 100 m, 24% within 1 km.
- Checked how the closest papers split their data: Dong et al. random 7:2:1 on patches (with 5-fold cross-validation); Lang et al. 20% of 100 km tiles held out plus independent LiDAR; H. Zhou et al. leave-one-out on 46 plots. Two papers (Abu Bakar, Ploton) were unreadable.
- User chose a **random 70/15/15 split of the cells** for now and dropped the held-out region and spatial block design. Wrote `make_split.py` and `data/processed/labels/gedi_split.csv` (seed 42; 8,846 / 1,896 / 1,896 cells; similar biomass in each). Updated `Project.md` sections 1 (novelty wording), 3, 5, 8 and 9 (the ban on random splits), [[Context]] section 6 and the layout in `Plan.md`. Deleted the region map. The limitation is recorded in `Project.md` section 3.


## 2026-11-24: static data verified, cell feature table built

### Static satellite data
- Earth Engine: `sabah_static` COMPLETED (4 pieces in Drive); the composite task is still READY (never started; restricted mode). Static pieces downloaded to `data/raw/satellite/`.
- Verified: the 4 pieces tile the 43,995 x 36,200 grid exactly (EPSG:32650, 10 m, int16, bands DEM_x10 / HH_x100 / HV_x100). DEM covers 100% of mangrove (median 9.8 m, a surface model so it includes canopy); HH and HV cover **98.92%** (the docs said 99.4%); 89 of the 12,638 label cells have no PALSAR pixel in their window (61 train, 15 val, 13 test).
- `derive_channels.py` is deliberately not run until the composite is done.

### Dong et al. (2023) comparison
- The fetch agent failed on the 17 MB PDF (10 MB limit); downloaded it locally and read the full text. Not a mangrove study (Guangdong forest). Details of the comparison are in [[Context]] section 7. No XGBoost in the paper; their RF uses all 29 channels incl. lat/lon, our RF/XGBoost use 21 without coordinates.

### build_cell_features.py
- `planner` agent scoped it; defaults chosen for its open questions (partial vs final outputs, ddof 0, raw values stored, no coordinates in the table, stitching across pieces, shared `channels.py`, write-once final statistics).
- Wrote `code/processing/features/channels.py`, `build_cell_features.py`, `check_features.py`. `code-reviewer` found no correctness bugs; fixed its three warnings: a missing or incomplete piece now stops the run (it could have locked in write-once statistics from incomplete data), lower memory (raw dtype strips, converted per cell only), more tests (float32 derived files with band names, grid edge, missing piece, main logic in partial and final mode). Also: split/cells row count assert, statistics file written before the table.
- Real run on the 3 static channels: 12,638 rows (8,846 / 1,896 / 1,896), `cell_features_partial.csv`, `norm_stats_partial.json`; HH and HV NaN for exactly 89 cells; 40 random cells re-read directly from the files agree to 4e-15.
- Docs: static status corrected in [[Context]]; section 7 added.

### Missing PALSAR values: decision
- The 89 label cells without a PALSAR pixel (and ~1% of mangrove pixels when mapping) get the training mean for the missing channels, the same for all four models (user decision). Added `code/processing/features/normalise.py` (z-score with the stored train-only statistics, NaN to 0) with tests in `check_features.py`; `Project.md` 2.4 updated. No separate accuracy check for these cells (user declined).

### Evaluation module (Model Analysis/evaluation)
- `planner` scoped it; conventional definitions chosen (see [[Context]] section 8). Wrote `metrics.py`, `blocks.py`, `predictions.py`, `null_model.py`, `evaluate.py`, `diagnose.py`, `check_evaluation.py`. First placed under `Data Preparation/code/processing/evaluation/` by mistake (I had not noticed the empty `Model Evaluation` folder); on the user's request moved to `Model Evaluation/code/`, then renamed to `Model Analysis/evaluation/` with an empty `models/` folder added.
- `diagnose.py` on the 1,896 test cells: 4 km blocks give 284 blocks (median 3 cells), used as the default. Aggregated accuracy: too few test cells per 50-100 m square (20 squares with 2+ at 50 m), so it cannot be done from test cells.
- Null model on the real split: test RMSE 132.6 Mg/ha [121.3, 143.9], R2 about 0, bias +0.4.
- `code-reviewer` (report took a reminder to arrive): no critical issues; fixed its four warnings and most suggestions, with new tests. All check scripts pass (biomass, cells, features, evaluation).

- 2026-11-24: Aggregated accuracy decision: keep both the test-cell version and the independent-data version. Added `evaluation/aggregate.py`, `results_aggregated.csv` and the printed table; Project.md 5.1 reworded. Null model: 50 m 20 squares (RMSE 95.8, no CI, 14 blocks), 100 m 93 squares (RMSE 117.9, CI 90.5-145.7). All checks pass.

## 2026-11-25 Composite arrived; final cell features
- Composite checked: 12 GeoTIFF pieces (3 rows x 4 columns, 13,568 px tiles) in `data/raw/satellite/composite/`, EPSG:32650, 10 m, origins and sizes tile exactly onto the 43,995 x 36,200 grid, 12 int16 bands with the names in `channels.py`, nodata -32768; value ranges plausible (B8 about 0.30, VV about -8.6 dB, VH about -15 dB).
- Files had been moved into `static/` and `composite/` subfolders: patterns in `channels.py` and the examples in `derive_channels.py` updated.
- `derive_channels.py` run for the composite (7 min) and static (2.5 min): 16 derived files in `data/processed/satellite/`.
- `build_cell_features.py`: all 12,638 cells have all 21 channels (no window pixel uncovered). Missing: only HH, HV, HVHH_db for the same 89 cells (filled with the training mean at normalisation). Wrote `cell_features.csv` and the write-once `norm_stats.json`; partial files deleted. Cross-checks: NDVI mean 0.81 matches (B8-B4)/(B8+B4); VVVH_db 6.42 = VV - VH; HVHH_db -6.24 = HV - HH.

## 2026-11-25 Robustness evaluations
- Built in `Model Analysis/evaluation/`: `noise_floor.py` (floor RMSE about 81 Mg/ha, R2 ceiling about 0.62 from 162 repeat-shot pairs), `bias_pct` and `calib_slope` in `metrics.py`, `spatial.py` (5-fold 10 km block hold-out with 5 km buffer; `make`, `null`, `evaluate`), `seed_variance.py`. Tests in `check_robustness.py` (all pass, as does `check_evaluation.py`).
- Spatial folds written once to `Model Analysis/data/spatial/spatial_folds.csv`. Null model spatial RMSE 132.2 (by fold 114-152); null model on the random split 132.6.

## 2026-11-25 RF and XGBoost scripts written (not yet run)
- `Model Analysis/models/train.py` (modes `main`, `seeds`, `spatial`) and `models/config.yaml` (tuning grids, seeds, fixed settings; the only file to edit for tuning). Decisions: squared-error loss for both; RF writes `pred_std` from the spread of its trees, XGBoost none; final models are refitted on train cells only (validation stays independent); spatial runs reuse the saved best settings with per-fold train-only z-score statistics.
- Written without running anything (user request): untested until the first run. Needs scikit-learn, xgboost and pyyaml installed.
- Revised 2026-11-25 (still not run): RF and XGBoost are now ensembles of one model per seed (5), prediction = member mean, `pred_std` = member spread (same definition as the U-Nets); `main` also writes the members as `<model>_s<seed>.csv`, so the separate `seeds` mode was removed. Loss: XGBoost pseudo-Huber (`huber_slope` 100 Mg/ha, base score set to the train mean), RF squared error because scikit-learn's forest has no Huber loss. Config comments updated.
- 2026-11-25: loss set to squared error for both RF and XGBoost (clean comparison; a pseudo-Huber XGBoost variant can be run later as a labelled extra). `xgboost` could not be installed: pip failed on an SSL certificate error, and with `--trusted-host` the download was refused (HTTP 403 from files.pythonhosted.org), so the network blocks it. RF can run without it.
- 2026-11-25: restructured the tree-model scripts (still not run): `models/rf.py` + `models/rf.yaml`, `models/xgb.py` + `models/xgb.yaml`, shared code in `models/train.py` (renamed from common.py) (data, tuning loop, ensemble, `main` and `spatial` modes). Replaces `train.py` and `config.yaml`. Run: `python models/rf.py main`, `python models/xgb.py main`, then `... spatial`.
- 2026-11-25: code review of `models/` (read-only, nothing run): no leakage, no API mismatches. Fixed: final models and the spatial runs now use the settings saved in `data/tuning/<model>_best.json`, not the current yaml. To mention in the write-up: the XGBoost tuning `val_rmse` is the minimum over boosting rounds on the validation cells (slightly optimistic, not a comparison number); the tree count tuned on the full train set is reused when spatial folds train on fewer cells.
- 2026-11-25: `models/common.py` renamed to `models/train.py` (rf.py and xgb.py import from it).

## 2026-11-25 - Fold ensemble for RF and XGBoost (all four models to follow)
- Decision: the 5 ensemble members differ by **train fold** as well as seed. The 8,846 train cells are cut once into 5 random folds (seed 42, `Model Analysis/data/cv/cv_folds.csv`, write-once). Member k is fitted without fold k (80% of train). Validation cells (tuning, early stopping) and test cells are not in the folds, so train and validation are not pooled. `gedi_split.csv` is unchanged.
- Code: `models/train.py` (`cut_folds`, `cv_folds`, `cv_size`, `ensemble(..., fold)` returning members and out-of-fold predictions, `final_params(..., k)`); `xgb.final` scales the tuned tree count by 1/0.8; `cv_folds: 5` in `rf.yaml` / `xgb.yaml` (must equal the number of seeds); `models/check_models.py` (synthetic checks, pass). Evaluation checks still pass.
- New output per model: `data/predictions/<model>_oof.csv` (each train cell predicted by the one member that did not see it). The mean prediction of train cells is not out-of-fold, so score models on val/test or the oof file.
- Spatial mode: each spatial fold's train-role cells are cut into 5 sub-folds (seed 42 + fold).
- Consequence: seed variance now = seed + training-subset variation. U-Nets must use the same fold file. `Project.md` section 4 updated.
- Nothing trained yet. Planner and code-reviewer used; review pending at time of writing.

### Update (same day): fold ensemble now pools train + validation
- Changed after the user's design: tuning still runs once (train cells -> validation cells, no CV inside tuning). Afterwards the pooled train + validation cells (85%) are cut into 5 random folds (`data/cv/cv_folds.csv`, write-once, test never included); member k is fitted once without fold k (68% of all cells) and scored on fold k. `<model>_oof.csv` = cross-validation predictions. XGBoost tree count scaled by member size / tuning size. `Project.md` section 4 updated; the test set is unchanged. Checks pass; nothing trained on real data yet; review of this revision pending.

### Update 2 (same day): tuning uses the fold split too
- Tuning now fits every grid combination on 4 of the 5 pooled folds (68%) and scores on the tuning fold (fold 0, 17%): the same split each member uses, so no tree-count scaling is needed (removed). `tuning_fold: 0` in both yaml files. The cross-validation score is reported on the 4 folds other than the tuning fold; the test cells stay the clean headline. The train/validation labels of `gedi_split.csv` now only define the pool (test is unchanged). Checks pass; nothing trained on real data yet.

### Update 3 (same day): two cuts, two phases
- Phase 1 (tuning, once, no ensemble): pool (train + validation, 85%) cut into 5 folds (`data/cv/tuning_folds.csv`, seed 42); fit on 4 folds, score on fold 0. Phase 2 (performance): the pool is shuffled again and cut into 5 NEW folds (`data/cv/cv_folds.csv`, seed 43); member k fitted once on 4 folds, scored on fold k; cross-validation score averaged over all pooled cells (no skipping of the tuning cells, user decision; slightly optimistic, stated). Test cells untouched. Checks pass; nothing trained on real data yet.

## 2026-11-26 - First real run: Random Forest and XGBoost
- Run order: fold files written first (`data/cv/tuning_folds.csv` seed 42, `cv_folds.csv` seed 43), then `rf.py main` and `xgb.py main` in parallel, then both `spatial` runs; scored with `evaluate.py`, `seed_variance.py`, `spatial.py evaluate`. A sandbox smoke test with tiny grids ran first and found no bugs (only a stale "validation RMSE" label, fixed).
- Tuning (fold 0 of the tuning cut): RF best 300 trees, min leaf 10, max_features 0.5 (95.61); XGBoost best depth 5, lr 0.03, min child weight 5 (95.96). All combinations within 0.6 (XGBoost) / 2.2 (RF) Mg/ha.
- Test cells (1,896): RF RMSE 94.5 [83.9, 106.1], R2 0.493, bias +1.5, slope 1.05; XGBoost 94.2 [83.3, 106.1], R2 0.496, bias +0.7, slope 1.03; null 132.6. Noise floor about 81 Mg/ha (R2 ceiling about 0.62). Paired XGBoost - RF RMSE -0.3 [-1.0, +0.4]; seed sd 0.1-0.2: no winner. Cross-validation RMSE 95.3 for both (consistent with test).
- Dense stands underestimated (400+ bin bias -165 vs -308 for the null model). Uncertainty: Spearman rho +0.27 (RF), +0.21 (XGBoost).
- Spatial hold-out (5 folds, 10 km blocks, 5 km buffer): RF and XGBoost RMSE 98.6, R2 0.438, null 132.2; same test cells random vs spatial model: +3% RMSE (upper bound on the random-split optimism). Both models beat the null in every fold (fold RMSE 87-119).
- Nothing committed to git this session (user instruction).

- 2026-11-26: compared the baseline evaluations with Dong et al. (2023, arXiv 2311.03067; forest-wide, scores not comparable) and noted ideas for better and more trustworthy predictions in [[Improvements]]. Nothing decided or built.

## 2026-11-27 - Review session: results, processing explained, git ignore rules
- **Results chart:** `Model Analysis/evaluation/plot_results.py` draws `baseline_results.png` from the saved results (run `python evaluation/plot_results.py` from `Model Analysis`). Reviewed once, then rewritten to fix the review findings; the rewrite has not had a second review. The five panels were explained: overall error, bias by biomass level, predicted vs observed (RF, XGBoost), member spread vs error.

![[baseline_results.png]]

- **Status:** RF and XGBoost baselines done (tied; weak on dense stands; uncertainty far too narrow). Next: U-Nets (open question: training hardware), then model comparison, uncertainty fix, map, carbon totals, independent checks, web app.
- **`.gitignore`:** data folders stay ignored except the small files needed to rerun models and evaluation (GEDI shot, cell and split tables; feature table and normalisation statistics; tuning, performance and spatial folds; tuning and results files). Raw and derived rasters and prediction files stay ignored. Checked with `git status` (24 new files, about 4 MB); nothing added or committed. `.gitignore` still lists itself.
- **Processing explained and written down** in [[Processing]]: each step from raw data to model inputs, in simple terms, with the responsible code.
- **Key clarifications** (also at the top of [[Context]]): GEDI height is not a model input; biomass labels are estimated from height with the equation; RF and XGBoost inputs are area-weighted averages of the 3 x 3 pixels; PALSAR resizing (bicubic) adds no real detail.
- **Context.md:** added "How the model works (read this first)" and corrected the outdated line that put labels on a single 10 m pixel.
- **Open:** back up or commit the fold files before moving to another machine; some older `Context.md` sections still describe the data-preparation stage; the code relies on Sentinel-1 having only descending passes over Sabah (no orbit filter).
- Up to this point no model, evaluation or data-processing code was changed and nothing was retrained.
- **Run times (added later the same day):** `models/train.py` now writes `data/results/timing_<model>_<mode>.json` (tuning, per-member fit and prediction seconds, machine details). Checks and a tiny-grid run on a throwaway copy pass; code-reviewed and findings fixed. The existing baselines have **no timing files yet**: they appear when the models are rerun.
- **Rerun to record times (2026-11-27):** `rf.py` and `xgb.py` (main, then spatial) run one at a time, then `evaluate.py` (null rf xgb; pairs rf:null, xgb:rf), `seed_variance.py` and `spatial.py evaluate`, as before. All folds, tuning logs, predictions and results match the 2026-11-26 outputs to within rounding (largest difference below 3e-13). 16-core CPU, `n_jobs` -1; one run each.

| Time (s) | Random Forest | XGBoost |
|---|---|---|
| Tuning (RF 30 settings, XGBoost 18) | 108.4 | 5.8 |
| Main: fit 5 members | 12.2 | 1.8 |
| Main: predict 12,638 cells, 5 members | 0.97 | 0.05 |
| Spatial: fit 25 members (5 folds) | 32.5 | 7.4 |
| Spatial: predict, all folds | 2.47 | 0.09 |
