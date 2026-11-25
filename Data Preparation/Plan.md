# Sabah mangrove canopy height: data preparation

Data for a mangrove canopy-height model of Sabah, Malaysia. Everything is built in Google Earth
Engine (project `carbon-mapping-mangroves`), exported to Drive as GeoTIFF, and downloaded into `data/raw/`. Everything built locally from those files goes into `data/processed/`.

## Status

| Dataset | Folder | Status |
|---|---|---|
| Mangrove mask | `data/raw/mask/` | v5 (GAUL 2025, new grid) exported, downloaded and verified |
| GEDI rh98 labels | `data/raw/gedi/` | v2 (stricter filters, new grid) exported, downloaded, VRT built, counts recorded: 12,800 training labels |
| Inner mask (30 m) | `data/raw/mask/` | Rebuilt from v5 |
| Satellite static (DEM, PALSAR) | `data/raw/satellite/static/` | Exported, downloaded (4 pieces) and verified 2026-11-24 (DEM 100% of mangrove, PALSAR 98.92%) |
| Satellite composite 2019-2023 (Sentinel-2, Sentinel-1) | `data/raw/satellite/` | Exported, downloaded (12 pieces in `data/raw/satellite/composite/`) and verified 2026-11-25 |

## Common grid

Mask and satellite files share one grid, so they line up pixel for pixel.

- CRS EPSG:32650 (UTM zone 50N), 10 m pixels
- 43,995 columns x 36,200 rows, top-left corner (312050, 816500)
- Extent: x 312,050 to 752,000 m, y 454,500 to 816,500 m (the GAUL 2025 Sabah bounding box, edges on 50 m multiples)

GEDI uses its native 25 m pixels over the same extent: 17,598 x 14,480, top-left corner (312050, 816500), so its
pixel edges coincide with the mask's every 50 m.

## 1. Mangrove mask

`data/raw/mask/sabah_mangrove_mask.tif`. Format: 1 band `mangrove`, uint8, binary, no nodata value, 17.6 MB. Verified:
EPSG:32650, 10 m, 43,995 x 36,200, top-left (312050, 816500), values only 0 and 1.

| Value | Meaning | Area |
|---|---|---|
| 1 | Stable mangrove | 2,679.5 km2 (26,795,143 pixels, 267,951 ha; under the 298,000 ha sanity limit) |
| 0 | Everything else (other land, sea, outside the Sabah outline) | rest of the grid |

- Stable mangrove = Global Mangrove Watch v4.0.19 (2020) AND GMW v3 mangrove in all of 2015-2020.
- Pixels that were mangrove in only some years are labelled 0.
- Sabah outline: FAO GAUL 2025 level 1. The earlier geoBoundaries ADM1 outline cut off real mangrove along the coast
  (about 290 km2 compared with GAUL 2025). See `outline_compare.png` in the notes vault (`Project Context/Blue Carbon Mapping/`).
- Built by `code/collecting/mask/mangrove_mask.py`, exported by `code/collecting/mask/export_mask.py`.

**Inner mask:** `data/raw/mask/sabah_mangrove_inner_mask.tif` (uint8, binary, no nodata; the 30 m distance is stored in the file tag `INNER_MASK_METRES`), rebuilt from the v5 export.
Same grid as the full mask. 1 = mangrove at least 30 m from any non-mangrove pixel (1,916 km2, 71.5% of the
full mangrove), 0 = everything else. Built locally by
`code/processing/mask/inner_mask.py [metres]`. The inner mask **selects the GEDI training labels** (a label cell is kept
when the inner mask is 1 at its centre pixel); the **full mask is used for mapping**.

## 2. GEDI labels

`data/raw/gedi/sabah_gedi_rh98_monthly_v2-*.tif` (pieces) plus `sabah_gedi_rh98_monthly_v2.vrt`, which stitches
them together. Keep the pieces next to the VRT. The v2 export replaced the older `sabah_gedi_rh98_monthly-*` files.

- Grid 17,598 x 14,480 at 25 m, 12 pieces. 48 float32 bands, one per month: `rh98_2019_04` ... `rh98_2023_03`.
- Value: GEDI L2A rh98 (relative height at 98% of returned energy), in metres above ground.
- Nodata −1. A pixel holds a height only if all of these hold (current `export_gedi.py`):
  - `quality_flag` = 1, `degrade_flag` = 0
  - sensitivity at least 0.98 and at most 1
  - power beam only (`beam` 5, 6, 8 or 11)
  - night-time shot (`solar_elevation` below 0)
  - rh98 between 0 and 50 m
  - the pixel is on mask value 1 (the full mask; the inner mask is applied later, at label selection)
- Source: `LARSE/GEDI/GEDI02_A_002_MONTHLY`. If two shots fall in one cell in one month, one is kept (not averaged).
- **Label counts** (all of Sabah, from `code/processing/gedi/count_labels.py`; a label cell is placed on the 10 m pixel at its centre):

| Step | Shots (monthly values) | Cells |
|---|---|---|
| Valid in the v2 export (Earth Engine filters, on mask = 1) | 18,336 | 18,093 |
| Centre pixel on the full mask | 17,843 | 17,609 |
| ... and on the inner mask (30 m): the training labels | 12,907 | 12,741 |
| ... and rh98 <= 33.9 m (biomass equation limit) | **12,800** | **12,638** |

  12,800 is "several thousand or more", so the go/no-go in `Project.md` 2.3 is **go**. 40 of 48 months have labels
  (median 130 per month, maximum 1,656). 2.7% of label centres sit on a mask pixel that is not 1 (25 m versus 10 m grid).
- Training labels: rh98 2.9 to 33.9 m, median 15.4 m (5th to 95th percentile 5.9 to 26.6 m). Biomass, main equation:
  median 186, mean 224 Mg/ha (mean carbon 101 Mg C/ha, close to the Sabah field mean of 95 ± 52 Mg C/ha from Lee et al.);
  sensitivity equation: median 164, mean 181 Mg/ha (mean carbon 82 Mg C/ha).
- Built by `code/collecting/gedi/export_gedi.py`; stitched by `code/processing/gedi/build_gedi_vrt.py`; counted by `code/processing/gedi/count_labels.py`.
- **Height to biomass** (`code/processing/gedi/biomass.py`, checks in `check_biomass.py`): the export holds rh98 in metres only. Labels are made locally with the Southeast Asia equation AGB = exp(3.9042 + 0.0858 · rh98) Mg/ha (Simard et al. 2019, Supp. Table 8; R² 0.55, RMSE 148 Mg/ha), valid up to rh98 = 33.9 m (taller labels are dropped). The sensitivity check uses AGB = 2.572 · rh98^1.5191 on the same heights. Carbon = AGB × 0.451.
- **Label tables** (`code/processing/gedi/make_labels.py`, geometry in `cells.py`, checks in `check_cells.py`), written to `data/processed/labels/`:
  - `gedi_shots.csv`: 12,800 rows, one per shot (cell, month, rh98, biomass under both equations, carbon).
  - `gedi_cells.csv`: 12,638 rows, one per 25 m cell = the training labels. Biomass is the mean over the cell's shots
    (converted shot by shot, then averaged; 162 cells have two shots). Columns also give the cell centre (UTM x, y), the
    centre pixel, the top-left pixel of its 3 x 3 window, and `mangrove_frac` (all 1.0, because the inner mask puts every
    selected cell at least 30 m from non-mangrove).
  - A label belongs to the cell, not to one pixel. The cell covers 2.5 x 2.5 pixels, so it overlaps a 3 x 3 window of 10 m
    pixels with weights equal to each pixel's share of the cell (e.g. 0.4, 0.4, 0.2 along an axis); `cells.window(i, j)`
    returns the window and weights. Pixel models use the weighted mean of the channels over the window
    (`cells.cell_mean`); the U-Nets compare the weighted mean of their predictions over the window with the label. The
    centre pixel is stored only for reference. All models are evaluated at the cell level.

## 3. Satellite data

All files: mask grid, 10 m, only inside the mangrove zone (mangrove plus a 200 m margin, built on a 100 m
grid). Everything else is nodata. Large exports are split into pieces.

### Static: `sabah_static`  (about 0.55 GB, int16, nodata −32768)

| Band | Contents | To physical units |
|---|---|---|
| `DEM_x10` | Copernicus GLO-30 (`GLO30_2024_1`) surface model, metres above EGM2008, bilinear from 30 m | divide by 10 |
| `HH_x100` | PALSAR-2 HH in dB: 10*log10(DN^2) − 83; land pixels only; median of the 2019-2023 yearly mosaics; bicubic from 25 m (DN clamped at 1 or more) | divide by 100 |
| `HV_x100` | PALSAR-2 HV, same processing | divide by 100 |

The median over years fills PALSAR gaps and covers 99.4% of mangrove pixels. Rounding to 0.1 m and 0.01 dB is
far finer than the accuracy of either source.

### Composite: `sabah_composite_2019_2023`  (about 2.2 GB, int16, nodata −32768)

One median over every clear scene in 2019-2023 (not five yearly files, and not a median of yearly medians). The
labels are effectively static, and a five-year median is cleaner and about 5x cheaper to export than five yearly files.
Each pixel's median uses about 5x more scenes (727 vs 147 in a 9 km window). The 44 km test export (`export_satellite.py testc`)
completed without a memory error (45 min queue, 78 min run) and agrees closely with the single-year 2020 composite
(correlation 0.97-0.99). The full export's runtime is uncertain (about half a day to several days). If it fails, export
yearly files (`full 2020` etc.) and take the median across years locally.

| Band | Contents | To physical units |
|---|---|---|
| `B2 B3 B4 B5 B6 B7 B8 B8A B11 B12` | Sentinel-2 surface reflectance, median over 2019-2023 of scenes with Cloud Score+ `cs_cdf` >= 0.6 (`COPERNICUS/S2_SR_HARMONIZED`); the six 20 m bands are medianed at 20 m, then bicubic to 10 m and clamped at 0 or more | divide by 10,000 |
| `VV_x100`, `VH_x100` | Sentinel-1 IW backscatter in dB, median over 2019-2023 (`COPERNICUS/S1_GRD`) | divide by 100 |

Built by `code/collecting/satellite/composites.py`; exported by `code/collecting/satellite/export_satellite.py`. A 44 km test window
passed all checks (grid, coverage, value ranges).

### Not included
Clear-observation counts, GEDI footprint vectors, and any train/validation split.

The 6 derived channels (NDVI, NDRE, NDMI, EVI, VV/VH, HV/HH) are not exported. Compute them locally with
`code/processing/satellite/derive_channels.py` (see Workflow); they use no Earth Engine quota. Slope is dropped for
now (`Project.md` 2.4: 21 channels); the script can still write `slope_deg` if it is wanted later.

## Workflow

Run from the project root, after `ee.Initialize` access to the project is set up.

```
python code/collecting/mask/export_mask.py            # writes sabah_mangrove_mask
python code/collecting/gedi/export_gedi.py
python code/collecting/satellite/export_satellite.py full static
python code/collecting/satellite/export_satellite.py full composite   # 2019-2023 median (run `testc` first)
```

Then download the files from Drive into `data/raw/<kind>/` and build the stitched and derived products:

```
python code/processing/gedi/build_gedi_vrt.py data/raw/gedi/sabah_gedi_rh98_monthly_v2.vrt data/raw/gedi/sabah_gedi_rh98_monthly_v2-*.tif
python code/processing/mask/inner_mask.py 30
python code/processing/gedi/count_labels.py          # label counts per step
python code/processing/gedi/make_labels.py           # data/processed/labels/gedi_shots.csv and gedi_cells.csv
python code/processing/gedi/label_coverage.py        # label cells per district: label_coverage.csv and the map in the notes vault
python code/processing/satellite/derive_channels.py year   data/raw/satellite/composite/sabah_composite_2019_2023*.tif
python code/processing/satellite/derive_channels.py static data/raw/satellite/static/sabah_static*.tif
```

Earth Engine does not tag the nodata value in the files, so mask it when loading: −1 for GEDI and −32768 for the static and
yearly files (the mask files have none). Drive needs
about 3 GB free per export (empty the trash too); the whole satellite set (static + composite) is about 3 GB.

## Layout

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
    mask/        sabah_mangrove_mask.tif  sabah_mangrove_inner_mask.tif (built locally from the full mask by inner_mask.py)
    gedi/        sabah_gedi_rh98_monthly_v2-*.tif (pieces) and the .vrt that stitches them
    satellite/   sabah_static*.tif, sabah_composite_2019_2023*.tif
    boundaries/  sabah_districts_gaul2025.geojson (25 Sabah districts, FAO GAUL 2025 level 2, simplified to 100 m; fetched from Earth Engine by label_coverage.py if missing)
  processed/   built locally from raw by the scripts above
    labels/      gedi_shots.csv  gedi_cells.csv  label_coverage.csv
    satellite/   <name>_derived.tif (derived channels)
```
