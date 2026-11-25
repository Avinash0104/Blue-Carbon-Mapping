"""Compute the 7 derived input channels (Project.md 2.4) locally from the exported satellite files.

Nothing is run in Earth Engine. Band names and nodata are taken from the fixed export order, so the files need no preparation.

Usage (from the Data Preparation folder):
    python code/processing/satellite/derive_channels.py year   data/raw/satellite/composite/sabah_composite_2019_2023*.tif
        -> data/processed/satellite/<name>_derived.tif, 5 bands: NDVI, NDRE, NDMI, EVI, VVVH_db   (per piece;
           works for the 2019-2023 composite or any single-year file, they have the same 12 bands)
    python code/processing/satellite/derive_channels.py static data/raw/satellite/static/sabah_static*.tif
        -> data/processed/satellite/<name>_derived.tif, 2 bands: HVHH_db, slope_deg

Output is float32, nodata -9999, same grid as the input, so it stacks with the raw files. The raw files are
never modified; derived files go to data/processed/satellite/.

Definitions (reflectance = DN / 10000):
    NDVI = (B8 - B4) / (B8 + B4)            NDRE = (B8A - B5) / (B8A + B5)
    NDMI = (B8 - B11) / (B8 + B11)          EVI  = 2.5 (B8 - B4) / (B8 + 6 B4 - 7.5 B2 + 1)
    VVVH_db = VV_dB - VH_dB                 HVHH_db = HV_dB - HH_dB   (a ratio of powers is a difference in dB)
    slope_deg = steepness of the 10 m DEM from central differences (needs the 3x3 neighbourhood)

Slope is computed per file, so a pixel on the edge of a piece has no slope (nodata). If the static
export came in several pieces, build a VRT first and pass that, so the edges are correct.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

NODATA = -9999.0
OUT_DIR = Path("data/processed/satellite")       # derived files never go next to the raw ones
S2 =["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
IN_BANDS = {"year": S2 + ["VV_x100", "VH_x100"], "static": ["DEM_x10", "HH_x100", "HV_x100"]}
IN_NODATA = {"year": -32768, "static": -32768}     # nodata of the exported files (not tagged in the files)
ROWS = 1024             # rows per block


def ratio(a, b):
    """(a - b) / (a + b), NaN where the denominator is 0."""
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, (a - b) / den, np.nan)


def year_block(src, win, names, nd):
    """5 derived bands for one window of a yearly file."""
    d = {n: src.read(i + 1, window=win).astype("float32") for i, n in enumerate(names)}
    bad = (d["B8"] == nd) | (d["VV_x100"] == nd)
    for n in ("B2", "B4", "B5", "B8A", "B11", "VH_x100"):
        bad |= d[n] == nd
    s2 = {n: d[n] / 10000 for n in ("B2", "B4", "B5", "B8", "B8A", "B11")}
    evi = 2.5 * (s2["B8"] - s2["B4"]) / (s2["B8"] + 6 * s2["B4"] - 7.5 * s2["B2"] + 1)
    out = np.stack([ratio(s2["B8"], s2["B4"]), ratio(s2["B8A"], s2["B5"]),
                    ratio(s2["B8"], s2["B11"]), evi,
                    (d["VV_x100"] - d["VH_x100"]) / 100])
    out[:, bad] = np.nan
    return out


def static_block(src, win, r0, r1, names, nd):
    """HV-HH and slope for rows r0:r1; slope reads one extra row each side so edges are exact."""
    d = {n: src.read(i + 1, window=win).astype("float32") for i, n in enumerate(names)}
    hvhh = (d["HV_x100"] - d["HH_x100"]) / 100
    hvhh[(d["HV_x100"] == nd) | (d["HH_x100"] == nd)] = np.nan

    a0, a1 = max(r0 - 1, 0), min(r1 + 1, src.height)
    dem = src.read(1, window=((a0, a1), (0, src.width))).astype("float32")
    dem[dem == nd] = np.nan
    dem /= 10                                                # DEM_x10 -> metres
    res = src.res[0]
    dz_dy = (dem[2:, :] - dem[:-2, :]) / (2 * res)           # rows -> northing is the negative direction
    dz_dx = (dem[:, 2:] - dem[:, :-2]) / (2 * res)
    # central differences need a neighbour on every side: drop the first/last column and edge rows
    slope = np.full(dem.shape, np.nan, dtype="float32")
    inner_y = dz_dy[:, 1:-1]
    inner_x = dz_dx[1:-1, :]
    slope[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(inner_x, inner_y)))
    lo = r0 - a0
    return np.stack([hvhh, slope[lo:lo + (r1 - r0)]])


if __name__ == "__main__":
    kind, files = sys.argv[1], sys.argv[2:]
    names = {"year": ["NDVI", "NDRE", "NDMI", "EVI", "VVVH_db"], "static": ["HVHH_db", "slope_deg"]}[kind]
    for path in files:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = str(OUT_DIR / (Path(path).stem + "_derived.tif"))
        with rasterio.open(path) as src:
            if src.count != len(IN_BANDS[kind]):
                sys.exit(f"{path}: has {src.count} bands, expected {len(IN_BANDS[kind])} for '{kind}'")
            profile = src.profile | {"count": len(names), "dtype": "float32", "nodata": NODATA,
                                     "tiled": True, "blockxsize": 512, "blockysize": 512,
                                     "compress": "lzw", "predictor": 3, "BIGTIFF": "IF_SAFER"}
            profile.pop("photometric", None)
            with rasterio.open(out, "w", **profile) as dst:
                for r0 in range(0, src.height, ROWS):
                    r1 = min(r0 + ROWS, src.height)
                    win = ((r0, r1), (0, src.width))
                    blk = year_block(src, win, IN_BANDS[kind], IN_NODATA[kind]) if kind == "year" else static_block(src, win, r0, r1, IN_BANDS[kind], IN_NODATA[kind])
                    dst.write(np.nan_to_num(blk, nan=NODATA).astype("float32"), window=win)
                dst.descriptions = tuple(names)
        print("wrote", out)
