"""Count and describe the GEDI labels after each local selection step (Project.md 2.3 go/no-go table).

Usage (from the Data Preparation folder):
    python code/processing/gedi/count_labels.py

Reads data/raw/gedi/sabah_gedi_rh98_monthly_v2.vrt (nodata -1), the full mask and the 30 m inner mask.
A 25 m label cell is placed on the 10 m grid at its centre pixel: cell (i, j) has its centre at
(1.25 + 2.5 i, 1.25 + 2.5 j) in 10 m pixels from the top-left corner, so the pixel is the floor of that.

Steps (each count is shots = valid monthly values; cells = distinct 25 m cells with at least one value):
  1. valid values in the export (already filtered in Earth Engine, on mask value 1)
  2. centre pixel is on the full mask
  3. ... and on the inner mask (these are the training labels)
  4. ... and rh98 <= H_CAP (biomass equation limit)
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomass

VRT = "data/raw/gedi/sabah_gedi_rh98_monthly_v2.vrt"
FULL = "data/raw/mask/sabah_mangrove_mask.tif"
INNER = "data/raw/mask/sabah_mangrove_inner_mask.tif"
BLOCK = 128            # GEDI rows per block


def centre_index(n):
    """10 m pixel index of the centre of each of n 25 m cells along one axis."""
    return np.floor(1.25 + 2.5 * np.arange(n)).astype(int)


if __name__ == "__main__":
    with rasterio.open(VRT) as g, rasterio.open(FULL) as fm, rasterio.open(INNER) as im:
        assert (g.transform.c, g.transform.f) == (fm.transform.c, fm.transform.f), "grids differ"
        H, W, B = g.height, g.width, g.count
        cols = centre_index(W)
        counts = {k: 0 for k in ("valid", "full", "inner", "cap")}
        ncell = {k: 0 for k in counts}
        per_band = np.zeros(B, dtype=int)
        vals = []                                        # training labels (step 4)
        for r0 in range(0, H, BLOCK):
            r1 = min(r0 + BLOCK, H)
            a = g.read(window=((r0, r1), (0, W)))          # (B, rows, W)
            valid = a != -1
            if not valid.any():
                continue
            rows = centre_index(H)[r0:r1]
            m0, m1 = rows.min(), rows.max() + 1
            f = fm.read(1, window=((m0, m1), (0, fm.width)))[rows - m0][:, cols] == 1
            i = im.read(1, window=((m0, m1), (0, im.width)))[rows - m0][:, cols] == 1
            s_full = valid & f[None]
            s_inner = s_full & i[None]
            s_cap = s_inner & (a <= biomass.H_CAP) & (a >= 0)
            for k, s in (("valid", valid), ("full", s_full), ("inner", s_inner), ("cap", s_cap)):
                counts[k] += int(s.sum())
                ncell[k] += int(s.any(0).sum())
            per_band += s_cap.sum((1, 2))
            vals.append(a[s_cap])
        v = np.concatenate(vals)

    print("step                                         shots     cells")
    for k, name in (("valid", "valid in export (EE filters, mask=1)"), ("full", "centre on full mask"),
                    ("inner", "centre on inner mask (30 m)"), ("cap", "and rh98 <= %.1f m" % biomass.H_CAP)):
        print(f"{name:42s} {counts[k]:9,d} {ncell[k]:9,d}")
    print(f"\nmonths with labels: {(per_band > 0).sum()} of {B}; per month min/median/max: "
          f"{per_band.min()}/{int(np.median(per_band))}/{per_band.max()}")
    print("rh98 (m) of training labels: min %.1f  p5 %.1f  median %.1f  p95 %.1f  max %.1f" %
          ((v.min(),) + tuple(np.percentile(v, [5, 50, 95])) + (v.max(),)))
    for name, f in (("main (SE Asia)", biomass.agb_main), ("sensitivity (global Hmax)", biomass.agb_sensitivity)):
        agb = f(v)
        print(f"AGB {name}: median {np.median(agb):.0f}  mean {agb.mean():.0f}  min {agb.min():.0f}  max {agb.max():.0f} Mg/ha"
              f"  (carbon mean {biomass.carbon(agb).mean():.0f} Mg C/ha)")
