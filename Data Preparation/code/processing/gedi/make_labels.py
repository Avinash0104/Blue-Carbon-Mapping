"""Build the GEDI label tables from the v2 export (Project.md 2.3).

Usage (from the Data Preparation folder):
    python code/processing/gedi/make_labels.py
Writes:
    data/processed/labels/gedi_shots.csv   one row per shot (one cell in one month)
    data/processed/labels/gedi_cells.csv   one row per 25 m cell: the training labels (mean biomass over its shots)

Selection (same as count_labels.py): centre pixel on the inner (30 m) mask, 0 <= rh98 <= biomass.H_CAP.
Biomass is computed shot by shot with the main equation (and the sensitivity equation for the carbon-total
check), then averaged over the months of a cell: the equation is not linear, so the mean of biomass is not the
biomass of the mean height. See cells.py for the 3 x 3 pixel window and weights of each cell.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomass
import cells

VRT = "data/raw/gedi/sabah_gedi_rh98_monthly_v2.vrt"
FULL = "data/raw/mask/sabah_mangrove_mask.tif"
INNER = "data/raw/mask/sabah_mangrove_inner_mask.tif"
OUT = Path("data/processed/labels")
BLOCK = 128


def month_names(g):
    return [d.replace("rh98_", "").replace("_", "-") for d in g.descriptions]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with rasterio.open(VRT) as g, rasterio.open(FULL) as fm, rasterio.open(INNER) as im:
        months = month_names(g)
        H, W = g.height, g.width
        assert (H, W) == cells.GEDI_SHAPE and (fm.height, fm.width) == cells.PIX_SHAPE
        cols = np.floor(1.25 + 2.5 * np.arange(W)).astype(int)
        for r0 in range(0, H, BLOCK):
            r1 = min(r0 + BLOCK, H)
            a = g.read(window=((r0, r1), (0, W)))
            valid = a != -1
            if not valid.any():
                continue
            rr = np.floor(1.25 + 2.5 * np.arange(r0, r1)).astype(int)
            m0, m1 = rr.min(), rr.max() + 1
            inner = im.read(1, window=((m0, m1), (0, im.width)))[rr - m0][:, cols] == 1
            sel = valid & inner[None] & (a >= 0) & (a <= biomass.H_CAP)
            b, i, j = np.nonzero(sel)
            rows.append(pd.DataFrame({"cell_row": i + r0, "cell_col": j, "month": np.array(months)[b],
                                      "rh98": a[b, i, j].astype("float64")}))
        shots = pd.concat(rows, ignore_index=True).sort_values(["cell_row", "cell_col", "month"], ignore_index=True)
        shots["agb_main"] = biomass.agb_main(shots["rh98"])
        shots["agb_sens"] = biomass.agb_sensitivity(shots["rh98"])
        shots["carbon_main"] = biomass.carbon(shots["agb_main"])
        assert shots[["agb_main", "agb_sens"]].notna().all().all()

        cell = (shots.groupby(["cell_row", "cell_col"])
                .agg(n_shots=("rh98", "size"), rh98_mean=("rh98", "mean"),
                     agb_main=("agb_main", "mean"), agb_sens=("agb_sens", "mean"))
                .reset_index())
        cell["carbon_main"] = biomass.carbon(cell["agb_main"])
        # position on the 10 m grid
        xy = np.array([cells.centre_xy(i, j) for i, j in zip(cell.cell_row, cell.cell_col)])
        cell["x_utm"], cell["y_utm"] = xy[:, 0], xy[:, 1]
        cp = np.array([cells.centre_pixel(i, j) for i, j in zip(cell.cell_row, cell.cell_col)])
        cell["centre_row"], cell["centre_col"] = cp[:, 0], cp[:, 1]
        win = [cells.window(i, j) for i, j in zip(cell.cell_row, cell.cell_col)]
        cell["win_row0"] = [w[0] for w in win]
        cell["win_col0"] = [w[1] for w in win]
        # share of the cell's area that lies on full-mask mangrove (QA; all pixels are kept in the average)
        frac = np.zeros(len(cell))
        for k, (r0, c0, w) in enumerate(win):
            frac[k] = ((fm.read(1, window=((r0, r0 + 3), (c0, c0 + 3))) == 1) * w).sum()
        cell["mangrove_frac"] = frac

    shots.to_csv(OUT / "gedi_shots.csv", index=False)
    cell.to_csv(OUT / "gedi_cells.csv", index=False)
    print(f"shots {len(shots):,}  cells {len(cell):,}  (cells with >1 shot: {(cell.n_shots > 1).sum()})")
    print("mangrove_frac of the cell window: min %.2f  p5 %.2f  median %.2f" %
          (cell.mangrove_frac.min(), cell.mangrove_frac.quantile(.05), cell.mangrove_frac.median()))
    print("cell agb_main: median %.0f mean %.0f | agb_sens mean %.0f | carbon_main mean %.0f Mg C/ha" %
          (cell.agb_main.median(), cell.agb_main.mean(), cell.agb_sens.mean(), cell.carbon_main.mean()))
    print("wrote", OUT / "gedi_shots.csv", "and", OUT / "gedi_cells.csv")
