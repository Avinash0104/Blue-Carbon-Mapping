"""Where are the GEDI label cells? Map + counts per district (describes the labels; decides nothing).

Usage (from the Data Preparation folder, once Earth Engine access is set up):
    python code/processing/gedi/label_coverage.py

Reads  data/processed/labels/gedi_cells.csv, data/raw/mask/sabah_mangrove_mask.tif
Writes data/raw/boundaries/sabah_districts_gaul2025.geojson   (GAUL 2025 level 2, simplified to 100 m, lon/lat;
                                                               fetched from Earth Engine only if it is missing)
       data/processed/labels/label_coverage.csv
       ../Project Context/Blue Carbon Mapping/label_coverage.png   (the notes vault, where Context.md embeds it)
"""
import json
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin

CELLS = "data/processed/labels/gedi_cells.csv"
MASK = "data/raw/mask/sabah_mangrove_mask.tif"
DIST = Path("data/raw/boundaries/sabah_districts_gaul2025.geojson")
OUT_CSV = "data/processed/labels/label_coverage.csv"
FIG = "../Project Context/Blue Carbon Mapping/label_coverage.png"
CRS = "EPSG:32650"
MAX_SNAP = 1000          # m: a label outside every polygon (GAUL is generalised) joins the nearest district within this

# single-hue sequential ramp for magnitude (light = low biomass, dark = high)
RAMP = LinearSegmentedColormap.from_list("biomass", ["#cfe0f6", "#2a78d6", "#0a3d7a"])
INK, INK2, GRID, SURFACE, BLUE = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb", "#2a78d6"


def districts():
    if not DIST.exists():
        ee.Initialize(project="carbon-mapping-mangroves")
        fc = (ee.FeatureCollection("FAO/GAUL/2025/level2")
              .filter(ee.Filter.eq("GAUL0_NAME", "Malaysia")).filter(ee.Filter.eq("GAUL1_NAME", "Sabah"))
              .select(["GAUL2_NAME"]).map(lambda f: f.simplify(100)))
        DIST.parent.mkdir(parents=True, exist_ok=True)
        DIST.write_text(json.dumps(fc.getInfo()))
    g = gpd.read_file(DIST).rename(columns={"GAUL2_NAME": "district"})
    return g[["district", "geometry"]].to_crs(CRS)


def mangrove_area_km2(dist):
    """Mangrove area per district from the 10 m mask, read at ~100 m with averaging (fraction of mangrove)."""
    with rasterio.open(MASK) as m:
        shape = (3620, 4400)
        frac = np.empty(shape, dtype="float32")
        m.read(1, out=frac, resampling=Resampling.average)
        tr = from_origin(m.transform.c, m.transform.f, m.width * 10 / shape[1], m.height * 10 / shape[0])
    cell_km2 = (m.width * 10 / shape[1]) * (m.height * 10 / shape[0]) / 1e6
    out = {}
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(dist.geometry)), out_shape=shape, transform=tr, dtype="int32")
    for i, name in enumerate(dist.district):
        out[name] = float(frac[ids == i + 1].sum() * cell_km2)
    out["(outside all districts)"] = float(frac[ids == 0].sum() * cell_km2)
    return out


if __name__ == "__main__":
    cells = pd.read_csv(CELLS)
    dist = districts()
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.x_utm, cells.y_utm), crs=CRS)
    j = gpd.sjoin(pts, dist, how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated()]            # a cell exactly on a shared border matches two districts: keep the first
    miss = j.district.isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss.values], dist, how="left", max_distance=MAX_SNAP).drop(columns="index_right")
        near = near[~near.index.duplicated()]
        j.loc[miss, "district"] = near.district.reindex(j.index[miss]).values
    print(f"cells {len(j):,}; outside every polygon {int(miss.sum())}; still unassigned {int(j.district.isna().sum())}")
    area = mangrove_area_km2(dist)
    print(f"mangrove area from the coarse read: {sum(area.values()):,.0f} km2 (the 10 m mask gives 2,679.5)")

    t = (j.groupby("district").agg(cells=("agb_main", "size"), biomass_mean=("agb_main", "mean"),
                                   biomass_median=("agb_main", "median"),
                                   p10=("agb_main", lambda s: s.quantile(.1)), p90=("agb_main", lambda s: s.quantile(.9)))
         .reset_index())
    t["mangrove_km2"] = t.district.map(area)
    t["cells_per_km2"] = t.cells / t.mangrove_km2
    t["share_of_labels_pct"] = 100 * t.cells / t.cells.sum()
    t = t.sort_values("cells", ascending=False).reset_index(drop=True)
    t.round(2).to_csv(OUT_CSV, index=False)
    print(t.round(2).to_string(index=False))
    zero = sorted(set(dist.district) - set(t.district))
    print("districts with no label cells:", zero)

    # ---- figure ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(13, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0], facecolor=SURFACE)
    dist.boundary.plot(ax=ax, color=GRID, linewidth=0.6)
    order = j.sort_values("agb_main")                      # darkest (highest) drawn last
    sc = ax.scatter(order.x_utm, order.y_utm, c=order.agb_main, cmap=RAMP, s=2.5, vmin=60, vmax=600, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.5)
    cb.set_label("Biomass, main equation (Mg/ha)", color=INK2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    ax.set_title("Each dot is one label cell (n = %s)" % f"{len(j):,}", loc="left", fontsize=11, color=INK)

    ax2 = fig.add_subplot(gs[1], facecolor=SURFACE)
    top = t.iloc[::-1]
    y = np.arange(len(top))
    ax2.barh(y, top.cells, height=0.62, color=BLUE)
    for yy, n, p in zip(y, top.cells, top.share_of_labels_pct):
        pct = "<1%" if p < 0.5 else f"{p:.0f}%"
        ax2.text(n + t.cells.max() * 0.015, yy, f"{n:,}  ({pct})", va="center", fontsize=9, color=INK)
    ax2.set_yticks(y); ax2.set_yticklabels(top.district, fontsize=9)
    ax2.set_xlim(0, t.cells.max() * 1.3)
    ax2.xaxis.set_visible(False)
    for s in ("top", "right", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["left"].set_color(GRID)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Label cells per district", loc="left", fontsize=11, color=INK)
    fig.suptitle("Where the GEDI label cells are (30 m inner mask, rh98 <= 33.9 m)", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.savefig(FIG, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", FIG)
