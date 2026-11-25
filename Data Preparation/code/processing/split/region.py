"""District of every label cell (for district summaries and the variogram script).

Districts: data/raw/boundaries/sabah_districts_gaul2025.geojson
(GAUL 2025 level 2, simplified to 100 m). Points outside every polygon go to the nearest one, measured in metres.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CELLS = ROOT / "data" / "processed" / "labels" / "gedi_cells.csv"
DISTRICTS = ROOT / "data" / "raw" / "boundaries" / "sabah_districts_gaul2025.geojson"


def load_cells():
    """gedi_cells.csv plus a 'district' column."""
    c = pd.read_csv(CELLS)
    d = gpd.read_file(DISTRICTS).to_crs(32650)[["GAUL2_NAME", "geometry"]]
    g = gpd.GeoDataFrame(c, geometry=gpd.points_from_xy(c.x_utm, c.y_utm), crs=32650)
    j = gpd.sjoin(g, d, how="left", predicate="within")
    j = j[~j.index.duplicated()]                                   # a point on a shared border matches twice
    miss = j.GAUL2_NAME.isna()
    if miss.any():
        n = gpd.sjoin_nearest(g[miss], d, how="left")
        n = n[~n.index.duplicated()]
        j.loc[miss, "GAUL2_NAME"] = n.GAUL2_NAME
    c["district"] = j.GAUL2_NAME.values
    assert not c.district.isna().any(), "a label cell has no district"
    return c
