"""Counting diagnostics that decide two evaluation settings (run once, before any model result exists).

1. Block size for the bootstrap: how many blocks do the TEST cells form for each side length, and how many cells per block?
2. Aggregated accuracy (0.25-1 ha): how many G x G squares hold at least k test cells?
Run (from the Model Analysis folder): python evaluation/diagnose.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocks
import predictions as P

sys.path.insert(0, str(P.DATA_PREP / "code" / "processing" / "gedi"))      # cells.py: the 10 m grid origin
import cells as C

if __name__ == "__main__":
    cells = pd.read_csv(P.LABELS / "gedi_cells.csv")
    split = P.load_split()
    t = cells.merge(split, on=P.KEY)
    t = t[t["set"] == "test"]
    print(f"{len(t)} test cells\n\nBootstrap blocks (side S, test cells only)")
    print("S (m)   blocks   cells per block: median  p90  max   blocks with >=5 cells")
    for s in [500, 1000, 2000, 4000, 6000, 10000]:
        code = blocks.block_codes(t.x_utm, t.y_utm, s)
        n = np.bincount(code)
        print(f"{s:6d} {len(n):8d}   {np.median(n):22.0f} {np.percentile(n, 90):4.0f} {n.max():5d}   {(n >= 5).sum():10d}")

    print("\nGroups of test cells in G x G squares (grid anchored at the 10 m grid origin)")
    print("G (m)  area (ha)  squares with >=1   >=2   >=3   >=4   >=8")
    for g in [50, 75, 100]:
        key = (np.floor((t.x_utm - C.X0) / g).astype(int).astype(str) + "_" + np.floor((C.Y0 - t.y_utm) / g).astype(int).astype(str))
        n = key.value_counts().to_numpy()
        print(f"{g:5d} {g * g / 1e4:9.2f} " + " ".join(f"{(n >= k).sum():6d}" for k in [1, 2, 3, 4, 8]))
