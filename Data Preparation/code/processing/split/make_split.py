"""Random 70/15/15 split of the GEDI label cells (train / validation / test), shared by all four models.

Decided 2026-11-23: a simple random split of cells, as in Dong et al. (2023). Fixed seed; the file is written
once and never changed. Writes data/processed/labels/gedi_split.csv (cell_row, cell_col, set).
Run: python code/processing/split/make_split.py   (refuses to overwrite an existing split file)
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
CELLS = ROOT / "data" / "processed" / "labels" / "gedi_cells.csv"
OUT = ROOT / "data" / "processed" / "labels" / "gedi_split.csv"
SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

if __name__ == "__main__":
    if OUT.exists():
        sys.exit(f"{OUT} already exists: the split is fixed once written. Delete it only on purpose.")
    c = pd.read_csv(CELLS)
    assert not c.duplicated(["cell_row", "cell_col"]).any(), "duplicate cell in gedi_cells.csv"
    n = len(c)
    order = np.random.default_rng(SEED).permutation(n)
    n_val, n_test = round(FRACTIONS["val"] * n), round(FRACTIONS["test"] * n)
    label = np.empty(n, dtype=object)
    label[order[:n_test]] = "test"
    label[order[n_test:n_test + n_val]] = "val"
    label[order[n_test + n_val:]] = "train"
    out = c[["cell_row", "cell_col"]].assign(set=label)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} (seed {SEED}), sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")

    xy = c[["x_utm", "y_utm"]].values
    tr = label == "train"
    d, _ = cKDTree(xy[tr]).query(xy[~tr])
    for s in ("train", "val", "test"):
        m = label == s
        print(f"{s:5s} {m.sum():5d} cells ({100*m.mean():.1f}%)  agb_main mean {c.agb_main[m].mean():6.1f}  "
              f"median {c.agb_main[m].median():6.1f}  p10-p90 {c.agb_main[m].quantile(.1):.0f}-{c.agb_main[m].quantile(.9):.0f}")
    print(f"leakage check: val+test cells with a train cell within 100 m: {100*(d<100).mean():.0f}%, "
          f"within 1 km: {100*(d<1000).mean():.0f}%, median distance {np.median(d):.0f} m")
