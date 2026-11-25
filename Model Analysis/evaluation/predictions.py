"""The predictions file every model writes, and its checks. The evaluation reads only this format.

File: Model Analysis/data/predictions/<model>.csv with columns
    cell_row, cell_col   the cell key (same as gedi_cells.csv / gedi_split.csv)
    set                  train / val / test, must equal gedi_split.csv
    pred                 CELL-LEVEL prediction in Mg/ha: RF / XGBoost = the model output for the cell's feature row;
                         U-Nets = the weighted mean of the predicted pixels over the cell's 3 x 3 window, with the
                         weights from gedi/cells.py (never the centre pixel alone)
    pred_std  (optional) ensemble spread of that cell-level prediction: the standard deviation, over the members,
                         of each member's cell-mean prediction (not the mean of pixel spreads)
Rows are matched by key, not by order. Labels and coordinates are NOT in the file: the evaluation joins them from
gedi_cells.csv, so a model can never change what it is scored against. All test cells must be present.
A sidecar <model>.json records the model name, seeds and other run settings.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_ROOT = Path(__file__).resolve().parents[1]                      # the "Model Analysis" folder
DATA_PREP = EVAL_ROOT.parent / "Data Preparation"                    # labels and split live there (read only)
LABELS = DATA_PREP / "data" / "processed" / "labels"
PRED_DIR = EVAL_ROOT / "data" / "predictions"
KEY = ["cell_row", "cell_col"]


def validate(df, split, require_test=True):
    """Raise AssertionError with a clear message if the predictions are not usable."""
    need = KEY + ["set", "pred"]
    assert all(c in df.columns for c in need), f"missing columns: {[c for c in need if c not in df.columns]}"
    assert not df.duplicated(KEY).any(), "duplicate cell keys"
    m = df.merge(split, on=KEY, how="left", suffixes=("", "_split"))
    assert m["set_split"].notna().all(), "some cells are not in the split file"
    assert (m["set"] == m["set_split"]).all(), "'set' disagrees with the split file"
    assert np.isfinite(df["pred"].to_numpy(float)).all(), "pred has NaN or infinite values"
    if "pred_std" in df.columns:
        s = df["pred_std"].to_numpy(float)
        assert np.isfinite(s).all() and (s >= 0).all(), "pred_std must be finite and >= 0"
    if require_test:
        n_test = int((split["set"] == "test").sum())
        assert int((df["set"] == "test").sum()) == n_test, f"not all {n_test} test cells are present"


def load_split():
    return pd.read_csv(LABELS / "gedi_split.csv")


def write(df, name, meta=None, pred_dir=PRED_DIR, split=None, require_test=True):
    """Validate and write <name>.csv and <name>.json. meta: dict of run settings (seed, ensemble size, ...)."""
    split = load_split() if split is None else split
    validate(df, split, require_test)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cols = KEY + ["set", "pred"] + (["pred_std"] if "pred_std" in df.columns else [])
    df[cols].to_csv(pred_dir / f"{name}.csv", index=False)
    side = {"model": name, "n_rows": len(df), "split_sha256": split_sha(), **(meta or {})}
    (pred_dir / f"{name}.json").write_text(json.dumps(side, indent=1))


def read(name, pred_dir=PRED_DIR):
    """Read a predictions file; if its sidecar records a split hash, it must match the current split file."""
    side = pred_dir / f"{name}.json"
    if side.exists():
        made_with, now = json.loads(side.read_text()).get("split_sha256"), split_sha()
        assert made_with is None or now is None or made_with == now, \
            f"{name}: predictions were made with a different split file (hash {made_with[:12]}, current {now[:12]})"
    return pd.read_csv(pred_dir / f"{name}.csv")


def with_labels(df, cells, split):
    """Predictions joined to the label `obs` (agb_main) and cell-centre x_utm / y_utm; sorted by key."""
    validate(df, split)
    m = df.merge(cells[KEY + ["agb_main", "x_utm", "y_utm"]], on=KEY, how="left", validate="1:1")
    assert m["agb_main"].notna().all(), "some predicted cells have no label"
    assert np.isfinite(m[["x_utm", "y_utm"]].to_numpy(float)).all(), "some cells have no coordinates"
    return m.rename(columns={"agb_main": "obs"}).sort_values(KEY, ignore_index=True)


def split_sha():
    """sha256 of the current split file, or None if it is not there (synthetic tests)."""
    f = LABELS / "gedi_split.csv"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
