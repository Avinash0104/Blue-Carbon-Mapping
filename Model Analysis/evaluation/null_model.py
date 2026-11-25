"""Null model: predict the mean of the TRAIN labels for every cell (Project.md 4, 5.2). Every model must beat it.

Run (from the Model Analysis folder): python evaluation/null_model.py
Writes data/predictions/null.csv (all three sets, so the same evaluation path is used as for the models).
The mean uses train cells only; validation and test labels never enter.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predictions as P


def make(cells, split):
    """Predictions DataFrame (key, set, pred) with the train-mean constant for every cell."""
    m = split.merge(cells[P.KEY + ["agb_main"]], on=P.KEY, how="left", validate="1:1")
    mean = float(m.loc[m["set"] == "train", "agb_main"].mean())
    return split[P.KEY + ["set"]].assign(pred=mean), mean


if __name__ == "__main__":
    cells = pd.read_csv(P.LABELS / "gedi_cells.csv")
    split = P.load_split()
    df, mean = make(cells, split)
    P.write(df, "null", {"kind": "null model", "train_mean_agb_main": mean, "seed": None})
    print(f"wrote null.csv: every cell predicted as the train mean, {mean:.2f} Mg/ha")
