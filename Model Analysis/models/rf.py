"""Random Forest for cell-level mangrove biomass (Project.md 4). Settings and tuning grid: models/rf.yaml.

    python models/rf.py main                  tune on one fold, fit 5 fold members on train + validation cells,
                                              write data/predictions/rf.csv and the members rf_s<seed>.csv
    python models/rf.py spatial [--folds 0 1] spatial hold-out with the saved settings -> data/predictions_spatial/rf.csv
Run from the Model Analysis folder, then score with evaluation/evaluate.py (see train.py for what is shared with xgb.py).
"""
from pathlib import Path

import numpy as np

from train import Spec, cli


def build(params, seed, y_train):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=seed, **params)


def score(model, Xtr, ytr, Xva, yva):
    """Fit for tuning. Returns (tuning-fold RMSE, number of trees)."""
    model.fit(Xtr, ytr)
    return float(np.sqrt(np.mean((model.predict(Xva) - yva) ** 2))), int(model.n_estimators)


def final(params, best):
    return params                                  # nothing to adjust: the tuned settings are used as they are


SPEC = Spec("rf", Path(__file__).with_name("rf.yaml"), build, score, final)

if __name__ == "__main__":
    cli(SPEC, __doc__)
