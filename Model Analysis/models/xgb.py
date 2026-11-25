"""XGBoost for cell-level mangrove biomass (Project.md 4). Settings and tuning grid: models/xgb.yaml.

    python models/xgb.py main                  tune on one fold, fit 5 fold members on train + validation cells,
                                               write data/predictions/xgb.csv and the members xgb_s<seed>.csv
    python models/xgb.py spatial [--folds 0 1] spatial hold-out with the saved settings -> data/predictions_spatial/xgb.csv
Run from the Model Analysis folder, then score with evaluation/evaluate.py (see train.py for what is shared with rf.py).
Tuning stops each fit early on the tuning-fold RMSE; the best number of trees is then fixed for the final members.
"""
from pathlib import Path

from train import Spec, cli


def build(params, seed, y_train):
    import xgboost as xgb
    return xgb.XGBRegressor(random_state=seed, **params)


def score(model, Xtr, ytr, Xva, yva):
    """Fit with early stopping on the tuning fold. Returns (best tuning-fold RMSE, best number of trees)."""
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return float(model.best_score), int(model.best_iteration) + 1      # needs eval_metric: rmse and early_stopping_rounds in the yaml


def final(params, best):
    """Final members: the tuned settings, a fixed number of trees, no early stopping (no validation cells are used).
    The tree count was found on a split of the same size as each member's (4 of 5 folds), so it is used as it is."""
    params = {**params, "n_estimators": int(best["n_trees"])}
    params.pop("early_stopping_rounds", None)
    return params


SPEC = Spec("xgb", Path(__file__).with_name("xgb.yaml"), build, score, final)

if __name__ == "__main__":
    cli(SPEC, __doc__)
