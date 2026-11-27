"""Shared pieces of the RF and XGBoost scripts (rf.py, xgb.py): data, tuning, ensemble, and the two run modes.

Both models see exactly the same inputs: the 21 channels of each label cell (weighted mean over its 3 x 3 window), z-scored
with the TRAIN-only statistics, missing values set to the training mean (normalise.py). No lat/lon. Target: agb_main (Mg/ha).

Each model is an ENSEMBLE of 5 members (cv_folds = number of seeds in its yaml), the same recipe as the U-Nets:
    pred     = mean over the members          pred_std = standard deviation over the members
Two phases, both on the pooled TRAIN + VALIDATION cells (85%; the test cells are never in either cut):
  1. TUNING (no ensemble, once). The pool is cut into 5 random folds (data/cv/tuning_folds.csv, write-once, seed = yaml seed).
     Every grid combination is fitted on the 4 folds other than the tuning fold (68% of all cells) and scored on the tuning
     fold (17%, default fold 0). The winning settings are saved and fixed.
  2. PERFORMANCE (the ensemble). The same pool is shuffled again and cut into 5 NEW random folds (data/cv/cv_folds.csv,
     write-once, seed = yaml seed + 1; the U-Nets use this file too). Member k (seed seeds[k]) is fitted once, with the fixed
     settings, on the 4 folds outside fold k (68%) and predicts fold k. These out-of-fold predictions (the cross-validation
     score) are written to <model>_oof.csv.
The mean over the members is NOT out-of-fold for pooled cells (4 of the 5 members saw each one). The cross-validation score is
averaged over all pooled cells, including the ones that chose the settings in phase 1 (now spread over all new folds), so it
is slightly optimistic; only the test cells are fully clean.
The members are also written one by one (<model>_s<seed>.csv) for evaluation/seed_variance.py.

Run times (wall clock, seconds) go to data/results/timing_<model>_<mode>.json. tuning = the whole grid search (main mode only);
fit = building and fitting a member; predict = that member predicting the cells reported. Library import, out-of-fold
predictions and file writing are not counted. Each run replaces the file of its mode, and its totals cover that run only
(see folds_run in spatial mode). Times depend on the machine and its load: run one model at a time when they are compared.

What a model file (rf.py / xgb.py) provides is a Spec: its name, its yaml, and three small functions:
    build(params, seed, y_train)            an unfitted model
    score(model, Xtr, ytr, Xva, yva)        fit it for tuning (Xva = the tuning fold); return (tuning-fold RMSE, number of trees)
    final(params, best)                     the settings of the final members, from the saved tuning winner
"""
import argparse
import itertools
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]                      # the "Model Analysis" folder
DATA_PREP = ROOT.parent / "Data Preparation"
sys.path[:0] = [str(ROOT / "evaluation"), str(DATA_PREP / "code" / "processing" / "features")]
import predictions as P                                          # noqa: E402  (evaluation/)
import spatial as S                                              # noqa: E402  (evaluation/)
from channels import CHANNELS                                    # noqa: E402  (Data Preparation)
from normalise import normalise                                  # noqa: E402  (Data Preparation)

TABLE = DATA_PREP / "data" / "processed" / "features" / "cell_features.csv"
STATS = DATA_PREP / "data" / "processed" / "features" / "norm_stats.json"
TUNING = ROOT / "data" / "tuning"
TIMING = ROOT / "data" / "results"
CV_FILE = ROOT / "data" / "cv" / "cv_folds.csv"                 # phase 2: the folds of the ensemble members
TUNING_CUT = ROOT / "data" / "cv" / "tuning_folds.csv"         # phase 1: the folds used for tuning only
TARGET = "agb_main"


@dataclass
class Spec:
    name: str
    config: Path
    build: Callable
    score: Callable
    final: Callable


# ------------------------------------------------------------------ data
def load_table():
    """The cell table and its normalised inputs (train-only statistics from norm_stats.json, missing = training mean)."""
    for f in (TABLE, STATS):
        if not f.exists():
            raise SystemExit(f"{f} not found: run Data Preparation/code/processing/features/build_cell_features.py first")
    df = pd.read_csv(TABLE)
    missing = [c for c in P.KEY + ["set", TARGET] + CHANNELS if c not in df.columns]
    assert not missing, f"cell_features.csv lacks columns: {missing}"
    assert np.isfinite(df[TARGET].to_numpy(float)).all(), "the target has NaN or infinite values"
    X = normalise(df, STATS)
    assert list(X.columns) == CHANNELS, "the statistics file and channels.py disagree on the channels"
    return df, X


def fold_inputs(df, train_mask, fold):
    """Normalised inputs for one spatial fold: mean/std from that fold's train cells only (never test or buffer cells)."""
    tr = df.loc[train_mask, CHANNELS]
    mean, std = tr.mean(), tr.std(ddof=0)
    assert (std > 0).all(), "a channel has no spread in the train cells of this fold"
    path = ROOT / "data" / "spatial" / f"norm_stats_fold{fold}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"channels": CHANNELS, "note": f"train cells of spatial fold {fold} only",
                                "stats": {c: {"mean": float(mean[c]), "std": float(std[c])} for c in CHANNELS}}, indent=1))
    return normalise(df, path)


# ------------------------------------------------------------------ folds and ensemble
def cut_folds(n, k, seed):
    """Fold id (0..k-1) for each of n rows: a seeded shuffle, dealt out in turn, so the folds differ in size by at most 1."""
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, dtype=int)
    fold[perm] = np.arange(n) % k
    return fold


def cv_folds(df, k, seed=42, path=CV_FILE):
    """Fold id of every row of df (-1 for test cells). The folds are cut from the TRAIN + VALIDATION cells (never the test
    cells) and written once to data/cv/cv_folds.csv; later runs read that file, which must still match those cells exactly."""
    train = (df["set"] != "test").to_numpy()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        f = df.loc[train, P.KEY].reset_index(drop=True)
        f["cv_fold"] = cut_folds(len(f), k, seed)
        f.to_csv(path, index=False)
        print(f"wrote {path} ({len(f)} train + validation cells in {k} folds, seed {seed}); it is never overwritten")
    f = pd.read_csv(path)
    assert not f.duplicated(P.KEY).any(), f"{path.name} has duplicate cells"
    assert f["cv_fold"].between(0, k - 1).all() and f["cv_fold"].nunique() == k, \
        f"{path.name} does not hold {k} folds: did cv_folds in the yaml change? Delete the file only if you mean to re-cut"
    m = df[P.KEY + ["set"]].merge(f, on=P.KEY, how="outer", indicator=True, validate="1:1")
    ok = (m["_merge"] == "both") & (m["set"] != "test")
    assert ok.sum() == train.sum() == len(f), f"{path.name} does not match the current train + validation cells"
    return df[P.KEY].merge(f, on=P.KEY, how="left")["cv_fold"].fillna(-1).astype(int).to_numpy()


def ensemble(spec, params, seeds, Xtr, ytr, Xpred, fold, times=None):
    """Member k (seed seeds[k]) is fitted once on the rows NOT in fold k. fold: fold id of each row of Xtr (0..len(seeds)-1).
    Returns (members, oof): members = (members x cells) predictions for Xpred; oof = each row of Xtr predicted by the
    one member that did not see it. times: optional list; one dict of run times per member is appended to it."""
    fold = np.asarray(fold)
    assert len(fold) == len(ytr) and sorted(set(fold.tolist())) == list(range(len(seeds))), \
        "the folds must be 0..len(seeds)-1, one per member"
    out, oof = [], np.full(len(ytr), np.nan)
    for k, s in enumerate(seeds):
        keep = fold != k
        t0 = time.perf_counter()
        m = spec.build(params, s, ytr[keep])
        m.fit(Xtr[keep], ytr[keep])
        t1 = time.perf_counter()
        out.append(m.predict(Xpred))
        t2 = time.perf_counter()
        oof[~keep] = m.predict(Xtr[~keep])
        if times is not None:
            times.append({"member": k, "seed": int(s), "fit_cells": int(keep.sum()), "fit_s": t1 - t0,
                          "predict_cells": len(Xpred), "predict_s": t2 - t1})
    return np.vstack(out), oof


def combine(members):
    """(mean, standard deviation) over the members."""
    return members.mean(axis=0), members.std(axis=0)


def versions():
    out = {}
    for mod in ("sklearn", "xgboost", "numpy", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not installed"
    return out


# ------------------------------------------------------------------ run times
def totals(times):
    """Summed fit and predict seconds of the members (prediction of the ensemble = all members predicting)."""
    return {"fit_s": sum(t["fit_s"] for t in times), "predict_s": sum(t["predict_s"] for t in times)}


def save_timing(name, mode, params, record):
    """Write data/results/timing_<name>_<mode>.json: the record plus the machine it was measured on."""
    TIMING.mkdir(parents=True, exist_ok=True)
    machine = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
               "n_jobs": params.get("n_jobs")}
    path = TIMING / f"timing_{name}_{mode}.json"
    path.write_text(json.dumps({"model": name, "mode": mode, "written": datetime.now().isoformat(timespec="seconds"),
                                "unit": "seconds, wall clock", "machine": machine, "versions": versions(), **record}, indent=1))
    print(f"{name}: wrote {path}")


# ------------------------------------------------------------------ tuning
def tune(spec, cfg, Xn, y, tr, va):
    """Fit every grid combination on the tr cells (4 folds), score on the va cells (the tuning fold). Returns (log DataFrame, best dict)."""
    keys = list(cfg["grid"])
    combos = list(itertools.product(*(cfg["grid"][k] for k in keys)))
    print(f"{spec.name}: {len(combos)} combinations to try: fit on {int(tr.sum())} cells, scored on the {int(va.sum())} cells of the tuning fold")
    rows = []
    for i, vals in enumerate(combos, 1):
        combo = dict(zip(keys, vals))
        model = spec.build({**cfg.get("fixed", {}), **combo}, cfg["seed"], y[tr])
        rmse, n_trees = spec.score(model, Xn[tr], y[tr], Xn[va], y[va])
        rows.append({**combo, "n_trees": n_trees, "tuning_rmse": rmse})
        print(f"  [{i}/{len(combos)}] {combo} -> tuning-fold RMSE {rmse:.2f} Mg/ha")
    top = rows[int(np.argmin([r["tuning_rmse"] for r in rows]))]          # the original dict: ints stay ints (a pandas row gives floats)
    best = {"model": spec.name, "params": {k: top[k] for k in keys}, "n_trees": int(top["n_trees"]), "tuning_rmse": float(top["tuning_rmse"]),
            "seed": cfg["seed"], "grid": cfg["grid"], "fixed": cfg.get("fixed", {}), "versions": versions()}
    return pd.DataFrame(rows).sort_values("tuning_rmse", ignore_index=True), best


def load_best(spec):
    f = TUNING / f"{spec.name}_best.json"
    if not f.exists():
        raise SystemExit(f"{f} not found: run 'python models/{spec.name}.py main' first")
    return json.loads(f.read_text())


def final_params(spec, best):
    """Settings of the final members, from what was SAVED at tuning time (so editing the yaml later cannot change them silently)."""
    return spec.final({**best["fixed"], **best["params"]}, best)


def cv_size(cfg):
    """Number of folds = number of members. The yaml must say both, and they must agree."""
    k = int(cfg["cv_folds"])
    assert k == len(cfg["seeds"]) and k >= 2, f"cv_folds ({k}) must equal the number of seeds ({len(cfg['seeds'])}) and be at least 2"
    return k


# ------------------------------------------------------------------ run modes
def run_main(spec, cfg, df, X):
    """Tune once (fit on 4 folds, score on the tuning fold), then fit the 5 fold members once each on the pooled train + validation cells."""
    Xn, y, seeds, k = X.to_numpy(), df[TARGET].to_numpy(float), [int(s) for s in cfg["seeds"]], cv_size(cfg)
    pool = (df["set"] != "test").to_numpy()
    tuning_cut = cv_folds(df, k, cfg["seed"], TUNING_CUT)             # phase 1 cut (tuning only)
    folds = cv_folds(df, k, cfg["seed"] + 1, CV_FILE)                # phase 2 cut: a fresh shuffle of the same pool
    assert (tuning_cut[pool] == folds[pool]).mean() < 0.9, "the tuning cut and the performance cut must be different shuffles"
    tf = int(cfg.get("tuning_fold", 0))
    assert 0 <= tf < k, f"tuning_fold ({tf}) must be one of the folds 0..{k - 1}"
    tr, va = pool & (tuning_cut != tf), tuning_cut == tf             # phase 1: fit on 4 folds, score on the tuning fold
    spec.build(cfg.get("fixed", {}), cfg["seed"], y[tr])               # untimed: loads the model library before the timers
    t0 = time.perf_counter()
    log, best = tune(spec, cfg, Xn, y, tr, va)
    tuning_s = time.perf_counter() - t0
    best["tuning_fold"] = tf
    TUNING.mkdir(parents=True, exist_ok=True)
    log.to_csv(TUNING / f"{spec.name}_tuning.csv", index=False)
    (TUNING / f"{spec.name}_best.json").write_text(json.dumps(best, indent=1))
    print(f"{spec.name}: best {best['params']} (tuning-fold RMSE {best['tuning_rmse']:.2f}); fitting {k} members once each, on {k - 1} of the {k} folds of the {int(pool.sum())} train + validation cells")

    times, params = [], final_params(spec, best)
    members, oof = ensemble(spec, params, seeds, Xn[pool], y[pool], Xn, folds[pool], times)
    pred, sd = combine(members)
    tot = totals(times)
    print(f"{spec.name}: tuning {tuning_s:.1f} s; {k} members: fit {tot['fit_s']:.1f} s, predict {len(Xn)} cells {tot['predict_s']:.2f} s")
    base = df[P.KEY + ["set"]]
    P.write(base.assign(pred=pred, pred_std=sd), spec.name,
            {"seeds": seeds, "params": best["params"], "n_trees": best["n_trees"], "tuning_fold_rmse": best["tuning_rmse"], "tuning_fold": tf,
             "versions": versions(), "trained_on": f"train + validation cells; member k without fold k ({k} random folds, data/cv/cv_folds.csv)", "inputs": CHANNELS,
             "pred": "mean of the members", "pred_std": "sd over the members"})
    P.write(base[pool].assign(pred=oof), f"{spec.name}_oof", {"base_model": spec.name, "note": "out-of-fold: each train / validation "
            "cell predicted by the one member that did not see it; test cells not included"}, require_test=False)
    for s, member in zip(seeds, members):                                # the members, for evaluation/seed_variance.py
        P.write(base.assign(pred=member), f"{spec.name}_s{s}", {"seed": s, "base_model": spec.name, "params": best["params"]})
    save_timing(spec.name, "main", params, {"tuning_s": tuning_s, "n_combinations": len(log), **tot, "members": times})
    fp, rm = folds[pool], lambda m: float(np.sqrt(np.mean((oof[m] - y[pool][m]) ** 2)))
    print(f"{spec.name}: wrote data/predictions/{spec.name}.csv, {spec.name}_oof.csv and {len(seeds)} member files. "
          f"Cross-validation RMSE {rm(np.ones(len(fp), bool)):.1f} Mg/ha (per member {[round(rm(fp == j), 1) for j in range(k)]}; slightly "
          f"optimistic, the tuning cells are in the pool). The test cells are scored by evaluation/evaluate.py")


def run_spatial(spec, cfg, df, folds_to_run):
    """Spatial hold-out: per fold, an ensemble with the saved settings on that fold's train-role cells only. No re-tuning."""
    best, folds = load_best(spec), S.load_folds()
    seeds, k = [int(s) for s in cfg["seeds"]], cv_size(cfg)
    d = df[P.KEY + [TARGET] + CHANNELS].merge(folds, on=P.KEY, how="left", validate="1:1")
    assert d["fold"].notna().all(), "some cells are not in the spatial folds file"
    y, parts, per_fold, params = d[TARGET].to_numpy(float), [], [], final_params(spec, best)
    spec.build(params, seeds[0], y)                                    # untimed: loads the model library before the timers
    for f in folds_to_run:
        tr, te = (d[f"role_{f}"] == "train").to_numpy(), (d["fold"] == f).to_numpy()
        Xn = fold_inputs(d, tr, f).to_numpy()
        sub = cut_folds(int(tr.sum()), k, cfg["seed"] + f)               # this fold's train-role cells only, cut like the main run
        times = []
        pred, sd = combine(ensemble(spec, params, seeds, Xn[tr], y[tr], Xn[te], sub, times)[0])
        per_fold.append({"fold": int(f), **totals(times), "members": times})
        parts.append(d.loc[te, P.KEY + ["fold"]].assign(pred=pred, pred_std=sd))
        print(f"{spec.name}: fold {f}: {len(seeds)} members, each on {k - 1}/{k} of {int(tr.sum())} cells, predicted {int(te.sum())}; "
              f"RMSE {np.sqrt(np.mean((pred - y[te]) ** 2)):.1f} Mg/ha")
    S.write(pd.concat(parts, ignore_index=True), spec.name,
            {"seeds": seeds, "params": best["params"], "folds": list(folds_to_run), "versions": versions(),
             "note": "settings from the main run, no re-tuning; members fitted on random sub-folds of the fold's train-role cells"})
    print(f"{spec.name}: wrote data/predictions_spatial/{spec.name}.csv")
    save_timing(spec.name, "spatial", params, {"folds_run": [int(f) for f in folds_to_run],
                                              **totals([t for p in per_fold for t in p["members"]]), "folds": per_fold})


def cli(spec, doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["main", "spatial"])
    ap.add_argument("--config", default=str(spec.config), help="yaml with the settings (default: the one next to this script)")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="spatial mode: folds to run (default: all)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    df, X = load_table()
    if a.mode == "main":
        run_main(spec, cfg, df, X)
    else:
        run_spatial(spec, cfg, df, a.folds if a.folds else list(range(int(S.load_folds().fold.nunique()))))
