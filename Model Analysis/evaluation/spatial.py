"""Spatial hold-out check: do the scores, and the ranking of the models, survive when the test cells are in an area the
model has not seen? (Project.md 3, 5.1; Ploton et al., 2020)

Why: the main split is random, so most test cells have a training cell within 100 m (the share is printed by evaluate.py) and the scores are optimistic. This
module is an EXTRA evaluation. It does not touch gedi_split.csv and does not replace the headline numbers.

Design (fixed once, written to data/spatial/spatial_folds.csv, never overwritten):
  * cells are grouped into 10 km blocks; whole blocks are dealt to 5 folds, balanced by cell count (seed 42);
  * for fold f every cell has a role: test (in fold f), buffer (not in fold f but within 5 km of a test cell: the
    variogram range is 3.4-4.5 km, so these cells are too alike to train on) or train (everything else).
  Every cell is tested exactly once, by a model that never saw its neighbourhood.

What the models must do (RF / XGBoost; the U-Nets may do one fold only, see below):
  for each fold f: train on cells with role_f == 'train', predict the cells with role_f == 'test', with the SAME settings
  as the main run (no re-tuning on the held-out fold; if early stopping needs a validation set, take it from the train
  cells of that fold, by block). Write data/predictions_spatial/<model>.csv with cell_row, cell_col, fold, pred
  [, pred_std] through spatial.write(). A model that is run for only some folds is scored on the folds ALL models cover
  (a fold counts only if every one of its cells is present), so a U-Net run on fold 0 is compared on fold 0.
  Normalisation statistics for a fold come from that fold's train cells only (never from test or buffer cells).

Run (from the Model Analysis folder):
    python evaluation/spatial.py make                              # once
    python evaluation/spatial.py null                              # null-model predictions per fold
    python evaluation/spatial.py evaluate --models null rf xgb --pairs rf:null
Writes data/results/results_spatial_*.csv and spatial_settings.json. What it shows:
  metrics (with block-bootstrap intervals, 10 km blocks), per-fold scores (how much regions differ), paired differences,
  and 'vs random': the same cells scored by the random-split model and the spatial model, with the ranking under each.
  The RMSE change is an UPPER BOUND on the optimism of the random split: the spatial model also trained on fewer cells
  (the train-role count per fold is printed), so part of the change is the smaller training set.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocks as B
import evaluate as E
import metrics as M
import predictions as P

FOLDS_FILE = P.EVAL_ROOT / "data" / "spatial" / "spatial_folds.csv"
PRED_DIR = P.EVAL_ROOT / "data" / "predictions_spatial"
OUT = P.EVAL_ROOT / "data" / "results"
DEFAULTS = {"k": 5, "block_m": 10000, "buffer_m": 5000, "seed": 42}
KEY = P.KEY


# ---------------------------------------------------------------- the folds
def assign_folds(x, y, k, block_m, seed):
    """Fold of each cell: whole blocks of side block_m dealt to k folds, biggest blocks first, to the lightest fold."""
    codes = B.block_codes(x, y, block_m)
    counts = np.bincount(codes)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(counts)), -counts))          # by size (largest first), random tie-break
    load, fold_of_block = np.zeros(k), np.empty(len(counts), int)
    for b in order:
        f = int(np.argmin(load))
        fold_of_block[b] = f
        load[f] += counts[b]
    return fold_of_block[codes], codes


def roles(x, y, fold, k, buffer_m):
    """role_0..role_{k-1}: 'test', 'buffer' or 'train' for every cell under each fold."""
    xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    out = {}
    for f in range(k):
        test = fold == f
        assert test.any(), f"fold {f} has no cells"
        d, _ = cKDTree(xy[test]).query(xy)
        out[f"role_{f}"] = np.where(test, "test", np.where(d < buffer_m, "buffer", "train"))
    return pd.DataFrame(out)


def make_folds(cells, k=5, block_m=10000, buffer_m=5000, seed=42):
    c = cells.sort_values(KEY, ignore_index=True)
    assert np.isfinite(c[["x_utm", "y_utm"]].to_numpy(float)).all(), "NaN coordinates"
    fold, codes = assign_folds(c.x_utm.to_numpy(), c.y_utm.to_numpy(), k, block_m, seed)
    return pd.concat([c[KEY].assign(fold=fold, block=codes), roles(c.x_utm, c.y_utm, fold, k, buffer_m)], axis=1)


def fold_summary(cells, folds):
    d = folds.merge(cells[KEY + ["agb_main"]], on=KEY)
    rows = []
    for f in sorted(d.fold.unique()):
        r = d[f"role_{f}"]
        rows.append({"fold": f, "n_test": int((r == "test").sum()), "n_train": int((r == "train").sum()),
                     "n_buffer": int((r == "buffer").sum()), "n_blocks": int(d[d.fold == f].block.nunique()),
                     "mean_obs_test": d.loc[d.fold == f, "agb_main"].mean()})
    return pd.DataFrame(rows)


def folds_sha(path=FOLDS_FILE):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def load_folds(path=FOLDS_FILE):
    if not Path(path).exists():
        raise SystemExit(f"{path} not found: run 'python evaluation/spatial.py make' first")
    return pd.read_csv(path)


# ---------------------------------------------------------------- predictions
def validate(df, folds):
    need = KEY + ["fold", "pred"]
    assert all(c in df.columns for c in need), f"missing columns: {[c for c in need if c not in df.columns]}"
    assert not df.duplicated(KEY).any(), "duplicate cell keys"
    m = df.merge(folds[KEY + ["fold"]], on=KEY, how="left", suffixes=("", "_folds"))
    assert m["fold_folds"].notna().all(), "some cells are not in the folds file"
    assert (m["fold"] == m["fold_folds"]).all(), "'fold' disagrees with the folds file"
    assert np.isfinite(df["pred"].to_numpy(float)).all(), "pred has NaN or infinite values"
    if "pred_std" in df.columns:
        s = df["pred_std"].to_numpy(float)
        assert np.isfinite(s).all() and (s >= 0).all(), "pred_std must be finite and >= 0"


def complete_folds(df, folds):
    """Folds whose every cell has a prediction."""
    have, want = df.groupby("fold").size(), folds.groupby("fold").size()
    return sorted(int(f) for f in want.index if have.get(f, 0) == want[f])


def write(df, name, meta=None, pred_dir=PRED_DIR, folds=None, folds_path=FOLDS_FILE):
    folds = load_folds(folds_path) if folds is None else folds
    validate(df, folds)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cols = KEY + ["fold", "pred"] + (["pred_std"] if "pred_std" in df.columns else [])
    df[cols].to_csv(pred_dir / f"{name}.csv", index=False)
    side = {"model": name, "n_rows": len(df), "folds_sha256": folds_sha(folds_path), "complete_folds": complete_folds(df, folds), **(meta or {})}
    (pred_dir / f"{name}.json").write_text(json.dumps(side, indent=1))


def read(name, pred_dir=PRED_DIR, folds_path=FOLDS_FILE):
    side = pred_dir / f"{name}.json"
    if side.exists():
        made, now = json.loads(side.read_text()).get("folds_sha256"), folds_sha(folds_path)
        assert made is None or now is None or made == now, f"{name}: predictions were made with a different folds file"
    return pd.read_csv(pred_dir / f"{name}.csv")


def null_make(cells, folds):
    """Null model per fold: every test cell gets the mean label of that fold's train cells (never its own fold's labels)."""
    d = folds.merge(cells[KEY + ["agb_main"]], on=KEY, validate="1:1")
    parts = []
    for f in sorted(d.fold.unique()):
        mean = d.loc[d[f"role_{f}"] == "train", "agb_main"].mean()
        parts.append(d.loc[d.fold == f, KEY + ["fold"]].assign(pred=mean))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- evaluation
def nearest_train_distance(cells, folds, use_folds):
    """Distance (m) from each test cell of the used folds to the nearest train cell of its own fold."""
    c = folds.merge(cells[KEY + ["x_utm", "y_utm"]], on=KEY)
    out = []
    for f in use_folds:
        tr, te = c[c[f"role_{f}"] == "train"], c[c.fold == f]
        d, _ = cKDTree(tr[["x_utm", "y_utm"]].to_numpy()).query(te[["x_utm", "y_utm"]].to_numpy())
        out.append(d)
    return np.concatenate(out)


def evaluate(models, cells, folds, block_m=10000, n_boot=2000, seed=42, pairs=(), random_models=None, split=None):
    """models: {name: spatial predictions DataFrame}. random_models: {name: random-split predictions from
    predictions.with_labels} for the 'vs random' table. Returns (dict of DataFrames, info dict)."""
    names = list(models)
    for pair in pairs:
        for n in pair.split(":"):
            if n not in models:
                raise SystemExit(f"--pairs {pair}: '{n}' is not one of the models {names}")
    covered = {n: complete_folds(m, folds) for n, m in models.items()}
    common = sorted(set.intersection(*(set(v) for v in covered.values())))
    if not common:
        raise SystemExit(f"no fold is complete for every model: {covered}")
    ref = cells.merge(folds[KEY + ["fold"]], on=KEY, validate="1:1")
    ref = ref[ref.fold.isin(common)].sort_values(KEY, ignore_index=True)
    codes = B.block_codes(ref.x_utm, ref.y_utm, block_m)
    n_blocks, reps = int(codes.max()) + 1, B.replicates(codes, n_boot, seed)
    if n_blocks < E.MIN_BLOCKS:
        print(f"  WARNING: only {n_blocks} blocks of {block_m} m in the evaluated cells: no confidence interval is given")

    data = {}
    for n, m in models.items():
        d = ref[KEY + ["agb_main", "fold"]].merge(m[KEY + ["pred"]], on=KEY, how="left", validate="1:1")
        assert d.pred.notna().all(), f"{n}: missing predictions"
        data[n] = d.rename(columns={"agb_main": "obs"})
    out = {"metrics": [], "by_fold": [], "paired": [], "vs_random": []}
    for n, d in data.items():
        ci = B.confidence(d.obs, d.pred, reps)
        if d.pred.nunique() <= len(common):          # constant per fold (null model): a slope would be a regression on fold means
            ci.loc[ci.metric == "calib_slope", ["estimate", "ci_lo", "ci_hi"]] = np.nan
        if n_blocks < E.MIN_BLOCKS:
            ci[["ci_lo", "ci_hi"]] = np.nan
        out["metrics"].append(ci.assign(model=n, n=len(d), n_blocks=n_blocks))
        for f, g in d.groupby("fold"):
            out["by_fold"].append({"model": n, "fold": int(f), **M.all_metrics(g.obs, g.pred), "mean_obs": g.obs.mean()})
    for pair in pairs:
        a, b = pair.split(":")
        tab, share = B.paired(data[a].obs, data[a].pred, data[b].pred, reps)
        if n_blocks < E.MIN_BLOCKS:
            tab[["ci_lo", "ci_hi"]] = np.nan
        out["paired"].append(tab.assign(model_a=a, model_b=b, share_a_rmse_lower=share))

    # the same cells scored by the random-split model and by the spatial model: random-split TEST cells only, so that the
    # random-split model has never trained on them either
    split = P.load_split() if split is None else split
    test_keys = ref.merge(split[split["set"] == "test"][KEY], on=KEY)[KEY]
    for n, rnd in (random_models or {}).items():
        if n not in data:
            continue
        r = rnd[rnd["set"] == "test"].merge(test_keys, on=KEY)[KEY + ["obs", "pred"]].rename(columns={"pred": "pred_random"})
        s = data[n].merge(r[KEY + ["pred_random"]], on=KEY, validate="1:1")
        if len(s) < 2:
            continue
        mr, ms = M.all_metrics(s.obs, s.pred_random), M.all_metrics(s.obs, s.pred)
        out["vs_random"].append({"model": n, "n": len(s), "rmse_random": mr["rmse"], "rmse_spatial": ms["rmse"],
                                 "rmse_change_pct": 100 * (ms["rmse"] - mr["rmse"]) / mr["rmse"],
                                 "r2_random": mr["r2"], "r2_spatial": ms["r2"], "bias_random": mr["bias"], "bias_spatial": ms["bias"]})
    res = {k: (pd.concat(v, ignore_index=True) if k in ("metrics", "paired") and v else pd.DataFrame(v)) for k, v in out.items()}
    if not res["vs_random"].empty:
        v = res["vs_random"]
        assert v.n.nunique() == 1, "'vs random': models were compared on different cells"
        if len(v) == len(models):                    # ranks only if every model has a random-split file
            v["rank_random"] = v.rmse_random.rank(method="min").astype(int)
            v["rank_spatial"] = v.rmse_spatial.rank(method="min").astype(int)
        else:
            print(f"  note: only {len(v)} of {len(models)} models have random-split predictions; ranks not compared")
    dist = nearest_train_distance(cells, folds, common)
    rnd_med = E.leakage(cells, split)["median_distance_m"]
    n_train = {int(f): int((folds[f"role_{f}"] == "train").sum()) for f in common}
    info = {"folds_used": common, "random_median_distance_m": rnd_med, "train_cells_per_fold": n_train, "n_cells": len(ref), "n_blocks": n_blocks, "block_m": block_m,
            "distance_to_train_m": {"median": float(np.median(dist)), "min": float(dist.min()),
                                    "within_100m_pct": float(100 * (dist < 100).mean()), "within_1km_pct": float(100 * (dist < 1000).mean())}}
    return res, info


def text(res, info):
    m, lines = res["metrics"], []
    lines.append(f"\nSPATIAL HOLD-OUT: {info['n_cells']} cells in folds {info['folds_used']} ({info['n_blocks']} blocks of {info['block_m']} m); "
                 f"each cell predicted by a model trained only on cells at least {info['distance_to_train_m']['min']:.0f} m away "
                 f"(median {info['distance_to_train_m']['median']:.0f} m; random split: median {info['random_median_distance_m']:.0f} m). "
                 f"Training cells per fold {list(info['train_cells_per_fold'].values())} (random split: {int((P.load_split()['set'] == 'train').sum())})")
    lines.append(f"{'model':10s} {'R2':>7s} {'RMSE  [95% CI]':>26s} {'bias':>8s} {'bias%':>7s} {'slope':>6s} {'MAE':>7s}")
    for n in m.model.unique():
        r = m[m.model == n].set_index("metric")
        ci = "n/a" if np.isnan(r.ci_lo["rmse"]) else f"[{r.ci_lo['rmse']:6.1f},{r.ci_hi['rmse']:6.1f}]"
        lines.append(f"{n:10s} {r.estimate['r2']:7.3f} {r.estimate['rmse']:8.1f} {ci:>17s} {r.estimate['bias']:8.1f} "
                     f"{r.estimate['bias_pct']:7.1f} {r.estimate['calib_slope']:6.2f} {r.estimate['mae']:7.1f}")
    bf = res["by_fold"]
    if not bf.empty:
        lines.append("\nRMSE by fold (regions differ; a model that is good in one fold only is not robust):")
        for n in bf.model.unique():
            d = bf[bf.model == n]
            lines.append(f"{n:10s} " + "  ".join(f"fold {r.fold}: {r.rmse:.0f} (n={r.n})" for r in d.itertuples()))
    pr = res["paired"]
    if not pr.empty:
        lines.append("\nPaired differences A - B on the same blocks (error metric < 0 or R2 > 0 favours A)")
        for (a, b), d in pr.groupby(["model_a", "model_b"], sort=False):
            r = d.set_index("metric")
            lines.append(f"{a} vs {b}: RMSE {r.diff_estimate['rmse']:+.1f} [{r.ci_lo['rmse']:+.1f},{r.ci_hi['rmse']:+.1f}]  "
                         f"R2 {r.diff_estimate['r2']:+.3f}  A lower RMSE in {100 * d.share_a_rmse_lower.iloc[0]:.0f}% of resamples")
    vr = res["vs_random"]
    if not vr.empty:
        lines.append("\nSAME test cells, random-split model vs spatial model (an UPPER BOUND on the optimism of the random split: the spatial model also had fewer training cells):")
        for r in vr.itertuples():
            lines.append(f"{r.model:10s} n={r.n}  RMSE {r.rmse_random:.1f} -> {r.rmse_spatial:.1f} ({r.rmse_change_pct:+.0f}%)  "
                         f"R2 {r.r2_random:.3f} -> {r.r2_spatial:.3f}" + (f"  rank {r.rank_random} -> {r.rank_spatial}" if "rank_random" in vr else ""))
        if "rank_random" in vr and len(vr) > 1:
            changed = (vr.rank_random != vr.rank_spatial).any()
            lines.append("The model ranking " + ("CHANGES under spatial hold-out: do not rely on the random-split ranking alone."
                                                 if changed else "is the same under spatial hold-out."))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make", help="create the folds file once")
    sub.add_parser("null", help="write the null-model spatial predictions")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--models", nargs="+", required=True)
    ev.add_argument("--pairs", nargs="*", default=[])
    ev.add_argument("--block-m", type=int, default=DEFAULTS["block_m"])
    ev.add_argument("--n-boot", type=int, default=2000)
    ev.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    a = ap.parse_args()
    cells = pd.read_csv(P.LABELS / "gedi_cells.csv")

    if a.cmd == "make":
        if FOLDS_FILE.exists():
            raise SystemExit(f"{FOLDS_FILE} already exists: the folds are fixed once and never overwritten")
        f = make_folds(cells, DEFAULTS["k"], DEFAULTS["block_m"], DEFAULTS["buffer_m"], DEFAULTS["seed"])
        FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(FOLDS_FILE, index=False)
        (FOLDS_FILE.parent / "spatial_folds.json").write_text(json.dumps({**DEFAULTS, "sha256": folds_sha(), "n_cells": len(f)}, indent=1))
        print(fold_summary(cells, f).to_string(index=False))
        print(f"wrote {FOLDS_FILE}")
    elif a.cmd == "null":
        folds = load_folds()
        write(null_make(cells, folds), "null", {"note": "mean of each fold's train cells"})
        print(f"wrote {PRED_DIR / 'null.csv'}")
    else:
        folds = load_folds()
        models = {n: read(n) for n in a.models}
        for n, m in models.items():
            validate(m, folds)
        split = P.load_split()
        random_models = {}
        for n in a.models:
            if (P.PRED_DIR / f"{n}.csv").exists():
                random_models[n] = P.with_labels(P.read(n), cells, split)
        res, info = evaluate(models, cells, folds, a.block_m, a.n_boot, a.seed, a.pairs, random_models)
        OUT.mkdir(parents=True, exist_ok=True)
        for k, f in [("metrics", "results_spatial_metrics"), ("by_fold", "results_spatial_by_fold"),
                     ("paired", "results_spatial_paired"), ("vs_random", "results_spatial_vs_random")]:
            if not res[k].empty:
                res[k].to_csv(OUT / f"{f}.csv", index=False)
        (OUT / "spatial_settings.json").write_text(json.dumps({"models": a.models, "pairs": a.pairs, "seed": a.seed, "n_boot": a.n_boot,
                                                               "folds_sha256": folds_sha(), **info}, indent=1))
        print(text(res, info))
        print(f"\nwrote {OUT}")
