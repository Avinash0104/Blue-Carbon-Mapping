"""Evaluate one or more models from their predictions files (Project.md 5, 8). Same code for every model.

Run (from the Model Analysis folder):
    python evaluation/evaluate.py --models null rf xgb --pairs rf:null xgb:rf
Reads data/predictions/<model>.csv (see predictions.py). Writes data/results/:
    results_metrics.csv      per model and set: each metric with a block-bootstrap 95% interval
    results_by_range.csv     errors by observed biomass range (test cells)
    results_uncertainty.csv  error against the model's own uncertainty (models that give pred_std; test cells)
    results_noise_floor.csv  label-noise floor: best achievable RMSE / R2 from repeat GEDI shots (noise_floor.py)
    results_aggregated.csv   test cells averaged in 50 m / 100 m squares holding 2+ test cells (small sample, see aggregate.py)
    results_paired.csv       paired differences between the model pairs given in --pairs (test cells)
    settings.json            every setting, the bins, the split hash and the random-split caveat numbers
Headline = test cells. Train and validation cells are shown too but are NOT independent (they are the pool the models were fitted and tuned on).
The cells of each set, and the bootstrap blocks, come from the split file, never from one of the models, so every
model is scored on exactly the same cells. Outputs are derived and may be overwritten.

What the intervals mean: the bootstrap resamples test BLOCKS with the predictions held fixed. It captures sampling
variation between test blocks only. It does not capture training or seed variance, it is not bias-corrected
(plain percentile), and it does not remove the optimism of the random split (see the caveat printed at the end).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate as A
import blocks as B
import metrics as M
import noise_floor as NF
import predictions as P

OUT = P.EVAL_ROOT / "data" / "results"
DEFAULTS = {"block_m": 4000, "n_boot": 2000, "seed": 42}     # 4 km is about the variogram range (3.4-4.5 km)
MIN_BLOCKS = 20                                              # fewer blocks than this: no confidence interval is given


def leakage(cells, split, eval_set="test"):
    """Share of the evaluated cells that have a training cell within 100 m / 1 km (the random-split caveat)."""
    t = cells.merge(split, on=P.KEY)
    tr, ev = t[t["set"] == "train"], t[t["set"] == eval_set]
    d, _ = cKDTree(tr[["x_utm", "y_utm"]].to_numpy()).query(ev[["x_utm", "y_utm"]].to_numpy())
    return {"eval_set": eval_set, "within_100m_pct": float(100 * (d < 100).mean()),
            "within_1km_pct": float(100 * (d < 1000).mean()), "median_distance_m": float(np.median(d))}


def evaluate(models, reference, block_m=4000, n_boot=2000, seed=42, pairs=(), sets=("test", "val")):
    """models: {name: DataFrame from predictions.with_labels}.
    reference: the label cells joined to the split (cell_row, cell_col, set, x_utm, y_utm): it defines which cells
    are in each set and how they are grouped into blocks, the same for every model.
    Returns a dict of DataFrames and the number of blocks per set."""
    names = list(models)
    for pair in pairs:
        for n in pair.split(":"):
            if n not in models:
                raise SystemExit(f"--pairs {pair}: '{n}' is not one of the models {names}")
    ref = reference.sort_values(P.KEY, ignore_index=True)
    reps, n_blocks, keys = {}, {}, {}
    for s in sets:
        sub = ref[ref["set"] == s]
        if sub.empty:
            print(f"  set '{s}': no cells in the split, skipped")
            continue
        codes = B.block_codes(sub.x_utm, sub.y_utm, block_m)
        reps[s], n_blocks[s], keys[s] = B.replicates(codes, n_boot, seed), int(codes.max()) + 1, sub[P.KEY].to_numpy()
    if "test" not in keys:
        raise SystemExit("the split has no test cells")
    used = list(keys)
    for s in used:
        if n_blocks[s] < MIN_BLOCKS:
            print(f"  WARNING: only {n_blocks[s]} blocks of {block_m} m in the '{s}' cells: no confidence interval is given")

    def rows(name, s):
        """The cells of model `name` in set s, or None unless it covers exactly the split's cells of that set."""
        d = models[name][models[name]["set"] == s]
        return d if len(d) == len(keys[s]) and (d[P.KEY].to_numpy() == keys[s]).all() else None

    out = {"metrics": [], "by_range": [], "uncertainty": [], "paired": [], "aggregated": []}
    agg_reps = {}                                   # bootstrap replicates per square size (same squares for every model)
    for name in names:
        if rows(name, "test") is None:
            raise SystemExit(f"{name}: does not cover exactly the {len(keys['test'])} test cells of the split")
        for s in used:
            d = rows(name, s)
            if d is None:
                print(f"  {name}: set '{s}' does not cover every cell of the split, skipped")
                continue
            ci = B.confidence(d.obs, d.pred, reps[s])
            if n_blocks[s] < MIN_BLOCKS:
                ci[["ci_lo", "ci_hi"]] = np.nan
            out["metrics"].append(ci.assign(model=name, set=s, used_for_selection=(s != "test"), n=len(d), n_blocks=n_blocks[s]))
        d = rows(name, "test")
        out["by_range"].append(M.by_range(d.obs, d.pred).assign(model=name))
        for label, side in A.SIDES.items():            # test-cell version of aggregated accuracy (small sample)
            g = A.group_cells(d, side)
            if len(g) < 2:
                print(f"  {name}: fewer than 2 squares of {label} hold 2+ test cells, aggregated accuracy skipped")
                continue
            codes = B.block_codes(g.x_utm, g.y_utm, block_m)
            if side not in agg_reps:
                agg_reps[side] = B.replicates(codes, n_boot, seed)
            ci = B.confidence(g.obs, g.pred, agg_reps[side])
            nb = int(codes.max()) + 1
            if nb < MIN_BLOCKS:
                ci[["ci_lo", "ci_hi"]] = np.nan
            out["aggregated"].append(ci.assign(model=name, square=label, n_squares=len(g), n_cells=int(g.n_cells.sum()), n_blocks=nb))
        if "pred_std" in d.columns:
            tab, rho, mono = M.uncertainty_table(d.obs, d.pred, d.pred_std)
            if tab.empty:
                print(f"  {name}: pred_std has too few distinct values for an uncertainty check, skipped")
            else:
                out["uncertainty"].append(tab.assign(model=name, spearman_rho=rho, mae_rises_every_bin=mono))
    for pair in pairs:
        a, b = pair.split(":")
        da, db = rows(a, "test"), rows(b, "test")
        tab, share = B.paired(da.obs, da.pred, db.pred, reps["test"])
        if n_blocks["test"] < MIN_BLOCKS:
            tab[["ci_lo", "ci_hi"]] = np.nan
        out["paired"].append(tab.assign(model_a=a, model_b=b, share_a_rmse_lower=share))
    res = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in out.items()}
    return res, n_blocks


def summary(res, sets=("test", "val")):
    """Printable tables (headline = test)."""
    lines = []
    m = res["metrics"]
    for s in sets:
        d = m[m["set"] == s]
        if d.empty:
            continue
        if s == "test":
            lines.append(f"\nTEST cells ({int(d.n.iloc[0])} cells in {int(d.n_blocks.iloc[0])} blocks; not used for selection, "
                         f"but see the random-split caveat below)")
        else:
            lines.append(f"\nVAL cells ({int(d.n.iloc[0])} cells in {int(d.n_blocks.iloc[0])} blocks; NOT independent: "
                         f"used to choose the early-stopping point)")
        lines.append(f"{'model':10s} {'n':>5s} {'R2':>7s} {'R2 (r^2)':>9s} {'RMSE  [95% CI]':>26s} {'relRMSE%':>9s} {'bias':>8s} {'bias%':>7s} "
                     f"{'slope':>6s} {'MAE':>7s}")
        for name in d.model.unique():
            r = d[d.model == name].set_index("metric")
            lines.append(f"{name:10s} {int(r.n.iloc[0]):5d} {r.estimate['r2']:7.3f} {r.estimate['r2_pearson']:9.3f} "
                         f"{r.estimate['rmse']:8.1f} [{r.ci_lo['rmse']:6.1f},{r.ci_hi['rmse']:6.1f}] "
                         f"{r.estimate['rel_rmse_pct']:9.1f} {r.estimate['bias']:8.1f} {r.estimate['bias_pct']:7.1f} "
                         f"{r.estimate['calib_slope']:6.2f} {r.estimate['mae']:7.1f}")
    br = res["by_range"]
    if not br.empty:
        lines.append("\nErrors by OBSERVED biomass range (Mg/ha), test cells: n / bias / RMSE")
        lines.append("  Reading note: binning by the observed value mechanically gives positive bias in low bins and negative bias in "
                     "high bins for ANY model, even the null model (regression to the mean). Compare each model with the null row; "
                     "only error beyond that pattern says something about saturation. Bins with small n are noisy.")
        for name in br.model.unique():
            d = br[br.model == name]
            lines.append(f"{name:10s} " + "  ".join(f"{r.bin}: {r.n}/{r.bias:+.0f}/{r.rmse:.0f}" for r in d.itertuples()))
    un = res["uncertainty"]
    if not un.empty:
        lines.append("\nUncertainty check (test cells). Main evidence: Spearman rho between pred_std and |error| (should be > 0). "
                     "Secondary: mean |error| in each pred_std bin (strictly rising is a strict test; one noisy bin breaks it)")
        for name in un.model.unique():
            d = un[un.model == name]
            lines.append(f"{name:10s} rho {d.spearman_rho.iloc[0]:+.3f}  rises in every bin: {bool(d.mae_rises_every_bin.iloc[0])}  "
                         f"mean |error| by bin: " + ", ".join(f"{v:.0f}" for v in d.mae))
    ag = res["aggregated"]
    if not ag.empty:
        lines.append("\nAGGREGATED accuracy, test cells grouped into squares with 2+ test cells (SMALL SAMPLE: squares average only "
                     "2-3 cells, so this indicates the effect of averaging; it is not the map's 0.25-1 ha accuracy): "
                     "squares / RMSE [95% CI] / R2 / bias")
        for (name, sq), d in ag.groupby(["model", "square"], sort=False):
            r = d.set_index("metric")
            ci = "n/a (too few blocks)" if np.isnan(r.ci_lo["rmse"]) else f"[{r.ci_lo['rmse']:.1f},{r.ci_hi['rmse']:.1f}]"
            lines.append(f"{name:10s} {sq:15s} {int(d.n_squares.iloc[0]):4d} squares ({int(d.n_blocks.iloc[0])} blocks)  "
                         f"RMSE {r.estimate['rmse']:.1f} {ci}  R2 {r.estimate['r2']:.3f}  bias {r.estimate['bias']:+.1f}")
    pr = res["paired"]
    if not pr.empty:
        lines.append("\nPaired differences A - B on the same blocks (test cells; error metric < 0 or R2 > 0 favours A)")
        for (a, b), d in pr.groupby(["model_a", "model_b"], sort=False):
            r = d.set_index("metric")
            lines.append(f"{a} vs {b}: RMSE {r.diff_estimate['rmse']:+.1f} [{r.ci_lo['rmse']:+.1f},{r.ci_hi['rmse']:+.1f}]  "
                         f"R2 {r.diff_estimate['r2']:+.3f} [{r.ci_lo['r2']:+.3f},{r.ci_hi['r2']:+.3f}]  "
                         f"A has the lower RMSE in {100 * d.share_a_rmse_lower.iloc[0]:.0f}% of resamples")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="names of files in data/predictions/")
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B pairs to compare, stated in advance")
    ap.add_argument("--block-m", type=int, default=DEFAULTS["block_m"])
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    a = ap.parse_args()

    cells = pd.read_csv(P.LABELS / "gedi_cells.csv")
    split = P.load_split()
    models = {n: P.with_labels(P.read(n), cells, split) for n in a.models}
    reference = cells.merge(split, on=P.KEY, validate="1:1")
    assert np.isfinite(reference[["x_utm", "y_utm"]].to_numpy()).all(), "NaN coordinates in gedi_cells.csv"
    res, n_blocks = evaluate(models, reference, a.block_m, a.n_boot, a.seed, a.pairs)

    OUT.mkdir(parents=True, exist_ok=True)
    for k, f in [("metrics", "results_metrics"), ("by_range", "results_by_range"),
                 ("uncertainty", "results_uncertainty"), ("paired", "results_paired"),
                 ("aggregated", "results_aggregated")]:
        if not res[k].empty:
            res[k].to_csv(OUT / f"{f}.csv", index=False)
    leak = leakage(cells, split)
    floor = NF.compute(cells, split, block_m=a.block_m, n_boot=a.n_boot, seed=a.seed)
    floor.to_csv(OUT / "results_noise_floor.csv", index=False)
    settings = {"models": a.models, "pairs": a.pairs, "block_m": a.block_m, "n_boot": a.n_boot, "seed": a.seed, "ci_level": 0.95,
                "ci_method": "percentile, predictions held fixed (no training/seed variance)", "min_blocks_for_ci": MIN_BLOCKS,
                "bin_edges": [float(e) for e in M.BIN_EDGES], "n_blocks": n_blocks,
                "split_sha256": P.split_sha(), "random_split_caveat": leak, "noise_floor": floor.iloc[0].to_dict()}
    (OUT / "settings.json").write_text(json.dumps(settings, indent=1))

    print(summary(res))
    print("\n" + NF.describe(floor))
    print(f"\nSettings: blocks {a.block_m} m, {a.n_boot} resamples, seed {a.seed}, 95% percentile intervals; "
          f"bins {[float(e) for e in M.BIN_EDGES]} Mg/ha")
    print(f"CAVEAT (random split): {leak['within_100m_pct']:.0f}% of test cells have a training cell within 100 m, "
          f"{leak['within_1km_pct']:.0f}% within 1 km (median {leak['median_distance_m']:.0f} m). Scores are optimistic compared with an unseen "
          f"area; the block bootstrap handles dependence between test cells, not this proximity to training cells.")
    print(f"wrote {OUT}")
