"""Run-to-run (seed) variance, and whether a model difference is bigger than it (Project.md 5.2, RQ1).

The block-bootstrap intervals in evaluate.py hold the predictions fixed: they say how much a score changes with a
different draw of test blocks. They say nothing about how much the model itself changes when only the random seed
changes. If two models differ by less than their own seed-to-seed spread, no winner can be claimed.

Input: one predictions file per seed (same format as predictions.py), for example data/predictions/rf_s0.csv ... rf_s4.csv.
    RF / XGBoost : models/rf.py and models/xgb.py (main) write the members of each ensemble as <model>_s<seed>.csv (same split, features,
                   settings). Member k also leaves out fold k of the pooled train + validation cells, so the spread is seed PLUS training-subset (68%) variation.
    U-Nets       : use the individual ensemble members as the "seeds" (member-level scores). The ensemble mean stays the
                   headline in evaluate.py; the spread of members shows how much one trained network varies.
Run (from the Model Analysis folder):
    python evaluation/seed_variance.py --groups rf=rf_s0,rf_s1,rf_s2 xgb=xgb_s0,xgb_s1,xgb_s2 --pairs rf:xgb
Writes data/results/results_seeds.csv (one row per seed) and results_seed_summary.csv / results_seed_pairs.csv.
Scores are on the test cells. At least 3 seeds per model are advisable; with 2 the spread is barely estimated.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M
import predictions as P

OUT = P.EVAL_ROOT / "data" / "results"
SHOW = ["rmse", "r2", "bias", "calib_slope"]


def per_seed(groups):
    """groups: {model: [DataFrame from predictions.with_labels, one per seed]} -> one row per (model, seed) with test metrics.
    Every file must cover the same test cells (checked)."""
    rows, keys = [], None
    for name, runs in groups.items():
        for i, d in enumerate(runs):
            t = d[d["set"] == "test"].sort_values(P.KEY)
            if len(t) == 0:
                raise SystemExit(f"{name} seed {i}: no test cells")
            k = t[P.KEY].to_numpy()
            if keys is None:
                keys = k
            if not np.array_equal(k, keys):
                raise SystemExit(f"{name} seed {i}: does not cover exactly the same test cells as the other runs")
            rows.append({"model": name, "seed_index": i, **M.all_metrics(t.obs, t.pred)})
    return pd.DataFrame(rows)


def summary(seeds):
    """Mean, sd (ddof=1), min and max over seeds, per model."""
    rows = []
    for name, g in seeds.groupby("model", sort=False):
        r = {"model": name, "n_seeds": len(g)}
        for m in SHOW:
            r.update({f"{m}_mean": g[m].mean(), f"{m}_sd": g[m].std(ddof=1) if len(g) > 1 else np.nan,
                      f"{m}_min": g[m].min(), f"{m}_max": g[m].max()})
        rows.append(r)
    return pd.DataFrame(rows)


def pair_table(seeds, pairs):
    """For A:B: the difference of mean RMSE, the pooled seed sd, their ratio, and the share of all (seed of A, seed of B)
    combinations in which A has the lower RMSE (1 or 0 = the order never changes with the seed)."""
    rows = []
    for pair in pairs:
        a, b = pair.split(":")
        for n in (a, b):
            if n not in set(seeds.model):
                raise SystemExit(f"--pairs {pair}: '{n}' is not one of the models {list(seeds.model.unique())}")
        ra, rb = (seeds.loc[seeds.model == n, "rmse"].to_numpy() for n in (a, b))
        diff = ra.mean() - rb.mean()
        sd = np.sqrt((ra.var(ddof=1) + rb.var(ddof=1)) / 2) if len(ra) > 1 and len(rb) > 1 else np.nan
        share = float(((ra[:, None] < rb[None, :]) + 0.5 * (ra[:, None] == rb[None, :])).mean())      # ties count half
        se = sd * np.sqrt(1 / len(ra) + 1 / len(rb)) if sd == sd else np.nan
        rows.append({"model_a": a, "model_b": b, "rmse_diff_a_minus_b": diff, "pooled_seed_sd": sd,
                     "diff_over_sd": abs(diff) / sd if sd and sd > 0 else np.nan,
                     "diff_over_se": abs(diff) / se if se and se > 0 else np.nan, "share_a_lower_rmse": share,
                     "order_never_changes": share in (0.0, 1.0)})
    return pd.DataFrame(rows)


def text(summ, pairs):
    lines = ["\nSeed variance (test cells): mean +/- sd over seeds [min, max]"]
    for r in summ.itertuples():
        lines.append(f"{r.model:10s} {r.n_seeds} seeds  RMSE {r.rmse_mean:.1f} +/- {r.rmse_sd:.1f} [{r.rmse_min:.1f}, {r.rmse_max:.1f}]  "
                     f"R2 {r.r2_mean:.3f} +/- {r.r2_sd:.3f}  bias {r.bias_mean:+.1f}  slope {r.calib_slope_mean:.2f}")
    if not pairs.empty:
        lines.append("\nIs the difference bigger than the seed spread? (A - B; diff/sd above about 2 and an order that never changes "
                     "support a real difference (a heuristic: diff/se is the same difference against the standard error of the means); otherwise do not call a winner)")
        for r in pairs.itertuples():
            lines.append(f"{r.model_a} vs {r.model_b}: RMSE diff {r.rmse_diff_a_minus_b:+.1f}, pooled seed sd {r.pooled_seed_sd:.1f}, "
                         f"diff/sd {r.diff_over_sd:.1f}, diff/se {r.diff_over_se:.1f}, A lower in {100 * r.share_a_lower_rmse:.0f}% of seed combinations "
                         f"(order never changes: {r.order_never_changes})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True, help="model=file1,file2,... (names in data/predictions/)")
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B model names to compare")
    a = ap.parse_args()
    cells, split = pd.read_csv(P.LABELS / "gedi_cells.csv"), P.load_split()
    groups = {}
    for g in a.groups:
        name, files = g.split("=")
        groups[name] = [P.with_labels(P.read(f), cells, split) for f in files.split(",")]
    s = per_seed(groups)
    summ, pt = summary(s), pair_table(s, a.pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "results_seeds.csv", index=False)
    summ.to_csv(OUT / "results_seed_summary.csv", index=False)
    pt.to_csv(OUT / "results_seed_pairs.csv", index=False)
    print(text(summ, pt))
