"""Label-noise floor: the best RMSE any model could reach against these labels (Project.md 5.1).

The label of a cell is GEDI-derived and noisy. 162 of the 12,638 cells were hit by two shots in different months, and
the two shots disagree. If the two shots of a cell are independent measurements of the same cell with the same noise,
then for a pair (a, b):  Var(a - b) = 2 sigma^2, so  sigma = sqrt(mean((a - b)^2) / 2)  is the noise of ONE shot.
A cell label that averages n shots has noise sigma / sqrt(n). Even a perfect model cannot beat that noise, so for a set
of cells
    RMSE floor      = sqrt(mean(sigma^2 / n_shots))
    R2 ceiling      = 1 - floor^2 / var(obs)
Read a model's RMSE against the floor: far above it = room to improve; close to it = the labels, not the model, limit
the score; below it would suggest overfitting to noise or an unrepresentative sample of pairs.

Limits (state them when reporting):
  * Only the pairs (162, 1.3% of cells) are available, so sigma has a wide interval (block bootstrap, 4 km blocks).
  * Repeat shots fall in the same 25 m cell, so noise that two shots share (a common geolocation offset, the height-to-
    biomass equation error, a persistent canopy change) is NOT captured. The floor is an approximate LOWER bound.
  * Real change between the two months and different footprint positions inside the cell are counted as noise: they
    cannot be predicted from imagery at the cell level either, so this is intended.
  * Both shots of a pair passed the 0-33.9 m height cap, so the largest disagreements are truncated and sigma is slightly
    low (the safe direction for a lower bound).
  * The interval of the floor carries only the uncertainty of sigma (162 pairs, wide). It is not the sampling noise of a
    test RMSE, so "model RMSE below the floor" is a warning sign to look at, not a strict test.
  * One sigma for all biomass levels. Noise probably grows with biomass; this is an average over the pairs.

Run (from the Model Analysis folder): python evaluation/noise_floor.py   (writes data/results/results_noise_floor.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocks as B
import predictions as P

SHOTS = P.LABELS / "gedi_shots.csv"


def sigma_of(d):
    """Noise of one shot from the differences d = a - b of repeat shots."""
    d = np.asarray(d, float)
    return float(np.sqrt((d ** 2).mean() / 2)) if len(d) else np.nan


def pairs_table(shots, cells):
    """One row per cell with exactly two shots: a, b (agb_main of the earlier and later month) and the cell coordinates."""
    s = shots.sort_values(P.KEY + ["month"])
    two = s[s.groupby(P.KEY)["agb_main"].transform("size") == 2]
    p = two.groupby(P.KEY)["agb_main"].agg(a="first", b="last").reset_index()
    return p.merge(cells[P.KEY + ["x_utm", "y_utm"]], on=P.KEY, how="left", validate="1:1")


def estimate(pairs, block_m=4000, n_boot=2000, seed=42, level=0.95):
    """sigma of one shot with a block-bootstrap interval (blocks of pairs, same blocks logic as the main evaluation)."""
    d = (pairs.a - pairs.b).to_numpy(float)
    if len(d) < 2:
        return {"n_pairs": len(d), "sigma": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_blocks": 0, "sigma_rel_pct": np.nan}
    codes = B.block_codes(pairs.x_utm, pairs.y_utm, block_m)
    draws = np.array([sigma_of(d[r]) for r in B.replicates(codes, n_boot, seed)])
    a = (1 - level) / 2
    m = float(np.concatenate([pairs.a, pairs.b]).mean())
    sig = sigma_of(d)
    return {"n_pairs": len(d), "sigma": sig, "ci_lo": float(np.quantile(draws, a)), "ci_hi": float(np.quantile(draws, 1 - a)),
            "n_blocks": int(codes.max()) + 1, "sigma_rel_pct": 100 * sig / m}


def floor_for(obs, n_shots, sigma):
    """RMSE floor and R2 ceiling for cells with labels `obs` that each average `n_shots` shots."""
    obs, n_shots = np.asarray(obs, float), np.asarray(n_shots, float)
    rmse = float(np.sqrt((sigma ** 2 / n_shots).mean()))
    var = obs.var()
    return {"rmse_floor": rmse, "r2_ceiling": float(1 - rmse ** 2 / var) if var > 0 else np.nan}


def compute(cells, split, shots=None, block_m=4000, n_boot=2000, seed=42, eval_set="test"):
    """One-row DataFrame: sigma with interval, and the floor/ceiling for the cells of eval_set (the headline test cells)."""
    shots = pd.read_csv(SHOTS) if shots is None else shots
    est = estimate(pairs_table(shots, cells), block_m, n_boot, seed)
    c = cells.merge(split, on=P.KEY, validate="1:1")
    c = c[c["set"] == eval_set]
    fl = floor_for(c.agb_main, c.n_shots, est["sigma"])
    # the interval of sigma carries over to the floor (the floor is increasing in sigma)
    lo, hi = (floor_for(c.agb_main, c.n_shots, est[k])["rmse_floor"] for k in ("ci_lo", "ci_hi"))
    return pd.DataFrame([{**est, **fl, "floor_ci_lo": lo, "floor_ci_hi": hi, "eval_set": eval_set, "n_eval_cells": len(c),
                          "obs_sd": float(c.agb_main.std(ddof=0)), "obs_mean": float(c.agb_main.mean())}])


def describe(r):
    """Printable summary of the one-row table from compute()."""
    r = r.iloc[0]
    return (f"Label-noise floor ({int(r.n_pairs)} repeat-shot pairs in {int(r.n_blocks)} blocks): one shot has noise sigma = "
            f"{r.sigma:.1f} Mg/ha [{r.ci_lo:.1f}, {r.ci_hi:.1f}] ({r.sigma_rel_pct:.0f}% of the mean). On the {r.eval_set} cells the best "
            f"achievable RMSE is about {r.rmse_floor:.1f} Mg/ha [{r.floor_ci_lo:.1f}, {r.floor_ci_hi:.1f}] and the R2 ceiling about "
            f"{r.r2_ceiling:.2f} (null model RMSE is about {r.obs_sd:.1f}). Approximate lower bound: noise shared by both shots is not seen.")


if __name__ == "__main__":
    cells = pd.read_csv(P.LABELS / "gedi_cells.csv")
    res = compute(cells, P.load_split())
    out = P.EVAL_ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results_noise_floor.csv", index=False)
    print(describe(res))
