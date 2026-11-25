"""Block bootstrap confidence intervals and paired model comparison (Project.md 5.1, 5.2).

GEDI cells lie in dense tracks, so neighbouring cells are alike. Resampling single cells would give intervals that
are too narrow. Instead the cells are grouped into square blocks (side `size` metres, on the UTM x / y of the cell
centres) and whole blocks are resampled with replacement. The same resampled blocks are used for every model, so
differences between models are paired.

This only handles the dependence BETWEEN TEST CELLS in the interval. It does not remove the optimism caused by the
random split (test cells close to training cells); the two must not be confused in the text.
"""
import numpy as np
import pandas as pd

import metrics as M


def block_codes(x, y, size):
    """Integer block id (0..n_blocks-1) of each cell: the square of side `size` m that contains it."""
    ij = np.stack([np.floor(np.asarray(x, float) / size), np.floor(np.asarray(y, float) / size)], 1).astype(np.int64)
    return np.unique(ij, axis=0, return_inverse=True)[1].ravel()


def replicates(codes, n_boot, seed):
    """n_boot resamples. Each is the list of cell positions of n_blocks blocks drawn with replacement."""
    rng = np.random.default_rng(seed)
    n_blocks = int(codes.max()) + 1
    members = [np.nonzero(codes == k)[0] for k in range(n_blocks)]
    return [np.concatenate([members[k] for k in rng.integers(0, n_blocks, n_blocks)]) for _ in range(n_boot)]


def confidence(y, p, reps, level=0.95):
    """DataFrame (metric, estimate, ci_lo, ci_hi): estimate on all cells, percentile interval over the replicates."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    point = M.all_metrics(y, p)
    draws = pd.DataFrame([M.all_metrics(y[r], p[r]) for r in reps])
    a = (1 - level) / 2
    return pd.DataFrame([{"metric": m, "estimate": point[m], "ci_lo": draws[m].quantile(a), "ci_hi": draws[m].quantile(1 - a)}
                         for m in M.METRIC_NAMES])


def paired(y, pa, pb, reps, level=0.95):
    """Model A minus model B on the SAME resampled blocks. Returns (table, share of resamples with A's RMSE < B's).
    Difference of an error metric < 0 means A is better; for r2 > 0 means A is better."""
    y, pa, pb = (np.asarray(v, float) for v in (y, pa, pb))
    pt_a, pt_b = M.all_metrics(y, pa), M.all_metrics(y, pb)
    da = pd.DataFrame([M.all_metrics(y[r], pa[r]) for r in reps])
    db = pd.DataFrame([M.all_metrics(y[r], pb[r]) for r in reps])
    a = (1 - level) / 2
    rows = []
    for m in ["r2", "rmse", "mae", "bias"]:
        d = da[m] - db[m]
        rows.append({"metric": m, "diff_estimate": pt_a[m] - pt_b[m], "ci_lo": d.quantile(a), "ci_hi": d.quantile(1 - a)})
    return pd.DataFrame(rows), float((da["rmse"] < db["rmse"]).mean())
