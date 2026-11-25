"""Accuracy metrics shared by all models (Project.md 5.1, 5.3). Pure functions, no file access.

Everything is at the CELL level, in Mg/ha: y = the cell's GEDI-derived biomass (agb_main), p = the model's
cell-mean prediction. Definitions (decided 2026-11-24):
    r2            1 - SSres / SStot, with SStot around the mean of y (can be negative; the null model gives about 0)
    r2_pearson    squared Pearson correlation (reported separately and named, because a biased model can look perfect)
    rmse, mae     in Mg/ha
    rel_rmse_pct  100 * RMSE / mean(y)
    bias          mean(p - y): positive = over-prediction
    bias_pct      100 * bias / mean(y): the over/under-prediction as a share of the mean, i.e. the bias of a carbon total
    calib_slope   slope of y regressed on p (1 = calibrated). Below 1: the prediction range is too wide (extremes
                  over-stated). Above 1: the range is too narrow (compressed, e.g. saturation in dense stands).
Empty or degenerate input gives NaN, not an error.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Biomass bins for "errors by range" (Mg/ha, left-closed, last one open). Fixed in advance of any model results and
# assigned by the OBSERVED value. Training labels have p10-p90 of about 100-397 Mg/ha.
BIN_EDGES = [0, 100, 200, 300, 400, np.inf]
METRIC_NAMES = ["r2", "r2_pearson", "rmse", "rel_rmse_pct", "bias", "mae", "bias_pct", "calib_slope"]


def _prep(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    assert y.shape == p.shape and y.ndim == 1, "y and p must be 1-D and the same length"
    return y, p


def r2(y, p):
    y, p = _prep(y, p)
    if len(y) == 0:
        return np.nan
    sst = ((y - y.mean()) ** 2).sum()
    return np.nan if sst == 0 else 1 - ((y - p) ** 2).sum() / sst


def r2_pearson(y, p):
    y, p = _prep(y, p)
    if len(y) < 2 or y.std() == 0 or p.std() == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1] ** 2)


def rmse(y, p):
    y, p = _prep(y, p)
    return np.nan if len(y) == 0 else float(np.sqrt(((p - y) ** 2).mean()))


def mae(y, p):
    y, p = _prep(y, p)
    return np.nan if len(y) == 0 else float(np.abs(p - y).mean())


def bias(y, p):
    y, p = _prep(y, p)
    return np.nan if len(y) == 0 else float((p - y).mean())


def bias_pct(y, p):
    y, p = _prep(y, p)
    return np.nan if len(y) == 0 or y.mean() == 0 else 100 * bias(y, p) / y.mean()


def calib_slope(y, p):
    y, p = _prep(y, p)
    if len(y) < 2 or p.std() == 0:
        return np.nan
    return float(((p - p.mean()) * (y - y.mean())).mean() / p.var())


def rel_rmse_pct(y, p):
    y, p = _prep(y, p)
    return np.nan if len(y) == 0 or y.mean() == 0 else 100 * rmse(y, p) / y.mean()


def all_metrics(y, p):
    """dict: n plus every metric in METRIC_NAMES."""
    y, p = _prep(y, p)
    return {"n": len(y), "r2": r2(y, p), "r2_pearson": r2_pearson(y, p), "rmse": rmse(y, p),
            "rel_rmse_pct": rel_rmse_pct(y, p), "bias": bias(y, p), "mae": mae(y, p),
            "bias_pct": bias_pct(y, p), "calib_slope": calib_slope(y, p)}


def by_range(y, p, edges=BIN_EDGES):
    """Errors by biomass range: one row per bin of the observed value y (left-closed [lo, hi)).
    R2 is not given per bin (the variance inside a bin is truncated, so it would mislead).
    Reading note: because the bins use the OBSERVED value, bias is mechanically positive in low bins and negative in
    high bins for any model (regression to the mean), the null model included. Compare with the null row before
    calling it saturation. Bins with small n are noisy."""
    y, p = _prep(y, p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y >= lo) & (y < hi)
        label = f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}"
        e = p[m] - y[m]
        rows.append({"bin": label, "n": int(m.sum()),
                     "mean_obs": y[m].mean() if m.any() else np.nan, "mean_pred": p[m].mean() if m.any() else np.nan,
                     "bias": e.mean() if m.any() else np.nan,
                     "rmse": np.sqrt((e ** 2).mean()) if m.any() else np.nan,
                     "mae": np.abs(e).mean() if m.any() else np.nan})
    out = pd.DataFrame(rows)
    assert out.n.sum() == (y >= edges[0]).sum(), "a value fell in no bin"
    return out


def uncertainty_table(y, p, sd, nbins=5):
    """Does error rise with the model's own uncertainty (Project.md 5.3)?
    Cells are split into nbins equal-count bins of sd (the ensemble spread of the cell-mean prediction).
    Returns (table, spearman rho between sd and |error|, True if the mean |error| rises in every bin step)."""
    y, p = _prep(y, p)
    sd = np.asarray(sd, float)
    assert sd.shape == y.shape and (sd >= 0).all()
    if len(y) < nbins or np.unique(sd).size < nbins:        # no spread to rank (e.g. one seed): nothing to check
        return pd.DataFrame(), np.nan, False
    err = np.abs(p - y)
    b = pd.qcut(pd.Series(sd).rank(method="first"), nbins, labels=False).to_numpy()   # ranks: ties cannot merge bins
    rows = [{"bin": k + 1, "n": int((b == k).sum()), "mean_sd": sd[b == k].mean(),
             "rmse": float(np.sqrt((err[b == k] ** 2).mean())), "mae": err[b == k].mean()} for k in range(nbins)]
    table = pd.DataFrame(rows)
    rho = float(spearmanr(sd, err).statistic)
    return table, rho, bool((np.diff(table.mae.to_numpy()) > 0).all())
