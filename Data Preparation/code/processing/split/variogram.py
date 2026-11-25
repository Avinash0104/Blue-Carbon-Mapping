"""Empirical variogram of the GEDI cell labels (diagnostic for the random split).

Tells how far apart two label cells must be before they stop resembling each other; that distance shows
how much a random split lets neighbouring cells leak between train and test. Run: python code/processing/split/variogram.py
Writes data/processed/labels/variogram.csv and the figure in the notes vault.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import region

ROOT = region.ROOT
OUT_CSV = ROOT / "data" / "processed" / "labels" / "variogram.csv"
OUT_PNG = ROOT.parent / "Project Context" / "Blue Carbon Mapping" / "variogram.png"
EDGES = np.r_[np.arange(0, 5_000, 250), np.arange(5_000, 20_000, 1_000), np.arange(20_000, 81_000, 5_000)]


def empirical(xy, v, edges):
    """Semivariance per distance bin: mean of 0.5*(v_i - v_j)^2 over all pairs; also the pair counts."""
    t = cKDTree(xy)
    pairs = t.query_pairs(edges[-1], output_type="ndarray")
    d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    g = 0.5 * (v[pairs[:, 0]] - v[pairs[:, 1]]) ** 2
    k = np.digitize(d, edges) - 1
    n = np.bincount(k, minlength=len(edges) - 1)[: len(edges) - 1]
    s = np.bincount(k, weights=g, minlength=len(edges) - 1)[: len(edges) - 1]
    dm = np.bincount(k, weights=d, minlength=len(edges) - 1)[: len(edges) - 1]
    ok = n > 0
    return dm[ok] / n[ok], s[ok] / n[ok], n[ok]


def spherical(h, nug, sill, a):
    r = np.minimum(h / a, 1)
    return nug + (sill - nug) * (1.5 * r - 0.5 * r ** 3)


def exponential(h, nug, sill, a):
    return nug + (sill - nug) * (1 - np.exp(-h / a))


def fit(model, h, g, n, v):
    p0 = [0.1 * v, v, 10_000]
    p, _ = curve_fit(model, h, g, p0=p0, sigma=1 / np.sqrt(n), bounds=([0, 0, 200], [v, 3 * v, 200_000]), maxfev=20000)
    return p


if __name__ == "__main__":
    c = pd.read_csv(region.CELLS)
    print(f"{len(c)} cells")
    xy = c[["x_utm", "y_utm"]].values
    rows, fig_data = [], {}
    for col in ("agb_main", "rh98_mean"):
        v = c[col].values.astype(float)
        h, g, n = empirical(xy, v, EDGES)
        rows += [dict(variable=col, lag_m=a, semivariance=b, pairs=k) for a, b, k in zip(h, g, n)]
        var = v.var()
        res = {}
        for name, m, k in (("spherical", spherical, 1), ("exponential", exponential, 3)):
            nug, sill, a = fit(m, h, g, n, var)
            res[name] = (nug, sill, a, a * k)
            print(f"{col:10s} {name:11s} nugget {nug:9.2f} sill {sill:9.2f} (variance {var:9.2f})  "
                  f"range a={a/1000:6.1f} km  practical range {a*k/1000:6.1f} km")
        fig_data[col] = (h, g, n, res, var)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (col, (h, g, n, res, var)) in zip(axs, fig_data.items()):
        ax.scatter(h / 1000, g, s=np.clip(n / n.max() * 80, 8, 80), c="#2a6f4f", zorder=3)
        hh = np.linspace(1, EDGES[-1], 400)
        ax.plot(hh / 1000, spherical(hh, *res["spherical"][:3]), c="#d62728", label=f"spherical, range {res['spherical'][3]/1000:.0f} km")
        ax.plot(hh / 1000, exponential(hh, *res["exponential"][:3]), c="#1f77b4", label=f"exponential, practical range {res['exponential'][3]/1000:.0f} km")
        ax.axhline(var, c="#888", ls=":", label="total variance")
        ax.set_xlabel("distance between cells (km)"); ax.set_ylabel("semivariance")
        ax.set_title(col + " (all label cells)"); ax.legend(fontsize=8); ax.set_xlim(0, EDGES[-1] / 1000)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110); print(OUT_PNG)
