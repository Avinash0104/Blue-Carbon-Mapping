"""Geometry of a GEDI label cell on the 10 m grid (shared by labels, pixel features and the U-Net loss).

A GEDI label belongs to a 25 m cell (row i, column j on the 25 m GEDI grid), not to one 10 m pixel. The cell
covers 2.5 x 2.5 pixels of the 10 m grid, so it always overlaps a 3 x 3 window of pixels. Each pixel's weight is
the share of the cell's area that lies on it. The weights are the same shape for every cell on the same
even/odd pattern, and they sum to 1.

Pixel models (RF, XGBoost): features of a cell = weighted mean of the 21 channels over the window.
U-Nets: the loss compares the weighted mean of the predicted biomass over the window with the cell's label.
Both use `weights(i, j)`. The centre pixel is kept only as a reference.

Both grids start at the same corner (312050, 816500); the GEDI cell size is 25 m and the pixel size 10 m.
"""
import numpy as np

CELL = 25               # m, GEDI grid
PIX = 10                # m, mask / satellite grid
X0, Y0 = 312050, 816500
GEDI_SHAPE = (14480, 17598)      # rows, columns of the 25 m grid
PIX_SHAPE = (36200, 43995)       # rows, columns of the 10 m grid


def _axis(k):
    """First pixel index and the 3 overlap weights (sum 1) of 25 m cell index k along one axis."""
    p0 = (CELL * k) // PIX
    lo = CELL * k
    w = []
    for p in range(3):
        a = max(lo, PIX * (p0 + p))
        b = min(lo + CELL, PIX * (p0 + p + 1))
        w.append(max(0, b - a) / CELL)
    return int(p0), np.array(w)


def window(i, j):
    """(row0, col0, weights) for cell (i, j): the 3 x 3 pixel window starting at (row0, col0) and its weights."""
    r0, wr = _axis(i)
    c0, wc = _axis(j)
    return r0, c0, np.outer(wr, wc)


def centre_pixel(i, j):
    """(row, col) of the 10 m pixel at the cell centre."""
    return int((CELL * (i + 0.5)) // PIX), int((CELL * (j + 0.5)) // PIX)


def centre_xy(i, j):
    """UTM 50N (x, y) of the cell centre in metres."""
    return X0 + CELL * (j + 0.5), Y0 - CELL * (i + 0.5)


def cell_mean(stack, i, j):
    """Weighted mean over the cell's window of a (channels, rows, cols) array. Option B for pixel models.

    NaNs (nodata) in the window are skipped and the remaining weights are renormalised;
    a cell with no valid pixel gives NaN. The exported int16 files use -32768 for nodata, which is NOT
    skipped here: convert it to NaN (and apply the scale factors) before calling.
    """
    r0, c0, w = window(i, j)
    block = np.asarray(stack[:, r0:r0 + 3, c0:c0 + 3], dtype="float64")
    ok = np.isfinite(block)
    ww = np.where(ok, w[None], 0.0)
    den = ww.sum((1, 2))
    num = (np.where(ok, block, 0.0) * w[None]).sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
