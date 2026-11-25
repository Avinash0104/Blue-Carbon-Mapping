"""Checks for code/processing/gedi/cells.py. Run: python code/processing/gedi/check_cells.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells as c

# weights: 3 x 3, non-negative, sum to 1, for every cell pattern (and a spread of positions)
for i in list(range(0, 8)) + [1001, 14479]:
    for j in list(range(0, 8)) + [999, 17597]:
        r0, c0, w = c.window(i, j)
        assert w.shape == (3, 3) and abs(w.sum() - 1) < 1e-12 and (w >= 0).all()

# hand-computed: cell 0 spans 0-25 m -> pixels 0,1,2 with overlap 10,10,5 of 25 -> (.4,.4,.2)
r0, c0, w = c.window(0, 0)
assert (r0, c0) == (0, 0) and np.allclose(w, np.outer([.4, .4, .2], [.4, .4, .2]))
# cell 1 spans 25-50 m -> pixels 2,3,4 with 5,10,10 -> (.2,.4,.4)
r0, c0, w = c.window(1, 1)
assert (r0, c0) == (2, 2) and np.allclose(w, np.outer([.2, .4, .4], [.2, .4, .4]))
# mixed pattern: row index 1, column index 2
r0, c0, w = c.window(1, 2)
assert (r0, c0) == (2, 5) and np.allclose(w, np.outer([.2, .4, .4], [.4, .4, .2]))

# the window of the last cell stays inside the 10 m grid
r0, c0, _ = c.window(c.GEDI_SHAPE[0] - 1, c.GEDI_SHAPE[1] - 1)
assert r0 + 3 <= c.PIX_SHAPE[0] and c0 + 3 <= c.PIX_SHAPE[1]

# the centre pixel lies inside the window, and the centre xy is on the 25 m cell centre
for i in range(12):
    for j in range(12):
        r0, c0, _ = c.window(i, j)
        cr, cc = c.centre_pixel(i, j)
        assert r0 <= cr < r0 + 3 and c0 <= cc < c0 + 3
assert c.centre_pixel(0, 0) == (1, 1) and c.centre_pixel(1, 1) == (3, 3)
assert c.centre_xy(0, 0) == (312062.5, 816487.5)

# cell_mean: constant stack -> constant; a gradient gives the hand-computed weighted mean; NaNs are skipped
stack = np.full((2, 40, 40), 7.0)
assert np.allclose(c.cell_mean(stack, 3, 4), 7.0)
g = np.tile(np.arange(40, dtype=float), (40, 1))[None]       # value = column index
r0, c0, w = c.window(1, 1)
assert abs(c.cell_mean(g, 1, 1)[0] - (0.2 * 2 + 0.4 * 3 + 0.4 * 4)) < 1e-12      # columns 2,3,4 with .2,.4,.4
s2 = np.full((1, 40, 40), 5.0)
s2[0, 2, 2] = np.nan                                           # one pixel of cell (1,1) missing
assert abs(c.cell_mean(s2, 1, 1)[0] - 5.0) < 1e-12
s3 = np.full((1, 40, 40), np.nan)
assert np.isnan(c.cell_mean(s3, 1, 1)[0])
print("all cell tests passed")
