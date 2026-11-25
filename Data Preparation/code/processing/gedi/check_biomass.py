"""Checks for code/processing/gedi/biomass.py. Run: python code/processing/gedi/check_biomass.py"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomass as b


def close(a, e, tol=0.5):
    assert abs(a - e) <= tol, f"{a} != {e}"


# main: exp(3.9042 + 0.0858 h), hand-computed
for h, e in [(5, 76.2), (10, 117.0), (15, 179.7), (20, 275.9), (30, 650.7)]:
    close(float(b.agb_main(h)), e, 0.6)
close(float(b.agb_main(0)), math.exp(3.9042), 1e-9)          # 49.6 at zero height

# sensitivity: 2.572 h^1.5191
for h, e in [(5, 29.7), (10, 85.0), (15, 157.4), (20, 243.6), (30, 451.0)]:
    close(float(b.agb_sensitivity(h)), e, 0.6)
assert float(b.agb_sensitivity(0)) == 0.0

# cap and range: above the cap or below 0 -> NaN, for both
for f in (b.agb_main, b.agb_sensitivity):
    assert np.isnan(f(b.H_CAP + 0.01)) and np.isnan(f(-0.1)) and not np.isnan(f(b.H_CAP))
    assert np.isnan(f(np.nan))

# the cap keeps main at or below the paper's maximum AGB (910.5 Mg/ha), and just above it would not
assert float(b.agb_main(b.H_CAP)) <= 910.5
assert math.exp(3.9042 + 0.0858 * (b.H_CAP + 0.1)) > 910.5

# arrays and carbon
a = b.agb_main(np.array([10.0, 20.0, 40.0]))
assert a.shape == (3,) and np.isnan(a[2])
close(float(b.carbon(100.0)), 45.1, 1e-9)
print("all biomass tests passed")
