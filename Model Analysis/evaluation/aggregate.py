"""Aggregated accuracy on test cells (Project.md 5.1, "aggregated 0.25-1 ha accuracy", test-cell version).

Test cells are grouped into squares of side `side` m (50 m = 0.25 ha, 100 m = 1 ha, aligned to the UTM grid). A square is
kept only if it holds at least `min_cells` test cells. For each kept square: mean observed and mean predicted biomass
of its test cells; the usual metrics are then computed over the squares.

SMALL SAMPLE WARNING: GEDI cells lie on thin tracks and the test set is a random 15%, so very few squares hold two or
more test cells (on the real split: 20 squares at 50 m, 92 at 100 m, almost none with three). A square mean is then an
average of 2-3 cells, which removes only part of the GEDI noise. Treat this as an indication, not as the 0.25-1 ha
accuracy of the map. The accuracy of the map at that scale is checked against independent data (Project.md 5.4-5.5).
"""
import numpy as np

SIDES = {"0.25 ha (50 m)": 50, "1 ha (100 m)": 100}


def group_cells(df, side, min_cells=2):
    """df: test cells with obs, pred, x_utm, y_utm. Returns one row per kept square: n_cells, obs, pred, x_utm, y_utm."""
    assert np.isfinite(df[["obs", "pred", "x_utm", "y_utm"]].to_numpy(float)).all(), "NaN in obs, pred or coordinates"
    g = df.assign(i=np.floor(df.x_utm / side).astype(np.int64), j=np.floor(df.y_utm / side).astype(np.int64))
    out = g.groupby(["i", "j"]).agg(n_cells=("obs", "size"), obs=("obs", "mean"), pred=("pred", "mean"),
                                    x_utm=("x_utm", "mean"), y_utm=("y_utm", "mean")).reset_index(drop=True)
    return out[out.n_cells >= min_cells].reset_index(drop=True)
