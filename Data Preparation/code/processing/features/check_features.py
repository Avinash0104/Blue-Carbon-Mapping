"""Checks for build_cell_features.py and channels.py. Run: python code/processing/features/check_features.py

Uses small synthetic GeoTIFFs in a temporary folder (4 pieces, with a window that straddles all of them) and an
independent area calculation (1 m sub-samples of each 25 m cell) to test the weighted means.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gedi"))
import build_cell_features as b
import cells
import channels

ND = -32768

# --- channels.py -----------------------------------------------------------------------------------------
assert len(channels.CHANNELS) == 21
assert [s["name"] for s in channels.SOURCES] == ["static", "composite", "derived_composite", "derived_static"]
assert "slope_deg" not in [c for s in channels.SOURCES for c, _ in s["use"].values()]

# --- synthetic data: full image 40 rows x 60 cols, 3 int16 bands as in the static export -----------------------
R, C = 40, 60
rr, cc = np.mgrid[0:R, 0:C]
full = np.stack([3 * cc + rr, 500 + cc, np.full((R, C), 7)]).astype("int16")      # DEM_x10, HH_x100, HV_x100
full[0, 2, 2] = ND                      # one missing DEM pixel inside cell (1, 1)
full[1, 30:33, 30:33] = ND              # whole window of cell (12, 12) missing in HH
SRC = {"name": "t", "pattern": "t/*.tif", "nodata": ND, "file_bands": ["DEM_x10", "HH_x100", "HV_x100"],
       "use": {"DEM_x10": ("DEM", 10), "HH_x100": ("HH", 100)}}      # HV_x100 present but not used


def write_pieces(folder, row_cuts=(0, 18, 40), col_cuts=(0, 18, 60)):
    folder.mkdir(parents=True, exist_ok=True)
    for a in range(len(row_cuts) - 1):
        for k in range(len(col_cuts) - 1):
            r0, r1, c0, c1 = row_cuts[a], row_cuts[a + 1], col_cuts[k], col_cuts[k + 1]
            prof = dict(driver="GTiff", height=r1 - r0, width=c1 - c0, count=3, dtype="int16", crs="EPSG:32650",
                        transform=from_origin(cells.X0 + 10 * c0, cells.Y0 - 10 * r0, 10, 10))
            with rasterio.open(folder / f"p_{a}_{k}.tif", "w", **prof) as d:
                d.write(full[:, r0:r1, c0:c1])


def expected(i, j, band, scale):
    """Independent weighted mean: 1 m sub-samples of the 25 m cell, each taking the value of the 10 m pixel it is in."""
    ys, xs = np.mgrid[0:25, 0:25]
    prow, pcol = (25 * i + ys) // 10, (25 * j + xs) // 10
    v = full[band, prow, pcol].astype(float)
    v[v == ND] = np.nan
    v = v / scale
    return np.nan if np.isnan(v).all() else np.nanmean(v)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_pieces(root / "t")
    idx = pd.DataFrame([(i, j) for i, j in [(0, 0), (1, 1), (7, 7), (7, 2), (2, 7), (12, 12), (14, 22), (3, 20), (5, 5)]],
                       columns=["cell_row", "cell_col"])
    out = b.extract(idx, [SRC], root)
    assert list(out.columns) == ["HH", "DEM"], "columns follow the CHANNELS order, not the file order"

    # weighted means against the independent area calculation (incl. the cell on all four pieces: (7, 7))
    for k, (i, j) in enumerate(zip(idx.cell_row, idx.cell_col)):
        exp_dem, exp_hh = expected(i, j, 0, 10), expected(i, j, 1, 100)
        np.testing.assert_allclose(out.DEM[k], exp_dem, rtol=0, atol=1e-9, equal_nan=True, err_msg=f"DEM cell {(i, j)}")
        np.testing.assert_allclose(out.HH[k], exp_hh, rtol=0, atol=1e-9, equal_nan=True, err_msg=f"HH cell {(i, j)}")
    # (7, 7) window is rows 17-19, cols 17-19: it crosses the cut at 18 in both directions
    assert cells.window(7, 7)[:2] == (17, 17)
    # missing pixel: skipped and the weights renormalised (cell (1, 1) DEM); all-NaN window gives NaN (cell (12, 12) HH)
    k11, k1212 = 1, 5
    assert abs(out.DEM[k11] - expected(1, 1, 0, 10)) < 1e-9 and np.isfinite(out.DEM[k11])
    assert np.isnan(out.HH[k1212]) and np.isfinite(out.DEM[k1212])
    # scale applied: DEM is /10, HH is /100 (a hand value: cell (0, 0), rows/cols 0-2, weights .4 .4 .2)
    w = np.array([.4, .4, .2])
    hand_hh = sum(w[a] * w[c] * (500 + c) / 100 for a in range(3) for c in range(3))
    assert abs(out.HH[0] - hand_hh) < 1e-9

    # same rule as cells.cell_mean on the whole image
    fl = full.astype(float)
    fl[fl == ND] = np.nan
    ref = cells.cell_mean(fl[:1] / 10, 7, 7)[0]
    assert abs(out.DEM[2] - ref) < 1e-9

    # row order follows the input, and a second run is identical
    rev = b.extract(idx.iloc[::-1].reset_index(drop=True), [SRC], root)
    np.testing.assert_array_equal(rev.to_numpy(), out.to_numpy()[::-1])
    np.testing.assert_array_equal(b.extract(idx, [SRC], root).to_numpy(), out.to_numpy())

    # a source with no files is skipped; overlapping pieces fail loudly
    none = b.extract(idx, [{**SRC, "pattern": "nothing/*.tif"}], root)
    assert none.shape == (len(idx), 0)
    write_pieces(root / "t", row_cuts=(0, 40), col_cuts=(0, 60))          # one big piece over the four others
    try:
        b.extract(idx, [SRC], root)
        raise AssertionError("overlapping pieces were not detected")
    except AssertionError as e:
        assert "overlap" in str(e)

def raises_exit(fn, text):
    try:
        fn()
    except SystemExit as e:
        assert text in str(e), f"unexpected message: {e}"
        return
    raise AssertionError(f"expected SystemExit containing '{text}'")


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_pieces(root / "t")
    # last rows / columns of the file set: cell (15, 0) has window rows 37-39, cell (14, 22) cols 55-57
    edge = pd.DataFrame([(15, 0), (14, 22), (15, 22)], columns=["cell_row", "cell_col"])
    e = b.extract(edge, [SRC], root)
    for k, (i, j) in enumerate(zip(edge.cell_row, edge.cell_col)):
        assert abs(e.DEM[k] - expected(i, j, 0, 10)) < 1e-9, (i, j)
    # a window that runs past the outer edge of the files: loud failure, not silent NaN
    raises_exit(lambda: b.extract(pd.DataFrame({"cell_row": [16], "cell_col": [0]}), [SRC], root), "no file covers")
    # a missing piece: the cells that lie in it must stop the run (otherwise the channel would look complete)
    (root / "t" / "p_1_1.tif").unlink()
    raises_exit(lambda: b.extract(idx, [SRC], root), "no file covers")

with tempfile.TemporaryDirectory() as tmp:
    # float32 derived-style files: nodata -9999, band names set in the files, one band not used (slope-like)
    root = Path(tmp)
    (root / "d").mkdir()
    names = ["HVHH_db", "slope_deg"]
    dsrc = {"name": "d", "pattern": "d/*.tif", "nodata": -9999.0, "file_bands": names, "use": {"HVHH_db": ("HVHH_db", 1)}}
    val = (np.sin(rr / 5.0) + cc / 10.0).astype("float32")
    val[10:13, 10:13] = -9999.0
    for k, (c0, c1) in enumerate([(0, 24), (24, 60)]):
        prof = dict(driver="GTiff", height=R, width=c1 - c0, count=2, dtype="float32", crs="EPSG:32650",
                    transform=from_origin(cells.X0 + 10 * c0, cells.Y0, 10, 10))
        with rasterio.open(root / "d" / f"d{k}.tif", "w", **prof) as f:
            f.write(np.stack([val[:, c0:c1], np.zeros((R, c1 - c0), "float32")]))
            f.descriptions = tuple(names)
    # (4, 9): window cols 22-24 straddles the cut at col 24; (4, 4): window rows/cols 10-12 is all nodata
    cidx = pd.DataFrame([(2, 2), (4, 9), (4, 4), (7, 20)], columns=["cell_row", "cell_col"])
    assert cells.window(4, 9)[:2] == (10, 22) and cells.window(4, 4)[:2] == (10, 10)
    o = b.extract(cidx, [dsrc], root)
    for k, (i, j) in enumerate(zip(cidx.cell_row, cidx.cell_col)):
        ys, xs = np.mgrid[0:25, 0:25]
        v = val[(25 * i + ys) // 10, (25 * j + xs) // 10].astype(float)
        v[v == -9999.0] = np.nan
        ref = np.nan if np.isnan(v).all() else np.nanmean(v)
        assert (np.isnan(ref) and np.isnan(o.HVHH_db[k])) or abs(o.HVHH_db[k] - ref) < 1e-6, (i, j)
    assert np.isnan(o.HVHH_db[2]), "window fully on nodata gives NaN"
    # band names in the file that disagree with the config stop the run
    bad = {**dsrc, "file_bands": ["HVHH_db", "something_else"]}
    raises_exit(lambda: b.extract(cidx, [bad], root), "differ from")

with tempfile.TemporaryDirectory() as tmp:
    # the main logic on a tiny made-up label set: partial and final naming, no coordinates, write-once statistics
    root = Path(tmp)
    write_pieces(root / "t")
    lab, out = root / "labels", root / "out"
    lab.mkdir()
    pairs = [(i, j) for i in range(0, 15, 2) for j in range(0, 23, 3)][:30]
    sets = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    lc = pd.DataFrame({"cell_row": [p[0] for p in pairs], "cell_col": [p[1] for p in pairs],
                       "win_row0": [cells.window(*p)[0] for p in pairs], "win_col0": [cells.window(*p)[1] for p in pairs],
                       "n_shots": 1, "rh98_mean": 10.0, "agb_main": 200.0, "agb_sens": 150.0, "carbon_main": 90.0,
                       "x_utm": 1.0, "y_utm": 2.0})
    lc.to_csv(lab / "gedi_cells.csv", index=False)
    pd.DataFrame({"cell_row": lc.cell_row, "cell_col": lc.cell_col, "set": sets}).to_csv(lab / "gedi_split.csv", index=False)

    t = b.main(sources=[SRC], root=root, labels=lab, out=out)          # 2 of 21 channels: partial
    assert (out / "cell_features_partial.csv").exists() and (out / "norm_stats_partial.json").exists()
    assert not (out / "cell_features.csv").exists() and not (out / "norm_stats.json").exists()
    back = pd.read_csv(out / "cell_features_partial.csv")
    assert len(back) == 30 and list(back.columns) == b.META + ["HH", "DEM"]
    assert not any(c.endswith("utm") for c in back.columns)
    assert (back[["cell_row", "cell_col"]].to_numpy() == lc[["cell_row", "cell_col"]].to_numpy()).all()   # row order kept
    b.main(sources=[SRC], root=root, labels=lab, out=out)              # partial files may be overwritten

    real = b.CHANNELS
    b.CHANNELS = ["HH", "DEM"]                                         # pretend these are all the channels: final mode
    try:
        b.main(sources=[SRC], root=root, labels=lab, out=out)
        assert (out / "cell_features.csv").exists() and (out / "norm_stats.json").exists()
        raises_exit(lambda: b.main(sources=[SRC], root=root, labels=lab, out=out), "write-once")
    finally:
        b.CHANNELS = real

# --- normalisation statistics: train rows only ------------------------------------------------------------------
rng = np.random.default_rng(0)
t = pd.DataFrame({"set": ["train"] * 60 + ["val"] * 20 + ["test"] * 20, "A": rng.normal(5, 2, 100), "B": rng.normal(0, 1, 100)})
t.loc[3, "A"] = np.nan
s1 = b.norm_stats(t, ["A", "B"])
tr = t[t.set == "train"]
assert abs(s1["A"]["mean"] - np.nanmean(tr.A)) < 1e-12 and abs(s1["A"]["std"] - np.nanstd(tr.A)) < 1e-12   # ddof = 0
assert s1["A"]["n_valid"] == 59 and s1["B"]["n_valid"] == 60
t2 = t.copy()
t2.loc[t2.set != "train", ["A", "B"]] = 1e6                       # change val and test: statistics must not move
assert b.norm_stats(t2, ["A", "B"]) == s1
t3 = t.copy()
t3["A"] = 4.0
try:
    b.norm_stats(t3, ["A"])
    raise AssertionError("zero std was not rejected")
except AssertionError as e:
    assert "zero std" in str(e)

# --- normalise.py: z-score with the stored statistics, missing values -> training mean (0) ----------------------
import json

import normalise as nm

with tempfile.TemporaryDirectory() as tmp:
    sp_ = Path(tmp) / "stats.json"
    sp_.write_text(json.dumps({"channels": ["A", "B"], "stats": s1}))
    z = nm.normalise(t, sp_)
    assert list(z.columns) == ["A", "B"] and z.shape == (100, 2) and z.notna().all().all()
    assert z.loc[3, "A"] == 0.0                                          # the NaN became the training mean
    ok = t.index != 3
    assert np.allclose(z.loc[ok, "A"], (t.loc[ok, "A"] - s1["A"]["mean"]) / s1["A"]["std"])
    zt = z[t.set == "train"]
    assert abs(zt["B"].mean()) < 1e-9 and abs(zt["B"].std(ddof=0) - 1) < 1e-9     # train z-scores: mean 0, std 1
    try:
        nm.normalise(t[["A"]], sp_)
        raise AssertionError("a missing channel was not detected")
    except AssertionError as e:
        assert "missing from the data" in str(e)

print("all feature tests passed")
