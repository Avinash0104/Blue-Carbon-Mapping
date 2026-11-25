"""Checks for the evaluation module. Run (from the Model Analysis folder): python evaluation/check_evaluation.py
Hand-computed values, synthetic data, and a smoke test of the null model on the real split (skipped if files are missing)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as A
import blocks as B
import evaluate as E
import metrics as M
import null_model as N
import predictions as P


def raises(fn, text, exc=AssertionError):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected '{text}' in: {e}"
        return
    raise AssertionError(f"expected an error containing '{text}'")


# --- metrics: a hand-computed case --------------------------------------------------------------------------------
y = np.array([100., 200, 300, 400])
p = np.array([110., 190, 330, 380])          # errors p - y: +10, -10, +30, -20
m = M.all_metrics(y, p)
assert m["n"] == 4
assert abs(m["bias"] - 2.5) < 1e-12 and abs(m["mae"] - 17.5) < 1e-12
assert abs(m["rmse"] - np.sqrt(375)) < 1e-12                     # (100 + 100 + 900 + 400) / 4
assert abs(m["r2"] - (1 - 1500 / 50000)) < 1e-12                 # SSres 1500, SStot 50000
assert abs(m["rel_rmse_pct"] - 100 * np.sqrt(375) / 250) < 1e-12
assert abs(m["r2_pearson"] - np.corrcoef(y, p)[0, 1] ** 2) < 1e-12
assert M.bias(y, y + 5) > 0 and M.bias(y, y - 5) < 0              # over-prediction is positive
assert M.r2(y, y) == 1 and M.rmse(y, y) == 0
assert abs(M.r2(y, np.full(4, y.mean()))) < 1e-12                 # predicting the mean gives R2 = 0
assert M.r2(y, np.full(4, y.mean() + 50)) < 0                     # a worse constant is negative
assert np.isnan(M.r2(np.full(4, 5.0), p)) and np.isnan(M.rmse([], [])) and np.isnan(M.r2_pearson(y, np.ones(4)))
# a biased model: Pearson r^2 stays 1 while R2 does not
assert abs(M.r2_pearson(y, y + 100) - 1) < 1e-12 and M.r2(y, y + 100) < 0.5

# new metrics: bias as % of the mean, calibration slope (hand-computed)
assert abs(m["bias_pct"] - 100 * 2.5 / 250) < 1e-12
assert abs(M.calib_slope(y, 2 * y) - 0.5) < 1e-12 and abs(M.calib_slope(y, 0.5 * y) - 2) < 1e-12 and abs(M.calib_slope(y, y + 7) - 1) < 1e-12
assert np.isnan(M.calib_slope(y, np.ones(4))) and np.isnan(M.bias_pct(np.zeros(3), np.ones(3)))

# --- bins: left-closed, every value in exactly one bin, bins by the observed value -----------------------------------
vals = np.array([0, 99.9, 100, 199, 200, 400, 1000.0])
br = M.by_range(vals, vals + 10)
assert br.bin.tolist() == ["0-100", "100-200", "200-300", "300-400", "400+"]
assert br.n.tolist() == [2, 2, 1, 0, 2] and br.n.sum() == len(vals)           # 100 -> second bin, 400 -> last bin
assert np.isnan(br.bias[3]) and abs(br.bias[0] - 10) < 1e-12 and abs(br.rmse[4] - 10) < 1e-12

# --- blocks -----------------------------------------------------------------------------------------------------------
c = B.block_codes([500, 999.9, 1000, -1, 500], [500, 0, 0, -1, 500], 1000)
assert c[0] == c[1] == c[4] and len({*c}) == 3 and c[2] != c[3]       # x = 1000 starts a new block; negatives work
codes = np.array([0, 0, 0, 1, 1, 2])
reps = B.replicates(codes, 50, seed=3)
for r in reps:                                                          # whole blocks only: block 0 appears 3 cells at a time
    assert (r < 3).sum() % 3 == 0 and (r == 5).sum() >= 0 and ((r >= 3) & (r < 5)).sum() % 2 == 0
assert all(np.array_equal(a, b) for a, b in zip(reps, B.replicates(codes, 50, seed=3)))      # same seed: same resamples
assert not all(np.array_equal(a, b) for a, b in zip(reps, B.replicates(codes, 50, seed=4)))  # other seed: different

rng = np.random.default_rng(0)
yy = rng.uniform(50, 500, 600)
pp = yy + rng.normal(0, 10, 600)
reps1 = B.replicates(np.arange(600), 300, seed=1)                      # every cell its own block
ci = B.confidence(yy, pp, reps1).set_index("metric")
assert ci.ci_lo["rmse"] < 10.5 and ci.ci_hi["rmse"] > 9.5 and ci.ci_lo["rmse"] < ci.estimate["rmse"] < ci.ci_hi["rmse"]
ci_one = B.confidence(yy, pp, B.replicates(np.zeros(600, int), 20, seed=1)).set_index("metric")   # one block only
assert ci_one.ci_lo["rmse"] == ci_one.ci_hi["rmse"] == ci_one.estimate["rmse"]
ci_perfect = B.confidence(yy, yy, reps1).set_index("metric")
assert ci_perfect.ci_lo["rmse"] == 0 == ci_perfect.ci_hi["rmse"]

# --- paired comparison --------------------------------------------------------------------------------------------------
same, share = B.paired(yy, pp, pp, reps1)
assert (same.diff_estimate == 0).all() and (same.ci_lo == 0).all() and (same.ci_hi == 0).all() and share == 0
tab, share = B.paired(yy, yy + 1, yy + 10, reps1)                       # A is better everywhere
t = tab.set_index("metric")
assert abs(t.diff_estimate["rmse"] + 9) < 1e-9 and t.ci_hi["rmse"] < 0 and t.ci_lo["r2"] > 0 and share == 1

# --- uncertainty check ------------------------------------------------------------------------------------------------
sd = rng.uniform(1, 10, 2000)
yy2 = rng.uniform(50, 500, 2000)
tab, rho, mono = M.uncertainty_table(yy2, yy2 + sd * rng.normal(size=2000), sd)
assert tab.n.tolist() == [400] * 5 and rho > 0.4 and mono
tab2, rho2, mono2 = M.uncertainty_table(yy2, yy2 + sd * rng.normal(size=2000), rng.permutation(sd))   # shuffled spread
assert abs(rho2) < 0.1 and not mono2

# --- predictions file: every failure mode, and a valid file -------------------------------------------------------------
split = pd.DataFrame({"cell_row": [0, 0, 1, 1], "cell_col": [0, 1, 0, 1], "set": ["train", "train", "val", "test"]})
good = split.assign(pred=[10., 20, 30, 40])
P.validate(good, split)
raises(lambda: P.validate(pd.concat([good, good.iloc[[0]]]), split), "duplicate")
raises(lambda: P.validate(good.assign(cell_row=[0, 0, 1, 9]), split), "not in the split")
raises(lambda: P.validate(good.assign(set=["train", "train", "val", "val"]), split), "disagrees")
raises(lambda: P.validate(good.assign(pred=[10., np.nan, 30, 40]), split), "NaN")
raises(lambda: P.validate(good.iloc[:3], split), "test cells")
raises(lambda: P.validate(good.assign(pred_std=[1., 1, 1, -1]), split), "pred_std")
raises(lambda: P.validate(good.drop(columns="pred"), split), "missing columns")

# --- null model: the train mean only; val/test labels cannot change it ---------------------------------------------------
cells = split.assign(agb_main=[10., 20, 30, 1000], x_utm=[0., 10, 20, 30], y_utm=[0., 0, 0, 0])
nm, mean = N.make(cells, split)
assert mean == 15 and (nm.pred == 15).all() and nm["set"].tolist() == split["set"].tolist()
nm2, _ = N.make(cells.assign(agb_main=[10., 20, -5, 7]), split)
assert nm2.pred.equals(nm.pred)

# --- write / read round trip, labels joined by key (not by order) ----------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    P.write(good.iloc[::-1], "m", {"seed": 1}, pred_dir=Path(tmp), split=split)
    back = P.read("m", Path(tmp))
    wl = P.with_labels(back, cells, split)
    assert wl.obs.tolist() == [10, 20, 30, 1000] and wl.pred.tolist() == [10, 20, 30, 40]     # sorted by key, labels matched
    assert (Path(tmp) / "m.json").exists()

# --- the whole evaluation on synthetic cells ---------------------------------------------------------------------------------
rng = np.random.default_rng(5)
n = 600
cells = pd.DataFrame({"cell_row": np.arange(n), "cell_col": 0, "agb_main": rng.uniform(50, 500, n),
                      "x_utm": rng.uniform(0, 30000, n), "y_utm": rng.uniform(0, 30000, n)})
split = cells[["cell_row", "cell_col"]].assign(set=["train"] * 400 + ["val"] * 100 + ["test"] * 100)
noise = rng.uniform(5, 40, n)
good = split.assign(pred=cells.agb_main + noise * rng.normal(size=n), pred_std=noise)
null, _ = N.make(cells, split)
models = {"good": P.with_labels(good, cells, split), "null": P.with_labels(null, cells, split)}
reference = cells.merge(split, on=P.KEY)
res, nb = E.evaluate(models, reference, block_m=4000, n_boot=200, seed=1, pairs=["good:null"])
mt = res["metrics"].query("set == 'test'").set_index(["model", "metric"])
assert mt.estimate["good", "rmse"] < mt.estimate["null", "rmse"] and abs(mt.estimate["null", "r2"]) < 0.1
assert mt.n_blocks.iloc[0] == nb["test"] and res["metrics"].query("set == 'val'").shape[0] == 2 * len(M.METRIC_NAMES)
assert res["by_range"].model.nunique() == 2 and res["uncertainty"].model.unique().tolist() == ["good"]
assert res["paired"].share_a_rmse_lower.iloc[0] == 1.0
txt = E.summary(res)
assert "TEST cells" in txt and "NOT independent" in txt and "good vs null" in txt
lk = E.leakage(cells, split)
assert 0 <= lk["within_100m_pct"] <= lk["within_1km_pct"] <= 100
assert "used_for_selection" in res["metrics"].columns and "independent" not in res["metrics"].columns

# models that lack validation rows: no crash, no dependence on model order, the others still scored on all val cells
no_val = models["good"][models["good"]["set"] != "val"]
for order in (["partial", "good", "null"], ["good", "null", "partial"]):
    mm = {"partial": no_val, "good": models["good"], "null": models["null"]}
    res2, _ = E.evaluate({k: mm[k] for k in order}, reference, block_m=4000, n_boot=50, seed=1)
    v = res2["metrics"].query("set == 'val'")
    assert sorted(v.model.unique()) == ["good", "null"] and (v.n == 100).all()          # partial skipped, val complete
    assert sorted(res2["metrics"].query("set == 'test'").model.unique()) == ["good", "null", "partial"]
# a model that does not cover the test cells, and an unknown model in --pairs: clear stop
raises(lambda: E.evaluate({"x": models["good"][models["good"]["set"] != "test"]}, reference, n_boot=10), "test cells", SystemExit)
raises(lambda: E.evaluate(models, reference, n_boot=10, pairs=["good:nope"]), "not one of the models", SystemExit)
# very few blocks: the estimate stays, the interval is withheld (NaN) instead of a misleading zero-width one
res3, nb3 = E.evaluate(models, reference, block_m=10 ** 6, n_boot=20, seed=1, pairs=["good:null"])
assert nb3["test"] == 1
r3 = res3["metrics"].query("set == 'test' and metric == 'rmse'")
assert r3.estimate.notna().all() and r3.ci_lo.isna().all() and r3.ci_hi.isna().all()
assert res3["paired"].ci_lo.isna().all() and res3["paired"].diff_estimate.notna().all()
# no spread in pred_std (e.g. one seed): the uncertainty check is skipped, not crashed
const = models["good"].assign(pred_std=3.0)
res4, _ = E.evaluate({"c": const}, reference, n_boot=20)
assert res4["uncertainty"].empty and np.isnan(M.uncertainty_table(np.arange(50.), np.arange(50.) + 1, np.full(50, 3.0))[1])
assert M.uncertainty_table(np.arange(3.), np.arange(3.), np.arange(3.))[0].empty            # fewer cells than bins
# NaN coordinates are caught when the labels are joined
raises(lambda: P.with_labels(good, cells.assign(x_utm=np.where(cells.cell_row == 0, np.nan, cells.x_utm)), split), "coordinates")
# predictions made with another split are refused at read time
if P.split_sha() is not None:
    with tempfile.TemporaryDirectory() as tmp:
        P.write(good, "m", pred_dir=Path(tmp), split=split)
        P.read("m", Path(tmp))
        side = Path(tmp) / "m.json"
        side.write_text(side.read_text().replace(P.split_sha(), "0" * 64))
        raises(lambda: P.read("m", Path(tmp)), "different split")

# --- aggregated accuracy (test-cell squares) ---------------------------------------------------------------------------------
t = pd.DataFrame({"obs": [100., 200, 300, 50, 70], "pred": [110., 230, 290, 60, 90],
                  "x_utm": [5.0, 30, 70, 5, 6], "y_utm": [5.0, 30, 70, 5, 6]})
g = A.group_cells(t, 50)                                  # squares: (0,0) holds cells 0,1 ; (1,1) holds cell 2 ; (0,0) also 3,4
assert len(g) == 1 and g.n_cells[0] == 4 and abs(g.obs[0] - 105) < 1e-9 and abs(g.pred[0] - 122.5) < 1e-9
g100 = A.group_cells(t, 100)                              # cell 2 at (70,70) joins the same 100 m square: all 5 cells
assert len(g100) == 1 and g100.n_cells[0] == 5 and abs(g100.obs[0] - 144) < 1e-9 and abs(g100.pred[0] - 156) < 1e-9
assert len(A.group_cells(t, 50, min_cells=5)) == 0 and len(A.group_cells(t.iloc[[0, 2]], 50)) == 0   # singles dropped
raises(lambda: A.group_cells(t.assign(pred=np.nan), 50), "NaN")
# two models on 30 clusters of 3 test cells, 5 km apart (30 blocks of 4 km): same squares and resamples for both
rng = np.random.default_rng(0)
rows = [(k * 3 + j, 0, "test", 5000.0 * k + 10 * j, 5000.0 * k + 10 * j) for k in range(30) for j in range(3)]
ref2 = pd.DataFrame(rows, columns=["cell_row", "cell_col", "set", "x_utm", "y_utm"])
ob = rng.uniform(100, 400, len(ref2))
mod = {"a": ref2.assign(obs=ob, pred=ob + rng.normal(0, 20, len(ref2))), "b": ref2.assign(obs=ob, pred=np.full(len(ref2), 250.0))}
res5, _ = E.evaluate(mod, ref2, n_boot=50, seed=1, sets=("test",))
ag5 = res5["aggregated"]
assert not ag5.empty and set(ag5.square) == set(A.SIDES)
for sq, d in ag5.groupby("square"):
    assert d.groupby("model")[["n_squares", "n_cells", "n_blocks"]].first().nunique().eq(1).all()        # same squares for both
    assert d.n_squares.iloc[0] == 30 and d.n_cells.iloc[0] == 90 and d[d.metric == 'rmse'].ci_lo.notna().all()   # 30 blocks >= 20: CI given
ga_b = ag5[(ag5.model == "b") & (ag5.square == "0.25 ha (50 m)") & (ag5.metric == "bias")].estimate.iloc[0]
assert abs(ga_b - (250.0 - ob.reshape(30, 3).mean(1)).mean()) < 1e-9                                      # hand-computed square bias

# --- smoke test on the real split ------------------------------------------------------------------------------------------------
if (P.LABELS / "gedi_cells.csv").exists() and (P.LABELS / "gedi_split.csv").exists():
    rc, rs = pd.read_csv(P.LABELS / "gedi_cells.csv"), P.load_split()
    rn, rmean = N.make(rc, rs)
    w = P.with_labels(rn, rc, rs)
    tst, trn = w[w["set"] == "test"], w[w["set"] == "train"]
    assert len(tst) == 1896 and len(trn) == 8846
    mm = M.all_metrics(tst.obs, tst.pred)
    assert abs(mm["bias"] - (tst.obs.mean() * -1 + rmean)) < 1e-9 and -0.05 < mm["r2"] <= 0 and abs(rmean - trn.obs.mean()) < 1e-9
    print(f"real split: null model test R2 {mm['r2']:.4f}, bias {mm['bias']:+.2f} Mg/ha, RMSE {mm['rmse']:.1f}")

print("all evaluation checks passed")
