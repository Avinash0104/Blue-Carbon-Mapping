"""Checks for the robustness evaluations: label-noise floor, seed variance, spatial hold-out.
Run (from the Model Analysis folder): python evaluation/check_robustness.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_floor as NF
import predictions as P
import seed_variance as SV
import spatial as S


def raises(fn, text, exc=AssertionError):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected '{text}' in: {e}"
        return
    raise AssertionError(f"expected an error containing '{text}'")


# --- noise floor: hand-computed ---------------------------------------------------------------------------------------------
assert abs(NF.sigma_of([10, -10, 20, -20]) - np.sqrt(125)) < 1e-12                     # mean d^2 = 250, sigma = sqrt(250 / 2)
f = NF.floor_for([100., 200], [1, 2], 10.0)                                              # (100 + 50) / 2 = 75
assert abs(f["rmse_floor"] - np.sqrt(75)) < 1e-12 and abs(f["r2_ceiling"] - (1 - 75 / 2500)) < 1e-12
shots = pd.DataFrame({"cell_row": [0, 0, 1, 2, 2, 2, 3, 3], "cell_col": 0,
                      "month": ["2020-02", "2020-01", "2020-01", "2020-01", "2020-02", "2020-03", "2021-01", "2021-06"],
                      "agb_main": [110., 100, 50, 1, 2, 3, 300, 280]})
cl = pd.DataFrame({"cell_row": [0, 1, 2, 3], "cell_col": 0, "x_utm": [0., 5000, 10000, 15000], "y_utm": 0.0})
pt = NF.pairs_table(shots, cl)
assert pt[["cell_row", "a", "b"]].values.tolist() == [[0, 100, 110], [3, 300, 280]]     # only exactly-two cells; ordered by month
# the formula against a simulation: a model that knows the truth has RMSE equal to the floor
rng = np.random.default_rng(0)
n = 40000
truth = rng.uniform(50, 500, n)
nsh = np.where(rng.random(n) < 0.3, 2, 1)
sh = [pd.DataFrame({"cell_row": np.arange(n)[nsh >= j], "cell_col": 0, "month": f"2020-0{j}",
                    "agb_main": truth[nsh >= j] + rng.normal(0, 50, (nsh >= j).sum())}) for j in (1, 2)]
shots_sim = pd.concat(sh, ignore_index=True)
label = shots_sim.groupby("cell_row").agb_main.mean().reindex(range(n)).to_numpy()
cells_sim = pd.DataFrame({"cell_row": np.arange(n), "cell_col": 0, "agb_main": label, "n_shots": nsh,
                          "x_utm": rng.uniform(0, 100000, n), "y_utm": rng.uniform(0, 100000, n)})
split_sim = cells_sim[P.KEY].assign(set="test")
fl = NF.compute(cells_sim, split_sim, shots=shots_sim, n_boot=100).iloc[0]
assert abs(fl.sigma - 50) < 3, fl.sigma                                                  # recovers the simulated noise
true_rmse = np.sqrt(((truth - label) ** 2).mean())
assert abs(fl.rmse_floor - true_rmse) / true_rmse < 0.05, (fl.rmse_floor, true_rmse)     # floor = RMSE of the perfect model
assert fl.floor_ci_lo < fl.rmse_floor < fl.floor_ci_hi and "noise sigma" in NF.describe(fl.to_frame().T)
assert np.isnan(NF.estimate(NF.pairs_table(shots.iloc[:1], cl))["sigma"])                # no pairs: NaN, not a crash

# --- seed variance ------------------------------------------------------------------------------------------------------------
sp = pd.DataFrame({"cell_row": np.arange(300), "cell_col": 0, "set": ["train"] * 200 + ["val"] * 50 + ["test"] * 50})
cl2 = sp[P.KEY].assign(agb_main=np.linspace(100, 400, 300), x_utm=np.arange(300) * 100.0, y_utm=0.0)


def run(scale, seed):
    r = np.random.default_rng(seed)
    return P.with_labels(sp.assign(pred=cl2.agb_main + r.normal(0, scale, 300)), cl2, sp)


groups = {"a": [run(10, s) for s in range(4)], "b": [run(30, 10 + s) for s in range(4)], "c": [run(10.5, 20 + s) for s in range(4)]}
ps = SV.per_seed(groups)
assert len(ps) == 12 and set(ps.model) == {"a", "b", "c"} and (ps.n == 50).all()
sm = SV.summary(ps).set_index("model")
assert sm.n_seeds["a"] == 4 and abs(sm.rmse_mean["a"] - ps[ps.model == "a"].rmse.mean()) < 1e-12
assert abs(sm.rmse_sd["a"] - ps[ps.model == "a"].rmse.std(ddof=1)) < 1e-12 and sm.rmse_min["a"] <= sm.rmse_max["a"]
pt2 = SV.pair_table(ps, ["a:b", "a:c"]).set_index("model_b")
assert pt2.share_a_lower_rmse["b"] == 1 and bool(pt2.order_never_changes["b"]) and pt2.rmse_diff_a_minus_b["b"] < 0
assert pt2.diff_over_sd["b"] > 2                                                          # a clear difference
assert pt2.diff_over_sd["c"] < pt2.diff_over_sd["b"]                                      # a near-tie is judged smaller
assert "diff/sd" in SV.text(SV.summary(ps), SV.pair_table(ps, ["a:b"]))
raises(lambda: SV.pair_table(ps, ["a:zzz"]), "not one of the models", SystemExit)
other = sp.assign(set=["train"] * 200 + ["val"] * 51 + ["test"] * 49)
raises(lambda: SV.per_seed({"a": [run(10, 0)], "x": [P.with_labels(other.assign(pred=cl2.agb_main), cl2, other)]}), "same test cells", SystemExit)
one = SV.summary(SV.per_seed({"a": [run(10, 0)]}))
assert np.isnan(one.rmse_sd[0])                                                           # one seed: no spread, no crash

# --- spatial folds: construction ---------------------------------------------------------------------------------------------------
rng = np.random.default_rng(3)
nc = 3000
cells = pd.DataFrame({"cell_row": np.arange(nc), "cell_col": 0, "agb_main": rng.uniform(50, 500, nc),
                      "x_utm": rng.uniform(0, 80000, nc), "y_utm": rng.uniform(0, 80000, nc)})
folds = S.make_folds(cells, k=5, block_m=10000, buffer_m=5000, seed=42)
assert len(folds) == nc and sorted(folds.fold.unique()) == [0, 1, 2, 3, 4]
assert (folds.groupby("block").fold.nunique() == 1).all()                                  # a block is never split across folds
sizes = folds.fold.value_counts()
assert sizes.max() - sizes.min() < 0.15 * sizes.mean()                                     # balanced
assert folds.equals(S.make_folds(cells, 5, 10000, 5000, 42)) and not folds.fold.equals(S.make_folds(cells, 5, 10000, 5000, 7).fold)
xy = cells.sort_values(P.KEY)[["x_utm", "y_utm"]].to_numpy()
for f in range(5):
    r = folds[f"role_{f}"].to_numpy()
    assert ((r == "test") == (folds.fold.to_numpy() == f)).all()                           # test = the fold, nothing else
    tr, te = xy[r == "train"], xy[r == "test"]
    dmin = np.sqrt(((tr[:, None, :] - te[None, ::7, :]) ** 2).sum(-1)).min()
    assert dmin >= 5000, dmin                                                              # no train cell within the buffer of a test cell
    bu = xy[r == "buffer"]
    assert len(bu) > 0 and np.sqrt(((bu[:, None, :] - te[None, :, :]) ** 2).sum(-1)).min(1).max() < 5000   # buffer cells are all close
assert (folds[[f"role_{f}" for f in range(5)]].eq("test").sum(axis=1) == 1).all()           # every cell is tested exactly once
fs = S.fold_summary(cells, folds)
assert (fs.n_test + fs.n_train + fs.n_buffer == nc).all()
dist = S.nearest_train_distance(cells, folds, [0, 1, 2, 3, 4])
assert len(dist) == nc and dist.min() >= 5000

# --- spatial null model: only the fold's train labels -----------------------------------------------------------------------------
nm = S.null_make(cells, folds)
assert len(nm) == nc
f0 = folds.merge(cells, on=P.KEY)
assert abs(nm[nm.fold == 0].pred.iloc[0] - f0.loc[f0.role_0 == "train", "agb_main"].mean()) < 1e-9
c2 = cells.copy()
c2.loc[(f0.role_0 != "train").to_numpy(), "agb_main"] = 99999                              # change test and buffer labels
assert abs(S.null_make(c2, folds)[lambda d: d.fold == 0].pred.iloc[0] - nm[nm.fold == 0].pred.iloc[0]) < 1e-9

# --- spatial predictions: validation, round trip, hash guard ----------------------------------------------------------------------------
good = folds[P.KEY + ["fold"]].assign(pred=cells.agb_main)
S.validate(good, folds)
raises(lambda: S.validate(good.assign(fold=(good.fold + 1) % 5), folds), "disagrees")
raises(lambda: S.validate(good.assign(pred=np.nan), folds), "NaN")
raises(lambda: S.validate(pd.concat([good, good.iloc[[0]]]), folds), "duplicate")
raises(lambda: S.validate(good.drop(columns="pred"), folds), "missing columns")
assert S.complete_folds(good, folds) == [0, 1, 2, 3, 4] and S.complete_folds(good[good.fold != 2], folds) == [0, 1, 3, 4]
assert S.complete_folds(good.iloc[:-1], folds) == [f for f in range(5) if f != good.iloc[-1].fold]       # one missing cell: fold incomplete
with tempfile.TemporaryDirectory() as tmp:
    fp = Path(tmp) / "folds.csv"
    folds.to_csv(fp, index=False)
    S.write(good, "m", {"seed": 1}, pred_dir=Path(tmp), folds=folds, folds_path=fp)
    assert S.read("m", Path(tmp), fp).shape == good.shape
    import json
    side = Path(tmp) / "m.json"
    d = json.loads(side.read_text())
    d["folds_sha256"] = "0" * 64
    side.write_text(json.dumps(d))
    raises(lambda: S.read("m", Path(tmp), fp), "different folds file")

# --- spatial evaluation end to end ------------------------------------------------------------------------------------------------------------
split = cells[P.KEY].assign(set=rng.choice(["train", "val", "test"], nc, p=[0.7, 0.15, 0.15]))
noise = rng.normal(0, 20, nc)
sp_models = {"good": good.assign(pred=cells.agb_main + noise), "null": nm,
             "partial": good[good.fold.isin([0, 1])].assign(pred=cells.agb_main[good.fold.isin([0, 1]).to_numpy()] + 80)}
rnd = {n: P.with_labels(split.assign(pred=cells.agb_main + rng.normal(0, s, nc)), cells, split) for n, s in [("good", 10), ("null", 130)]}
res, info = S.evaluate({k: v for k, v in sp_models.items() if k != "partial"}, cells, folds, block_m=10000, n_boot=100, seed=1,
                       pairs=["good:null"], random_models=rnd, split=split)
mt = res["metrics"].set_index(["model", "metric"])
assert info["folds_used"] == [0, 1, 2, 3, 4] and info["n_cells"] == nc and mt.n.iloc[0] == nc
assert mt.estimate["good", "rmse"] < mt.estimate["null", "rmse"] and abs(mt.estimate["good", "rmse"] - 20) < 2
assert not np.isnan(mt.ci_lo["good", "rmse"]) and info["distance_to_train_m"]["min"] >= 5000
assert res["by_fold"].groupby("model").size().tolist() == [5, 5] and res["paired"].share_a_rmse_lower.iloc[0] == 1
vr = res["vs_random"].set_index("model")
assert vr.n["good"] == int(((split.set == "test")).sum()) and vr.rmse_spatial["good"] > vr.rmse_random["good"]    # spatial worse than random here
assert vr.rank_random["good"] == 1 and vr.rank_spatial["good"] == 1
txt = S.text(res, info)
assert "SPATIAL HOLD-OUT" in txt and "random-split model vs spatial model" in txt and "ranking is the same" in txt
# a model that covers only some folds: everyone is scored on the folds all models cover
res2, info2 = S.evaluate(sp_models, cells, folds, block_m=10000, n_boot=100, seed=1, split=split)
assert info2["folds_used"] == [0, 1] and (res2["metrics"].n == (folds.fold.isin([0, 1])).sum()).all()
raises(lambda: S.evaluate({"x": good[good.fold == 0], "y": good[good.fold == 1]}, cells, folds, split=split), "no fold is complete", SystemExit)
raises(lambda: S.evaluate({"good": good}, cells, folds, pairs=["good:zzz"], split=split), "not one of the models", SystemExit)
# too few blocks: estimates stay, intervals are withheld
res3, _ = S.evaluate({"good": good}, cells, folds, block_m=10 ** 6, n_boot=20, seed=1, split=split)
assert res3["metrics"].ci_lo.isna().all() and res3["metrics"].estimate.notna().any()

# --- the real files, if present ---------------------------------------------------------------------------------------------------------------
if S.FOLDS_FILE.exists() and (P.LABELS / "gedi_cells.csv").exists():
    rc, rf = pd.read_csv(P.LABELS / "gedi_cells.csv"), S.load_folds()
    assert len(rf) == len(rc) and rf.fold.nunique() == 5 and (rf.groupby("block").fold.nunique() == 1).all()
    assert S.make_folds(rc, **{k: v for k, v in S.DEFAULTS.items()}).equals(rf), "the folds file no longer matches the code"
    print("real folds: reproducible; fold sizes", rf.fold.value_counts().sort_index().tolist())

print("all robustness checks passed")
