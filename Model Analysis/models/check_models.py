"""Checks for the fold-ensemble logic in train.py, on small synthetic data (no real data, nothing written to data/).

    python models/check_models.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                                                    # noqa: E402
import xgb as XG                                                     # noqa: E402


def table(n_train=103, n_val=20, n_test=20):
    n = n_train + n_val + n_test
    return pd.DataFrame({"cell_row": np.arange(n), "cell_col": np.arange(n) * 2,
                         "set": ["train"] * n_train + ["val"] * n_val + ["test"] * n_test})


def raises(f, *a, **k):
    try:
        f(*a, **k)
    except (AssertionError, SystemExit):
        return True
    return False


class Stub:
    """Records which training rows each fit saw; predicts the mean of its training target."""
    seen = []

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        self.mean = float(np.mean(y))
        Stub.seen.append(set(X[:, 0].astype(int)))

    def predict(self, X):
        return np.full(len(X), self.mean)


class StubSpec:
    def build(self, params, seed, y):
        return Stub(seed)


def main():
    # cut_folds: balanced, deterministic, every row used, depends on the seed
    f = T.cut_folds(103, 5, 42)
    assert sorted(set(f)) == [0, 1, 2, 3, 4] and np.bincount(f).max() - np.bincount(f).min() <= 1
    assert (f == T.cut_folds(103, 5, 42)).all() and not (f == T.cut_folds(103, 5, 43)).all()

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cv" / "cv_folds.csv"
        df = table()
        a = T.cv_folds(df, 5, 42, path)
        assert path.exists() and (a[df.set == "test"] == -1).all() and (a[df.set != "test"] >= 0).all()
        saved = pd.read_csv(path)
        assert len(saved) == 123 and list(saved.columns) == ["cell_row", "cell_col", "cv_fold"]    # train + validation cells, never test
        # write-once: a later call with another seed reads the file and does not change it
        b = T.cv_folds(df, 5, 7, path)
        assert (a == b).all() and pd.read_csv(path).equals(saved)
        # the file must keep matching the train cells
        moved = df.copy()
        moved.loc[0, "set"] = "test"
        assert raises(T.cv_folds, moved, 5, 42, path), "a changed pool (a train cell moved to test) must be refused"
        assert raises(T.cv_folds, df, 4, 42, path), "a different number of folds must be refused"

    # two cuts: the tuning cut and the performance cut are different shuffles of the same pool, each saved once
    with tempfile.TemporaryDirectory() as d:
        df = table()
        a = T.cv_folds(df, 5, 42, Path(d) / "tuning_folds.csv")
        b = T.cv_folds(df, 5, 43, Path(d) / "cv_folds.csv")
        assert not (a == b).all() and (a[df.set == "test"] == -1).all() and (b[df.set == "test"] == -1).all()
        assert (np.bincount(a[a >= 0]) == np.bincount(b[b >= 0])).all(), "both cuts must have the same fold sizes"

    # row order: with the rows shuffled, every cell still gets the fold the file gives it (matched by key, not by position)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cv_folds.csv"
        df = table()
        base = T.cv_folds(df, 5, 42, path)
        shuf = df.sample(frac=1, random_state=3).reset_index(drop=True)
        got = T.cv_folds(shuf, 5, 42, path)
        want = shuf[T.P.KEY].merge(pd.read_csv(path), on=T.P.KEY, how="left")["cv_fold"].fillna(-1).astype(int).to_numpy()
        assert (got == want).all() and (got[shuf.set == "test"] == -1).all()
        # a frame without test cells is accepted as an out-of-fold predictions file (needs the real split, so use a matching fake one)
        split = df[T.P.KEY + ["set"]]
        oof_df = df[df.set != "test"][T.P.KEY + ["set"]].assign(pred=1.0)
        T.P.write(oof_df, "m_oof", {}, pred_dir=Path(d) / "pred", split=split, require_test=False)
        assert (Path(d) / "pred" / "m_oof.csv").exists()
        assert raises(T.P.write, oof_df, "m_oof2", {}, Path(d) / "pred", split, True), "without require_test a missing test set must fail"

    # ensemble: member k never sees fold k; out-of-fold predictions are one per train cell, from the member that did not see it
    df = table()
    ids = df.index.to_numpy(float)
    X = np.c_[ids, ids * 0]
    y = (ids % 7) * 10 + 5
    tr = (df.set != "test").to_numpy()                                  # the pooled train + validation cells
    fold = T.cut_folds(int(tr.sum()), 5, 42)
    Stub.seen = []
    members, oof = T.ensemble(StubSpec(), {}, [0, 1, 2, 3, 4], X[tr], y[tr], X, fold)
    assert members.shape == (5, len(df)) and oof.shape == (int(tr.sum()),) and np.isfinite(oof).all()
    train_ids = X[tr][:, 0].astype(int)
    for k in range(5):
        left_out = set(train_ids[fold == k])
        assert Stub.seen[k] == set(train_ids) - left_out, f"member {k} must be fitted on exactly the cells outside fold {k}"
        assert (oof[fold == k] == members[k][np.where(tr)[0][fold == k]]).all(), "out-of-fold value must come from member k"
    pred, sd = T.combine(members)
    assert np.allclose(pred, members.mean(0)) and np.allclose(sd, members.std(0))
    assert raises(T.ensemble, StubSpec(), {}, [0, 1, 2], X[tr], y[tr], X, fold), "5 folds but 3 members must be refused"

    # yaml: cv_folds must equal the number of seeds
    assert T.cv_size({"cv_folds": 5, "seeds": [0, 1, 2, 3, 4]}) == 5
    assert raises(T.cv_size, {"cv_folds": 5, "seeds": [0, 1, 2]})
    assert raises(T.cv_size, {"cv_folds": 1, "seeds": [0]})

    # xgboost final(): fixed tuned tree count, no early stopping
    p = XG.final({"early_stopping_rounds": 50, "max_depth": 3}, {"n_trees": 400})
    assert p["n_estimators"] == 400 and "early_stopping_rounds" not in p
    assert T.final_params(XG.SPEC, {"fixed": {"early_stopping_rounds": 50}, "params": {"max_depth": 3}, "n_trees": 400})["n_estimators"] == 400

    # tiny real models end to end (skipped if a library is missing)
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(120, 4))
    yr = 100 + 30 * Xr[:, 0] + rng.normal(size=120)
    fr = T.cut_folds(120, 5, 1)
    try:
        import rf
        m, o = T.ensemble(rf.SPEC, {"n_estimators": 5, "min_samples_leaf": 3, "n_jobs": 1}, [0, 1, 2, 3, 4], Xr, yr, Xr[:10], fr)
        assert m.shape == (5, 10) and np.isfinite(o).all() and np.corrcoef(o, yr)[0, 1] > 0.8
        print("rf ensemble ok")
    except ImportError:
        print("scikit-learn missing: rf end-to-end skipped")
    try:
        import xgboost  # noqa: F401
        m, o = T.ensemble(XG.SPEC, XG.final({"n_estimators": 3000, "early_stopping_rounds": 50, "max_depth": 2, "n_jobs": 1,
                                             "learning_rate": 0.3}, {"n_trees": 20}),
                          [0, 1, 2, 3, 4], Xr, yr, Xr[:10], fr)
        assert m.shape == (5, 10) and np.isfinite(o).all()
        print("xgboost ensemble ok")
    except ImportError:
        print("xgboost missing: xgboost end-to-end skipped")
    print("all model checks passed")


if __name__ == "__main__":
    main()
