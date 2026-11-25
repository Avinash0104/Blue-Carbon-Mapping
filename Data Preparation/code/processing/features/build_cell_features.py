"""Build the per-cell feature table and the normalisation statistics (Project.md 2.3, 2.4, 3, 4).

One row per GEDI label cell (gedi_cells.csv order). Each channel = weighted mean over the cell's 3 x 3 pixel
window (weights from cells.window), with nodata -> NaN and the scale factor applied first. Values are stored in
physical units; normalisation is applied later from the statistics file.

Usage (from the Data Preparation folder):
    python code/processing/features/build_cell_features.py
Reads whichever sources in channels.SOURCES have files. With fewer than 21 channels the result is PARTIAL and is
written to cell_features_partial.csv / norm_stats_partial.json (overwritable). With all 21 it writes
cell_features.csv and norm_stats.json; the statistics file is write-once, like the split.
Writes to data/processed/features/. Not stored: x_utm / y_utm (location must never be a feature).

Normalisation statistics (mean, std with ddof = 0, per channel) come from the TRAIN cells only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gedi"))
import cells
from channels import CHANNELS, SOURCES

ROOT = HERE.parents[2]
LABELS = ROOT / "data" / "processed" / "labels"
OUT = ROOT / "data" / "processed" / "features"
STRIP = 256             # rows of cells handled per read (raw stored dtype: ~170 MB for 12 int16 bands)
META = ["cell_row", "cell_col", "set", "n_shots", "rh98_mean", "agb_main", "agb_sens", "carbon_main"]


def window_mean(block, w):
    """Weighted mean of a (channels, 3, 3) block; NaNs skipped, weights renormalised; all-NaN -> NaN.
    Same rule as cells.cell_mean, for a block that is already cut out."""
    ok = np.isfinite(block)
    ww = np.where(ok, w[None], 0.0)
    den = ww.sum((1, 2))
    num = (np.where(ok, block, 0.0) * w[None]).sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def read_source(src, cell_idx, root):
    """Windows (n_cells, n_channels, 3, 3) of one source, in physical units (NaN = missing), plus the
    channel names. Works for any piece layout: each file fills the part of each window it covers.
    cell_idx: DataFrame with cell_row, cell_col. Returns (None, None) if the source has no files."""
    files = sorted(root.glob(src["pattern"]))
    if not files:
        return None, None
    n = len(cell_idx)
    r0 = np.empty(n, int)
    c0 = np.empty(n, int)
    for k, (i, j) in enumerate(zip(cell_idx.cell_row, cell_idx.cell_col)):
        r0[k], c0[k], _ = cells.window(i, j)
    names = [src["use"][b][0] for b in src["file_bands"] if b in src["use"]]
    out = np.full((n, len(names), 3, 3), np.nan)
    covered = np.zeros((n, 3, 3), bool)
    for f in files:
        with rasterio.open(f) as ds:
            assert ds.res == (cells.PIX, cells.PIX), f"{f.name}: pixel size {ds.res}, expected 10 m"
            assert "32650" in str(ds.crs), f"{f.name}: crs {ds.crs}, expected EPSG:32650"
            ro = (cells.Y0 - ds.transform.f) / cells.PIX
            co = (ds.transform.c - cells.X0) / cells.PIX
            assert ro == round(ro) and co == round(co), f"{f.name}: not on the 10 m grid"
            ro, co = int(ro), int(co)
            if ds.count != len(src["file_bands"]):
                raise SystemExit(f"{f.name}: {ds.count} bands, expected {len(src['file_bands'])}")
            if any(ds.descriptions) and list(ds.descriptions) != src["file_bands"]:
                raise SystemExit(f"{f.name}: band names {ds.descriptions} differ from {src['file_bands']}")
            sel = [(b, src["use"][b][1]) for b in src["file_bands"] if b in src["use"]]
            idx = [src["file_bands"].index(b) + 1 for b, _ in sel]
            scale = np.array([s for _, s in sel], float)[:, None, None]
            # cells whose window touches this file
            hit = np.nonzero((r0 + 3 > ro) & (r0 < ro + ds.height) & (c0 + 3 > co) & (c0 < co + ds.width))[0]
            hit = hit[np.argsort(r0[hit], kind="stable")]
            strip_of = (r0[hit] - ro) // STRIP
            for s in np.unique(strip_of):
                ks = hit[strip_of == s]
                ra, rb = max(r0[ks].min(), ro), min(r0[ks].max() + 3, ro + ds.height)
                ca, cb = max(c0[ks].min(), co), min(c0[ks].max() + 3, co + ds.width)
                data = ds.read(idx, window=((ra - ro, rb - ro), (ca - co, cb - co)))     # raw, stored dtype (small)
                for k in ks:
                    a0, a1 = max(r0[k], ro), min(r0[k] + 3, ro + ds.height)
                    b0, b1 = max(c0[k], co), min(c0[k] + 3, co + ds.width)
                    assert not covered[k, a0 - r0[k]:a1 - r0[k], b0 - c0[k]:b1 - c0[k]].any(), \
                        f"pieces of {src['name']} overlap at cell {k}"
                    blk = data[:, a0 - ra:a1 - ra, b0 - ca:b1 - ca].astype("float64")    # only the cell's pixels
                    blk[blk == src["nodata"]] = np.nan                                   # nodata first ...
                    out[k, :, a0 - r0[k]:a1 - r0[k], b0 - c0[k]:b1 - c0[k]] = blk / scale   # ... then the scale
                    covered[k, a0 - r0[k]:a1 - r0[k], b0 - c0[k]:b1 - c0[k]] = True
    # every window pixel must lie inside some file: a missing piece must not pass as "no data"
    # (nodata INSIDE a file is legitimate, e.g. outside the mangrove zone; absence of the file is not)
    n_bad = int((~covered.all((1, 2))).sum())
    if n_bad:
        raise SystemExit(f"source '{src['name']}': {n_bad} cells have window pixels that no file covers "
                         f"(missing or incomplete piece? {len(files)} files found)")
    return out, names


def extract(cell_idx, sources, root):
    """Channel table (one row per cell, in the order of cell_idx) for every source that has files."""
    n = len(cell_idx)
    weights = [cells.window(i, j)[2] for i, j in zip(cell_idx.cell_row, cell_idx.cell_col)]
    cols = {}
    for src in sources:
        blocks, names = read_source(src, cell_idx, root)
        if blocks is None:
            print(f"  source '{src['name']}': no files for {src['pattern']} (skipped)")
            continue
        vals = np.array([window_mean(blocks[k], weights[k]) for k in range(n)])
        for c, name in enumerate(names):
            cols[name] = vals[:, c]
        print(f"  source '{src['name']}': {len(names)} channels ({', '.join(names)})")
    ordered = [c for c in CHANNELS if c in cols]
    return pd.DataFrame({c: cols[c] for c in ordered}, index=cell_idx.index)


def norm_stats(table, channels):
    """Per-channel mean / std (ddof = 0) / valid count from the TRAIN rows only; NaNs ignored."""
    tr = table[table["set"] == "train"]
    stats = {}
    for c in channels:
        v = tr[c].to_numpy(float)
        v = v[np.isfinite(v)]
        std = float(v.std())
        assert v.size > 0 and std > 0, f"channel {c}: no valid train values or zero std"
        stats[c] = {"mean": float(v.mean()), "std": std, "n_valid": int(v.size)}
    return stats


def load_cells(labels=LABELS):
    c = pd.read_csv(labels / "gedi_cells.csv")
    sp = pd.read_csv(labels / "gedi_split.csv")
    assert not c.duplicated(["cell_row", "cell_col"]).any(), "duplicate cell in gedi_cells.csv"
    assert len(sp) == len(c), "split and cells have different numbers of rows"
    m = c.merge(sp, on=["cell_row", "cell_col"], how="left", validate="1:1")
    assert len(m) == len(c) and m["set"].notna().all(), "cells and split do not match"
    win = [cells.window(i, j)[:2] for i, j in zip(m.cell_row, m.cell_col)]
    assert (np.array(win) == m[["win_row0", "win_col0"]].to_numpy()).all(), "window columns disagree with cells.py"
    return m


def main(sources=SOURCES, root=ROOT, labels=LABELS, out=OUT):
    m = load_cells(labels)
    print(f"{len(m):,} cells; split: {m['set'].value_counts().to_dict()}")
    feats = extract(m[["cell_row", "cell_col"]], sources, root)
    # cell_row / cell_col stay in the table only as keys (they are location proxies): models must select
    # their inputs with `channels.CHANNELS`, never "every column that is not metadata".
    table = pd.concat([m[META], feats], axis=1)
    partial = len(feats.columns) < len(CHANNELS)
    tag = "_partial" if partial else ""
    stats = norm_stats(table, list(feats.columns))

    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / f"norm_stats{tag}.json"
    if not partial and stats_path.exists():
        sys.exit(f"{stats_path} already exists: the final statistics are write-once. Delete it only on purpose.")
    meta = {"channels": list(feats.columns), "partial": partial, "ddof": 0, "n_train_cells": int((table.set == "train").sum()),
            "split_sha256": hashlib.sha256((labels / "gedi_split.csv").read_bytes()).hexdigest(), "stats": stats}
    stats_path.write_text(json.dumps(meta, indent=1))               # statistics first, then the table
    table.to_csv(out / f"cell_features{tag}.csv", index=False)

    print(f"\nwrote {out / f'cell_features{tag}.csv'} ({len(table):,} rows, {len(feats.columns)} of {len(CHANNELS)} channels)")
    print(f"wrote {stats_path}" + ("  [PARTIAL: not the final statistics]" if partial else ""))
    print("\nchannel   missing cells   train mean      train std")
    for c in feats.columns:
        print(f"{c:9s} {int(feats[c].isna().sum()):9d}   {stats[c]['mean']:12.4f}  {stats[c]['std']:12.4f}")
    return table


if __name__ == "__main__":
    main()
