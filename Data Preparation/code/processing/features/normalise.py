"""Normalise channel values with the stored TRAIN-only statistics and fill missing values (Project.md 2.4).

One function for all four models, so they treat inputs identically:
    z = (value - train mean) / train std,   then   missing (NaN) -> 0, i.e. the training mean.
Decided 2026-11-24: the 89 label cells with no PALSAR pixel (and about 1% of mangrove pixels at mapping time)
get the training mean for the missing channels. The split, the label set and the cells are not changed.

Usage:
    from normalise import normalise
    X = normalise(table_or_pixels_as_dataframe, "data/processed/features/norm_stats.json")
Columns are taken from the statistics file, in its (CHANNELS) order; other columns are ignored.
"""
import json
from pathlib import Path

import pandas as pd


def load_stats(path):
    meta = json.loads(Path(path).read_text())
    return meta["channels"], meta["stats"]


def normalise(df, stats_path):
    """Z-scores of the channels listed in the statistics file; NaN becomes 0 (the training mean)."""
    channels, stats = load_stats(stats_path)
    missing = [c for c in channels if c not in df.columns]
    assert not missing, f"channels missing from the data: {missing}"
    mean = pd.Series({c: stats[c]["mean"] for c in channels})
    std = pd.Series({c: stats[c]["std"] for c in channels})
    return ((df[channels] - mean) / std).fillna(0.0)
