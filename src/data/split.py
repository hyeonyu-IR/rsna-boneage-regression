from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    val_frac: float = 0.15
    stratify_age_bins: int = 20
    seed: int = 42


def make_train_val_split(
    train_df: pd.DataFrame,
    cfg: SplitConfig,
    label_col: str = "boneage",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_idx, val_idx) indices into train_df.

    Strategy:
      - bin regression target into quantile bins
      - stratified split on those bins
      - fallback to random split if binning fails
    """
    if label_col not in train_df.columns:
        raise ValueError(f"label_col '{label_col}' not found. Columns={list(train_df.columns)}")

    y = train_df[label_col].astype(float).to_numpy()

    n_bins = max(int(cfg.stratify_age_bins), 2)
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(train_df))

    bins_series = None

    for b in range(n_bins, 1, -1):
        try:
            tmp = pd.qcut(y, q=b, labels=False, duplicates="drop")
            if int(tmp.nunique(dropna=True)) >= 2:
                bins_series = tmp
                break
        except Exception:
            continue

    if bins_series is None:
        rng.shuffle(idx)
        n_val = int(round(len(idx) * cfg.val_frac))
        return idx[n_val:], idx[:n_val]

    bins = bins_series.to_numpy()

    train_idx = []
    val_idx = []

    for bin_id in np.unique(bins):
        members = idx[bins == bin_id]
        rng.shuffle(members)
        n_val = int(round(len(members) * cfg.val_frac))
        val_idx.append(members[:n_val])
        train_idx.append(members[n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx
