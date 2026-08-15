"""Node splits, plus the degeneracy check that catches the id-ordering trap.

`generate` appends ring members at the end of the id range, so a contiguous
split puts every fraud node on one side. `contiguous_split` exists only so the
test suite and `experiments/degenerate_split_demo.py` can demonstrate that.
"""

from __future__ import annotations

import numpy as np


def stratified_split(
    y: np.ndarray, seed: int = 0, frac_train: float = 0.6, frac_val: float = 0.2
):
    """Class-stratified random split. Returns (train_idx, val_idx, test_idx)."""
    if not 0 < frac_train < 1 or not 0 <= frac_val < 1 or frac_train + frac_val >= 1:
        raise ValueError("need frac_train + frac_val < 1 with both in range")
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        a = int(round(frac_train * len(idx)))
        b = a + int(round(frac_val * len(idx)))
        for bucket, chunk in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            bucket.append(chunk)
    return tuple(np.sort(np.concatenate(p)) for p in parts)


def contiguous_split(
    n: int, frac_train: float = 0.6, frac_val: float = 0.2
):
    """Split by id order. Degenerate on graphs from `generate` -- that is the point."""
    a = int(round(frac_train * n))
    b = a + int(round(frac_val * n))
    return np.arange(a), np.arange(a, b), np.arange(b, n)


def degenerate_classes(y: np.ndarray, *splits) -> list[int]:
    """Indices of splits that are missing at least one class present overall."""
    all_classes = set(np.unique(y).tolist())
    return [
        i for i, s in enumerate(splits) if len(s) == 0 or set(np.unique(y[s]).tolist()) != all_classes
    ]


def assert_not_degenerate(y: np.ndarray, *splits) -> None:
    """Raise if any split is missing a class. Call this before training."""
    bad = degenerate_classes(y, *splits)
    if bad:
        counts = [dict(zip(*[a.tolist() for a in np.unique(y[s], return_counts=True)])) for s in splits]
        raise ValueError(f"degenerate split(s) at position {bad}: per-split class counts {counts}")
